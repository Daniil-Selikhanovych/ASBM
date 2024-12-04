# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import os
import glob
from multiprocessing import cpu_count
import argparse

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torch

from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from pytorch_fid.fid_score import calculate_frechet_distance

from metrics import compute_statistics_of_path_or_dataloader, calculate_cost
from datasets_prep.paired_colored_mnist import load_paired_colored_mnist

import torchvision.transforms as transforms
from datasets_prep.faces2comics import Faces2Comics
from datasets_prep.celeba import Celeba

from pytorch_fid.inception import InceptionV3

from sampling_utils import sample_from_model
from posteriors import BrownianPosterior_Coefficients, get_time_schedule, Posterior_Coefficients


def save_and_log_images(test_data_loader, device, args, fw_or_bw, pos_coeff, netG,
                        num_of_samples_for_trajectories_to_visualize, exp_path,
                        iteration, num_trajectories):
    for (x, y) in test_data_loader:
        x_0 = x.to(device)
        x_t_1 = y.to(device)
        break

    print(f"save_and_log_images, mode = {fw_or_bw}, x_t_1.shape = {x_t_1.shape}")
    # fake_sample in [-1, 1]
    # trajectory = [x_t_1, ...]
    # torchvision.utils.save_image(fake_sample, os.path.join(exp_path, 'sample_discrete_epoch_{}.png'.format(epoch)), normalize=True)
    fake_sample, trajectory = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, args,
                                                return_trajectory=True)

    trajectory_to_visualize = []
    for i in range(args.num_timesteps + 1):
        trajectory_to_visualize.append(trajectory[i][:num_of_samples_for_trajectories_to_visualize])

    trajectory_to_visualize = torch.stack(trajectory_to_visualize)

    trajectory_to_visualize_numpy = 0.5 * (trajectory_to_visualize + 1)
    trajectory_to_visualize_numpy = trajectory_to_visualize_numpy.mul(255) \
        .add_(0.5).clamp_(0, 255).permute(0, 1, 3, 4, 2).to("cpu", torch.uint8).numpy()
    # trajectory_to_visualize_numpy.shape = (args.num_timesteps+1, 8, H, W, C)
    fig, ax = plt.subplots(num_of_samples_for_trajectories_to_visualize, args.num_timesteps + 2,
                           figsize=(2 * args.num_timesteps, 2 * 6))

    x_0_to_visualize_numpy = 0.5 * (x_0 + 1)
    x_0_to_visualize_numpy = x_0_to_visualize_numpy.mul(255) \
        .add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

    for k in range(num_of_samples_for_trajectories_to_visualize):
        for j in range(args.num_timesteps + 2):
            ax[0][j].xaxis.tick_top()
            if j == 0:
                ax[k][j].imshow(trajectory_to_visualize_numpy[0, k])
                ax[0][j].set_title(f'Input, T = {args.num_timesteps}')
            elif j <= args.num_timesteps:
                ax[k][j].imshow(trajectory_to_visualize_numpy[j, k])
                ax[0][j].set_title(f'T = {args.num_timesteps - j}')
            else:
                ax[k][j].imshow(x_0_to_visualize_numpy[k])
                ax[0][j].set_title(f'Real data')
            ax[k][j].xaxis.tick_top()
            # ax[k][j].get_xaxis().set_visible(False)
            ax[k][j].set_xticks([])
            ax[k][j].get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.001)
    path_to_save_figures = os.path.join(exp_path,
                                        f'evolution_step_{iteration}_mode_{fw_or_bw}_is_train_False.png')

    plt.savefig(path_to_save_figures)
    print(f"saving {path_to_save_figures}")
    img_to_tensorboard = Image.open(path_to_save_figures)
    img_to_tensorboard = transforms.ToTensor()(img_to_tensorboard)[:3]

    num_starts = x_t_1.shape[0]
    print(f"batch size = {num_starts}")

    print(f"num_of_samples_for_trajectories_to_visualize = {num_of_samples_for_trajectories_to_visualize}")

    trajectory_to_visualize_same_start = []
    for num_traj in range(num_trajectories):
        torch.manual_seed(num_traj)
        fake_sample, trajectory = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, args,
                                                    return_trajectory=True)
        
        # print(f"fake sample shape = {fake_sample.shape}")
        # fake_sample = [[B, C, H, W], .... ] T+1 times
        # first_sample_dynamic = [trajectory[i][:1] for i in range(args.num_timesteps + 1)]
        first_sample_dynamic = [trajectory[i][:num_of_samples_for_trajectories_to_visualize] for i in range(args.num_timesteps + 1)]
        # first_sample_dynamic = torch.stack(first_sample_dynamic).permute(1, 0, 2, 3, 4)  # [T + 1, B, C, H, W]
        first_sample_dynamic = torch.stack(first_sample_dynamic) # [T + 1, B, C, H, W]
        # assert first_sample_dynamic.shape[0] == 1
        # first_sample_dynamic = first_sample_dynamic[0]  # [T + 1, C, H, W]
        # first_sample_dynamic = first_sample_dynamic[[0, -1]]  # [2, B, C, H, W]
        first_sample_dynamic = first_sample_dynamic # [T + 1, B, C, H, W]
        # print(f"append array {first_sample_dynamic.shape}")

        # trajectory_to_visualize_same_start.append(first_sample_dynamic)  # [T + 1, C, H, W]
        trajectory_to_visualize_same_start.append(first_sample_dynamic)  # [2, B, C, H, W] / [T + 1, B, C, H, W]

    # trajectory_to_visualize_same_start = torch.stack(trajectory_to_visualize_same_start)  # [N, T + 1, C, H, W]
    trajectory_to_visualize_same_start = torch.stack(trajectory_to_visualize_same_start)  # [num_trajectories, 2, B, C, H, W] / [num_trajectories, T + 1, B, C, H, W]
    trajectory_to_visualize_same_start_numpy_raw = 0.5 * (trajectory_to_visualize_same_start + 1)
    # trajectory_to_visualize_same_start_numpy = trajectory_to_visualize_same_start_numpy.mul(255) \
    #     .add_(0.5).clamp_(0, 255).permute(0, 1, 3, 4, 2).to("cpu", torch.uint8).numpy()
    trajectory_to_visualize_same_start_numpy =  trajectory_to_visualize_same_start_numpy_raw.mul(255) \
          .add_(0.5).clamp_(0, 255).permute(0, 1, 2, 4, 5, 3).to("cpu", torch.uint8).numpy()
    
    print(f"trajectory_to_visualize_same_start_numpy.shape = {trajectory_to_visualize_same_start_numpy.shape}")

    fig, ax = plt.subplots(num_of_samples_for_trajectories_to_visualize, num_trajectories + 1,
                           figsize=(2 * num_trajectories, 2 * 6))
    
    tensor_starts = trajectory_to_visualize_same_start_numpy_raw[0, 0].cpu().detach()
    # tensor_outputs = trajectory_to_visualize_same_start_numpy_raw[:, 1].permute((1, 0, 2, 3, 4)).cpu().detach()
    tensor_outputs = trajectory_to_visualize_same_start_numpy_raw[:, 1:].permute((2, 1, 0, 3, 4, 5)).cpu().detach()
    # [B, num_trajectories, C, H, W]

    print(f"shape of saved tensors start {tensor_starts.shape}")
    print(f"shape of saved tensors predictions {tensor_outputs.shape}")

    path_to_save_input_tensors = os.path.join(exp_path,
                                     f'tensor_inputs_step_{iteration}_mode_{fw_or_bw}_fixed_noise_trajectories_teaser.pt')
    torch.save(tensor_starts, path_to_save_input_tensors)
    path_to_save_predicted_tensors = os.path.join(exp_path,
                                     f'tensor_predictions_step_{iteration}_mode_{fw_or_bw}_fixed_noise_trajectories_teaser.pt')
    torch.save(tensor_outputs, path_to_save_predicted_tensors)

    # for k in range(num_trajectories):
    for k in range(num_of_samples_for_trajectories_to_visualize):
        # for j in range(args.num_timesteps + 2):
        for j in range(num_trajectories + 1):
            ax[0][j].xaxis.tick_top()
            if j == 0:
                # ax[k][j].imshow(trajectory_to_visualize_same_start_numpy[k, 0])
                ax[k][j].imshow(trajectory_to_visualize_same_start_numpy[0, 0, k])
                # ax[0][j].set_title(f'Input, T = {args.num_timesteps}')
                ax[0][j].set_title(f'Input')
            # elif j <= args.num_timesteps:
            #     ax[k][j].imshow(trajectory_to_visualize_same_start_numpy[k, j])
            #     ax[0][j].set_title(f'T = {args.num_timesteps - j}')
            # else:
            #     ax[k][j].imshow(x_0_to_visualize_numpy[0])
            #     ax[0][j].set_title(f'Real data')
            else:
                 ax[k][j].imshow(trajectory_to_visualize_same_start_numpy[j - 1, 1, k])
                 ax[0][j].set_title(f'Sample {j}')
            ax[k][j].xaxis.tick_top()
            # ax[k][j].get_xaxis().set_visible(False)
            ax[k][j].set_xticks([])
            ax[k][j].get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.001)
    # path_to_save_figures = os.path.join(exp_path,
    #                                 f'evolution_many_trajectories_step_{iteration}_mode_{fw_or_bw}_is_train_False.png')
    path_to_save_figures = os.path.join(exp_path,
                                     f'evolution_many_trajectories_step_{iteration}_mode_{fw_or_bw}_is_train_False_only_outputs_fixed_noise.png')
    
    plt.savefig(path_to_save_figures)
    print(f"saving {path_to_save_figures}")


# %%
def sample_and_test(args):
    torch.manual_seed(42)
    device = 'cuda:0'

    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celeba_256_stat.npy'
    elif args.dataset == 'lsun':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    elif args.dataset == 'paired_colored_mnist':
        train_dataset, test_dataset = load_paired_colored_mnist()
        print(f"Num images in train = {len(train_dataset)}")
        print(f"Num images in test = {len(test_dataset)}")
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=False,
                                                        drop_last=False,
                                                        num_workers=cpu_count())
        test_dataloader_input = torch.utils.data.DataLoader(test_dataset,
                                                         batch_size=args.batch_size,
                                                         shuffle=False,
                                                         drop_last=False,
                                                         num_workers=cpu_count())
        test_dataloader_target = torch.utils.data.DataLoader(test_dataset,
                                                          batch_size=args.batch_size,
                                                          shuffle=False,
                                                          drop_last=False,
                                                          num_workers=cpu_count())

    elif args.dataset == "faces2comics":
        print(f"Preparing faces2comics dataset")
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        data_root = args.data_root
        paired = args.paired
        print(f"data in {data_root}, paired = {paired}")

        train_dataset = Faces2Comics(data_root, train=True, transform=transform, part_test=0.1, paired=paired)
        print(f"Num train images = {len(train_dataset)}")
        test_dataset = Faces2Comics(data_root, train=False, transform=transform, part_test=0.1, paired=True)
        print(f"Num test images = {len(test_dataset)}")
        test_dataloader_input = torch.utils.data.DataLoader(test_dataset,
                                                         batch_size=args.batch_size,
                                                         shuffle=False,
                                                         drop_last=False,
                                                         num_workers=cpu_count())
        test_dataloader_target = torch.utils.data.DataLoader(test_dataset,
                                                         batch_size=args.batch_size,
                                                         shuffle=False,
                                                         drop_last=False,
                                                         num_workers=cpu_count())

    elif args.dataset == 'celeba_celeba_female_to_male':
        print(f"Preparing celeba male2female")
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        data_root = args.data_root
        train_dataset_input = Celeba(data_root, train=True, transform=transform, part_test=0.1, mode="male_to_female")
        print(f"Num train images input = {len(train_dataset_input)}")
        train_dataset_target = Celeba(data_root, train=True, transform=transform, part_test=0.1, mode="female_to_male")
        print(f"Num train images target = {len(train_dataset_target)}")
        test_dataset_input_fw = Celeba(data_root, train=False, transform=transform, part_test=0.1, mode="male_to_female")
        # (x, y) in dataloader, (female_img, male_img), indexing for males
        test_dataset_target_fw = Celeba(data_root, train=False, transform=transform, part_test=0.1, mode="female_to_male",
                                     input_is_second=False)

        test_dataset_input_bw = Celeba(data_root, train=False, transform=transform, part_test=0.1, mode="female_to_male")
        # (x, y) in dataloader, (female_img, male_img), indexing for males
        test_dataset_target_bw = Celeba(data_root, train=False, transform=transform, part_test=0.1, mode="male_to_female",
                                        input_is_second=False)

        # (x, y) in dataloader, (female_img, male_img), indexing for female
        print(
            f"Forward, num of input test images = {len(test_dataset_input_fw)}, "
            f"num of target test images = {len(test_dataset_target_fw)}")
        print(
            f"Backward, num of input test images = {len(test_dataset_input_bw)}, "
            f"num of target test images = {len(test_dataset_target_bw)}")
        test_dataloader_input_fw = torch.utils.data.DataLoader(test_dataset_input_fw,
                                                               batch_size=args.batch_size,
                                                               shuffle=False,
                                                               drop_last=False,
                                                               num_workers=cpu_count())
        test_dataloader_target_fw = torch.utils.data.DataLoader(test_dataset_target_fw,
                                                                batch_size=args.batch_size,
                                                                shuffle=False,
                                                                drop_last=False,
                                                                num_workers=cpu_count())
        test_dataloader_input_bw = torch.utils.data.DataLoader(test_dataset_input_bw,
                                                               batch_size=args.batch_size,
                                                               shuffle=False,
                                                               drop_last=False,
                                                               num_workers=cpu_count())
        test_dataloader_target_bw = torch.utils.data.DataLoader(test_dataset_target_bw,
                                                                batch_size=args.batch_size,
                                                                shuffle=False,
                                                                drop_last=False,
                                                                num_workers=cpu_count())


    save_dir_gt_fid_statistics = "./fid_results_test_imf/{}/{}".format(args.dataset, args.exp)
    os.makedirs(save_dir_gt_fid_statistics, exist_ok=True)
    path_to_save_gt_fid_statistics_forward = os.path.join(save_dir_gt_fid_statistics, "gt_statistics_forward.npz")
    path_to_save_gt_fid_statistics_backward = os.path.join(save_dir_gt_fid_statistics, "gt_statistics_backward.npz")

    compute_fid = True

    if compute_fid:
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model_fid = InceptionV3([block_idx]).to(device)

        mode = "features_gt"
        # "features_gt" -> (x, y) in dataloader, compute features of x
        # "features_pred" -> (x, y) in dataloader, comput features of f(y)
        m1_test_gt_for, s1_test_gt_for, _ = compute_statistics_of_path_or_dataloader(path_to_save_gt_fid_statistics_forward,
                                                                        test_dataloader_target_fw, model_fid,
                                                                        args.batch_size, dims, device,
                                                                        mode, None, None, None)
        m1_test_gt_back, s1_test_gt_back, _ = compute_statistics_of_path_or_dataloader(path_to_save_gt_fid_statistics_backward,
                                                                        test_dataloader_target_bw, model_fid,
                                                                        args.batch_size, dims, device,
                                                                        mode, None, None, None)
        
        if not os.path.exists(path_to_save_gt_fid_statistics_forward):
            print(f"Forward, saving stats for gt test data to {path_to_save_gt_fid_statistics_forward}")
            np.savez(path_to_save_gt_fid_statistics_forward, mu=m1_test_gt_for, sigma=s1_test_gt_for)

        if not os.path.exists(path_to_save_gt_fid_statistics_backward):
            print(f"Backward, saving stats for gt test data to {path_to_save_gt_fid_statistics_backward}")
            np.savez(path_to_save_gt_fid_statistics_backward, mu=m1_test_gt_back, sigma=s1_test_gt_back)

    path_to_ckpts = f"./saved_info/dd_gan_imf/{args.dataset}/{args.exp}"

    for fw_or_bw in ['bw', 'fw']:
        print(f"will evaluate ckpts in mode {fw_or_bw}")

        # num_steps = [i for i in range(1, args.num_timesteps + 1)]
        num_steps = [args.num_timesteps]
        fid_vals = {num_step: [] for num_step in num_steps}
        l2_cost_vals = {num_step: [] for num_step in num_steps}

        num_of_samples_for_trajectories_to_visualize = 100
        num_trajectories = 20

        all_content_ckpts = sorted(glob.glob(os.path.join(path_to_ckpts, f"content_{fw_or_bw}_imf_num_iter_*_0.pth")))

        all_basenames = sorted([os.path.basename(path) for path in all_content_ckpts])
        all_ckpts_iters = sorted([int(basename.split(".")[0].split("_")[-2]) for basename in all_basenames])
        print(f"all_basenames = {all_basenames}")

        all_ckpts_basenames = [f"content_{fw_or_bw}_imf_num_iter_{iteration}_0.pth" for iteration in all_ckpts_iters]
        print(f"all_ckpts_basenames = {all_ckpts_basenames}")

        if compute_fid:
            min_fid = {num_step: np.inf for num_step in num_steps}
            min_epoch = {num_step: all_ckpts_iters[0] for num_step in num_steps}
            l2_on_min_fid = {num_step: np.inf for num_step in num_steps}

            min_l2 = {num_step: np.inf for num_step in num_steps}
            min_l2_epoch = {num_step: all_ckpts_iters[0] for num_step in num_steps}
            fid_on_min_l2 = {num_step: np.inf for num_step in num_steps}

            dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            model_fid = InceptionV3([block_idx]).to(device)

            fig_fid, ax_fid = plt.subplots(figsize=(12, 12))
            fig_l2_cost, ax_l2_cost = plt.subplots(figsize=(12, 12))

        num_ckpts = 100500
        print(f"num ckpts to evaluate = {num_ckpts}")

        for iteration, ckpt_epoch in zip(all_ckpts_iters[:num_ckpts], all_ckpts_basenames[:num_ckpts]):
            ckpt_path = os.path.join(path_to_ckpts, ckpt_epoch)
            print(F"evaluate {ckpt_path}")
            # netG = NCSNpp(args).to(device)
            ckpt = torch.load(ckpt_path,
                            map_location=device)
            

            args_loaded = ckpt['args']
            print(f"args loaded = {args_loaded}")
            netG = NCSNpp(args_loaded).to(device)

            # print(f"ckpt.keys = {list(ckpt.keys())}")
            ckpt_netG = ckpt['netG_dict']
            for key in list(ckpt_netG.keys()):
                ckpt_netG[key[7:]] = ckpt_netG.pop(key)
            netG.load_state_dict(ckpt_netG)
            netG.eval()

            # global_step = ckpt['global_step']
            # epoch = ckpt['epoch']

            print(f"net was loaded!")

            for num_step in num_steps:
                print(f"will evaluate ckpts with num step = {num_step}")
                args.num_timesteps = num_step
                T = get_time_schedule(args, device)

                # pos_coeff = Posterior_Coefficients(args, device)
                if args.posterior == "ddpm":
                    pos_coeff = Posterior_Coefficients(args, device)
                elif args.posterior == "brownian_bridge":
                    pos_coeff = BrownianPosterior_Coefficients(args, device)

                save_dir_pred_fid_statistics = "{}/iteration_{}_t_{}_mode_{}".format(save_dir_gt_fid_statistics, 
                                                                                     iteration, num_step, fw_or_bw)
                if not os.path.exists(save_dir_pred_fid_statistics):
                    os.makedirs(save_dir_pred_fid_statistics)
                path_to_save_pred_fid_statistics = os.path.join(save_dir_pred_fid_statistics, "pred_statistics.npz")
                path_to_save_fid_txt = os.path.join(save_dir_pred_fid_statistics, "fid.txt")

                if fw_or_bw == 'fw':
                    save_and_log_images(test_dataloader_input_fw, device, args, fw_or_bw, pos_coeff,
                                netG, 
                                num_of_samples_for_trajectories_to_visualize, save_dir_pred_fid_statistics, iteration,
                                num_trajectories)
                else:
                    save_and_log_images(test_dataloader_input_bw, device, args, fw_or_bw, pos_coeff,
                                netG, 
                                num_of_samples_for_trajectories_to_visualize, save_dir_pred_fid_statistics, iteration,
                                num_trajectories)

                if not os.path.exists(path_to_save_fid_txt):
                    dims = 2048
                    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
                    model_fid = InceptionV3([block_idx]).to(device)

                    # for i in range(iters_needed):
                    mode = "features_pred"
                    if fw_or_bw == 'fw':
                        if compute_fid:
                            m1_test_pred, s1_test_pred, test_l2_cost = compute_statistics_of_path_or_dataloader(path_to_save_pred_fid_statistics,
                                                                                                test_dataloader_input_fw, model_fid,
                                                                                                args.batch_size, dims, device,
                                                                                                mode, netG, args,
                                                                                                pos_coeff)
                    else:
                        if compute_fid:
                            m1_test_pred, s1_test_pred, test_l2_cost = compute_statistics_of_path_or_dataloader(path_to_save_pred_fid_statistics,
                                                                                                test_dataloader_input_bw, model_fid,
                                                                                                args.batch_size, dims, device,
                                                                                                mode, netG, args,
                                                                                                pos_coeff)
                    
                    if compute_fid:
                        del model_fid
                        if fw_or_bw == 'fw':
                            fid_value_test = calculate_frechet_distance(m1_test_pred, s1_test_pred, m1_test_gt_for, s1_test_gt_for)
                        else:
                            fid_value_test = calculate_frechet_distance(m1_test_pred, s1_test_pred, m1_test_gt_back, s1_test_gt_back)
                        if not os.path.exists(path_to_save_pred_fid_statistics):
                            print(f"saving stats for pred on test data to {path_to_save_pred_fid_statistics}")
                            np.savez(path_to_save_pred_fid_statistics, mu=m1_test_pred, sigma=s1_test_pred)

                        with open(path_to_save_fid_txt, "w") as f:
                            f.write(f"FID = {fid_value_test}")

                        print(f"saving FID to {path_to_save_fid_txt}")
                else:
                    if compute_fid:
                        with open(path_to_save_fid_txt, "r") as f:
                            line = f.readlines()[0]
                            fid_value_test = float(line.split(" ")[-1])
                if compute_fid:
                    print(f'Test FID = {fid_value_test} on dataset {args.dataset} for exp {args.exp} and iteration = {iteration}')

                    fid_vals[num_step].append(fid_value_test)

                    path_to_save_cost_txt = os.path.join(save_dir_pred_fid_statistics, "l2_cost.txt")

                    if not os.path.exists(path_to_save_cost_txt):
                        with open(path_to_save_cost_txt, "w") as f:
                            f.write(f"L2 cost = {test_l2_cost}")

                        print(f"saving L2 cost to {path_to_save_cost_txt}")
                    else:
                        with open(path_to_save_cost_txt, "r") as f:
                            line = f.readlines()[0]
                            test_l2_cost = float(line.split(" ")[-1])
                    print(f'Test l2 cost = {test_l2_cost} on dataset {args.dataset} for exp {args.exp} and epoch = {iteration}')
                    l2_cost_vals[num_step].append(test_l2_cost)

                    if fid_value_test < min_fid[num_step]:
                        min_fid[num_step] = fid_value_test
                        min_epoch[num_step] = iteration
                        l2_on_min_fid[num_step] = test_l2_cost
                        print(f"Num step = {num_step}, new min FID = {fid_value_test} on epoch = {min_epoch[num_step]} with l2 cost = {test_l2_cost}")

                    if test_l2_cost < min_l2[num_step]:
                        min_l2[num_step] = test_l2_cost
                        min_l2_epoch[num_step] = iteration
                        fid_on_min_l2[num_step] = fid_value_test
                        print(f"Num step = {num_step}, new min l2 cost = {min_l2[num_step]} on epoch = {min_l2_epoch[num_step]} with FID = {fid_value_test}")


        if compute_fid:
            for num_step in num_steps:
                fid_vals[num_step] = np.array(fid_vals[num_step])
                ax_fid.plot(all_ckpts_iters, fid_vals[num_step], label=f"T = {num_step}, " + args.exp)
            ax_fid.set_title(f'FID, test, {args.dataset}, mode = {fw_or_bw}')
            ax_fid.set_xlabel("IMF iteration")
            ax_fid.set_ylabel("FID")
            ax_fid.legend()
            ax_fid.grid(True)

            path_to_save_figures = os.path.join(save_dir_gt_fid_statistics,
                                                f"test_fid_{args.dataset}_{args.exp}_mode_{fw_or_bw}.png")
            fig_fid.savefig(path_to_save_figures)
            print(f"saving fid graph in {path_to_save_figures}")

            for num_step in num_steps:
                l2_cost_vals[num_step] = np.array(l2_cost_vals[num_step])
                ax_l2_cost.plot(all_ckpts_iters, l2_cost_vals[num_step], label=f"T = {num_step}, " + args.exp)
            ax_l2_cost.set_title(f'L2 cost, test, {args.dataset}, mode = {fw_or_bw}')
            ax_l2_cost.set_xlabel("IMF iteration")
            ax_l2_cost.set_ylabel("Cost")
            ax_l2_cost.legend()
            ax_l2_cost.grid(True)

            path_to_save_figures = os.path.join(save_dir_gt_fid_statistics,
                                                f"test_l2_cost_{args.dataset}_{args.exp}_mode_{fw_or_bw}.png")
            fig_l2_cost.savefig(path_to_save_figures)
            print(f"saving fid graph in {path_to_save_figures}")

            for num_step in num_steps:
                print(f"Num step = {num_step}, experiment {args.exp}, dataset {args.dataset}, "
                    f"min FID = {min_fid[num_step]} on epoch = {min_epoch[num_step]} with l2 = {l2_on_min_fid[num_step]}")
                print(f"Num step = {num_step}, experiment {args.exp}, dataset {args.dataset}, "
                    f"min L2 cost = {min_l2[num_step]} on epoch = {min_l2_epoch[num_step]} with FID = {fid_on_min_l2[num_step]}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--posterior', type=str, default='ddpm',
                        help='type of posterior to use')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='variance of brownian bridge')
    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--data_root', default='')
    parser.add_argument('--paired', action='store_true', default=False)
    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')

    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')

    args = parser.parse_args()
    sample_and_test(args)
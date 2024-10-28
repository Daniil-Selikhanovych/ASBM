import os
import glob
import shutil

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.utils as tu
import torchvision.transforms as transforms
import torch.optim as optim

from torchvision.datasets import CIFAR10
from datasets_prep.lsun import LSUN
from datasets_prep.stackmnist_data import StackedMNIST, _data_transforms_stacked_mnist
from datasets_prep.lmdb_datasets import LMDBDataset
from datasets_prep.faces2comics import Faces2Comics
from torch.utils.data import TensorDataset
from datasets_prep.paired_colored_mnist import load_paired_colored_mnist
from datasets_prep.celeba import Celeba
from multiprocessing import cpu_count

from torch.multiprocessing import Process
import torch.distributed as dist

from metrics import calculate_cost, compute_statistics_of_path_or_dataloader

from sampling_utils import  q_sample_supervised_pairs, q_sample_supervised_pairs_brownian, \
    q_sample_supervised_trajectory, sample_posterior, sample_from_model
from posteriors import Diffusion_Coefficients, Posterior_Coefficients, \
    BrownianPosterior_Coefficients, get_time_schedule

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

from utils import TensorBoardWriter
from torch_ema import ExponentialMovingAverage


class CondLoaderSampler:
    def __init__(self, loader, reverse=False):
        self.loader = loader
        self.reverse = reverse
        self.it = iter(self.loader)

    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch_x, batch_y = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        except RuntimeError:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch_x) < size:
            return self.sample(size)

        if self.reverse:
            return batch_y[:size], batch_x[:size]

        return batch_x[:size], batch_y[:size]


class XSampler:
    def __init__(self, sampler: CondLoaderSampler):
        self.sampler = sampler

    def sample(self, size=5):
        return self.sampler.sample(size)[0]


class ModelCondSampler:
    def __init__(self, sampler: XSampler, model_sample_fn, ema_model=None):
        self.model_sample_fn = model_sample_fn
        self.sampler = sampler
        self.ema_model = ema_model
        if self.ema_model is not None:
            print(f"Will use EMA for ModelCondSampler sampling")
        else:
            print(f"Will not use EMA for ModelCondSampler sampling")

    def sample(self, size=5):
        sample_x = self.sampler.sample(size)
        if self.ema_model is not None:
            with self.ema_model.average_parameters():
                sample_y = self.model_sample_fn(sample_x)
        else:
            sample_y = self.model_sample_fn(sample_x)
        return sample_x, sample_y


def save_and_log_images(condsampler_gt, device, args, use_ema, pos_coeff, netG, 
                        batch_size, iteration, writer, num_of_samples_for_trajectories_to_visualize, exp_path,
                        num_trajectories, imf_num_iter, fw_or_bw):
    x, y = condsampler_gt.sample(args.batch_size)

    x_0, x_t_1 = x.to(device), y.to(device)

    # x_t_1.shape = (B, C, H, W)
    # print(f"x_0.shape ={x_0.shape}, x_t_1.shape = {x_t_1.shape}")

    # x_t_1 = torch.randn_like(real_data)
    # fake_sample in [-1, 1]
    # trajectory = [x_t_1, ...]
    fake_sample, trajectory = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, args,
                                                return_trajectory=True)
    if writer is not None:
        writer.add_image(iteration, f"sample_discrete_imf_iter_{imf_num_iter}_mode_{fw_or_bw}",
                         tu.make_grid(fake_sample, nrow=int(batch_size ** 0.5), normalize=True))

    trajectory_to_visualize = []
    for i in range(args.num_timesteps + 1):
        if writer is not None:
            writer.add_image(iteration, f"gen_x_{args.num_timesteps - i}_imf_iter_{imf_num_iter}_mode_{fw_or_bw}",
                             tu.make_grid(trajectory[i], nrow=int(batch_size ** 0.5), normalize=True))
        trajectory_to_visualize.append(trajectory[i][:num_of_samples_for_trajectories_to_visualize])

    trajectory_to_visualize = torch.stack(trajectory_to_visualize)
    print(f"trajectory_to_visualize.shape = {trajectory_to_visualize.shape}")

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
                ax[0][j].set_title(f'GT')
            ax[k][j].xaxis.tick_top()
            # ax[k][j].get_xaxis().set_visible(False)
            ax[k][j].set_xticks([])
            ax[k][j].get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.001)
    path_to_save_figures = os.path.join(exp_path,
                                        f'evolution_step_imf_iter_{imf_num_iter}_mode_{fw_or_bw}.png')
    print(f"saving {path_to_save_figures}")
    plt.savefig(path_to_save_figures)
    img_to_tensorboard = Image.open(path_to_save_figures)
    img_to_tensorboard = transforms.ToTensor()(img_to_tensorboard)[:3]
    if writer is not None:
        writer.add_image(iteration, f"evolution_different_starts_trajectories_imf_iter_{imf_num_iter}_mode_{fw_or_bw}",
                         img_to_tensorboard)

    trajectory_to_visualize_same_start = []
    for num_traj in range(num_trajectories):
        if use_ema:
            print(f"Sampling for the same start from EMA with decay = {args.ema_decay}")
        else:
            print(f"Sampling for the same start from single G")
        fake_sample, trajectory = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, args,
                                                    return_trajectory=True)
        # fake_sample = [[B, C, H, W], .... ] T+1 times
        first_sample_dynamic = [trajectory[i][:1] for i in range(args.num_timesteps + 1)]
        first_sample_dynamic = torch.stack(first_sample_dynamic).permute(1, 0, 2, 3, 4)  # [T + 1, 1, C, H, W]
        assert first_sample_dynamic.shape[0] == 1
        first_sample_dynamic = first_sample_dynamic[0]  # [T + 1, C, H, W]

        trajectory_to_visualize_same_start.append(first_sample_dynamic)  # [T + 1, C, H, W]

    trajectory_to_visualize_same_start = torch.stack(trajectory_to_visualize_same_start)  # [N, T + 1, C, H, W]
    trajectory_to_visualize_same_start_numpy = 0.5 * (trajectory_to_visualize_same_start + 1)
    trajectory_to_visualize_same_start_numpy = trajectory_to_visualize_same_start_numpy.mul(255) \
        .add_(0.5).clamp_(0, 255).permute(0, 1, 3, 4, 2).to("cpu", torch.uint8).numpy()

    for k in range(num_trajectories):
        for j in range(args.num_timesteps + 2):
            ax[0][j].xaxis.tick_top()
            if j == 0:
                ax[k][j].imshow(trajectory_to_visualize_same_start_numpy[k, 0])
                ax[0][j].set_title(f'Input, T = {args.num_timesteps}')
            elif j <= args.num_timesteps:
                ax[k][j].imshow(trajectory_to_visualize_same_start_numpy[k, j])
                ax[0][j].set_title(f'T = {args.num_timesteps - j}')
            else:
                ax[k][j].imshow(x_0_to_visualize_numpy[0])
                ax[0][j].set_title(f'GT')
            ax[k][j].xaxis.tick_top()
            # ax[k][j].get_xaxis().set_visible(False)
            ax[k][j].set_xticks([])
            ax[k][j].get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.001)
    path_to_save_figures = os.path.join(exp_path,
                                        f'evolution_many_trajectories_imf_iter_{imf_num_iter}_mode_{fw_or_bw}.png')
    plt.savefig(path_to_save_figures)
    print(f"saving {path_to_save_figures}")
    img_to_tensorboard = Image.open(path_to_save_figures)
    img_to_tensorboard = transforms.ToTensor()(img_to_tensorboard)[:3]
    if writer is not None:
        writer.add_image(iteration, f"evolution_same_start_trajectories_imf_iter_{imf_num_iter}_mode_{fw_or_bw}",
                         img_to_tensorboard)

    trajectory = q_sample_supervised_trajectory(pos_coeff, x_0, x_t_1)
    # x_start = x_0, x_end = x_t_1
    # trajectory = [x_t_1, ...]
    if writer is not None:
        for i in range(args.num_timesteps + 1):
            writer.add_image(iteration, f"x_{args.num_timesteps - i}_imf_iter_{imf_num_iter}_mode_{fw_or_bw}",
                             tu.make_grid(trajectory[i], nrow=int(batch_size ** 0.5), normalize=True))


def markovian_projection(data_loader, max_iter, args, condsampler, netG_proj, netD_proj,
                         opt_G_proj, opt_D_proj, ema_g, rank, pos_coeff, device, exp_path, imf_num_iter, T,
                         writer, test_dataloader_input, m1_test_gt, s1_test_gt, D_opt_steps=1, fw_or_bw='fw', compute_fid=True):
    num_of_samples_for_trajectories_to_visualize = 8
    num_trajectories = 8
    test_batch_size = args.test_batch_size
    condsampler_gt = CondLoaderSampler(data_loader)

    print(f"Mode {fw_or_bw}, start markovian projection on iteration = {imf_num_iter}")
    print(f"Will save EMA every {args.save_content_every} iteration")

    for iteration in tqdm.tqdm(range(max_iter)):
        if rank == 0:
            if args.save_content:
                if iteration % args.save_content_every == 0:
                    if args.use_ema:
                        with ema_g.average_parameters():
                            print('Saving content in EMA.')
                            content = {'iteration': iteration,  'imf_num_iter': imf_num_iter, 'fw_or_bw': fw_or_bw,
                                       'args': args, 'netG_dict': netG_proj.state_dict(),
                                       'optimizerG': opt_G_proj.state_dict(),
                                       'netD_dict': netD_proj.state_dict(),
                                       'optimizerD': opt_D_proj.state_dict()}

                            path_to_save_ckpt = os.path.join(exp_path,
                                                    f'content_{fw_or_bw}_imf_num_iter_{imf_num_iter}_{iteration}.pth')

                            torch.save(content, path_to_save_ckpt)
                            print(f"Saving {path_to_save_ckpt}")

                            if compute_fid:
                                save_dir_pred_fid_statistics = \
                                    "./fid_results_train_imf/{}/{}/{}/{}/iteration_{}".format(args.dataset,
                                                                                            args.exp,
                                                                                            fw_or_bw,
                                                                                            imf_num_iter,
                                                                                            iteration)
                                if not os.path.exists(save_dir_pred_fid_statistics):
                                    os.makedirs(save_dir_pred_fid_statistics)
                                path_to_save_pred_fid_statistics = os.path.join(save_dir_pred_fid_statistics,
                                                                                "pred_statistics.npz")
                                netG_proj.eval()

                                with torch.no_grad():
                                    exp_path_mode = os.path.join(exp_path, fw_or_bw)
                                    os.makedirs(exp_path_mode, exist_ok=True)
                                    exp_path_mode_num_iter = os.path.join(exp_path_mode, str(imf_num_iter))
                                    os.makedirs(exp_path_mode_num_iter, exist_ok=True)
                                    if args.use_ema:
                                        with ema_g.average_parameters():
                                            save_and_log_images(condsampler_gt, device, args, args.use_ema, pos_coeff,
                                                                netG_proj, 
                                                                test_batch_size, iteration, writer,
                                                                num_of_samples_for_trajectories_to_visualize,
                                                                exp_path_mode_num_iter, num_trajectories, imf_num_iter,
                                                                fw_or_bw)
                                    else:
                                        save_and_log_images(condsampler_gt, device, args, args.use_ema, pos_coeff,
                                                            netG_proj, 
                                                            test_batch_size, iteration, writer,
                                                            num_of_samples_for_trajectories_to_visualize,
                                                            exp_path_mode_num_iter, num_trajectories, imf_num_iter,
                                                            fw_or_bw)

                                    dims = 2048
                                    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
                                    model_fid = InceptionV3([block_idx]).to(device)

                                    mode = "features_pred"
                                    m1_test_pred, s1_test_pred, test_l2_cost = compute_statistics_of_path_or_dataloader(
                                        path_to_save_pred_fid_statistics,
                                        test_dataloader_input, model_fid,
                                        test_batch_size, dims, device,
                                        mode, netG_proj, args,
                                        pos_coeff)

                                    del model_fid
                                    fid_value_test = calculate_frechet_distance(m1_test_pred, s1_test_pred,
                                                                                m1_test_gt, s1_test_gt)

                                    path_to_save_fid_txt = os.path.join(save_dir_pred_fid_statistics, "fid.txt")
                                    with open(path_to_save_fid_txt, "w") as f:
                                        f.write(f"FID = {fid_value_test}")
                                    print(
                                        f'Test FID = {fid_value_test} on dataset {args.dataset} for '
                                        f'{args.exp} and iteration = {iteration}, num_imf_iter = {imf_num_iter}, '
                                        f'mode = {fw_or_bw}')

                                    path_to_save_cost_txt = os.path.join(save_dir_pred_fid_statistics, "l2_cost.txt")
                                    if not os.path.exists(path_to_save_cost_txt):
                                        with open(path_to_save_cost_txt, "w") as f:
                                            f.write(f"L2 cost = {test_l2_cost}")
                                        print(f"write to file {path_to_save_cost_txt} l2 cost = {test_l2_cost}")
                                    else:
                                        print(f"Read from file {path_to_save_cost_txt}")
                                        with open(path_to_save_cost_txt, "r") as f:
                                            line = f.readlines()[0]
                                            test_l2_cost = float(line.split(" ")[-1])

                                    print(
                                        f'Test l2 cost = {test_l2_cost} on dataset {args.dataset} for exp '
                                        f'{args.exp} and iteration = {iteration}, num_imf_iter = {imf_num_iter}, '
                                        f'mode = {fw_or_bw}')

                                writer.add_scalar(iteration, f"test_fid_{imf_num_iter}_mode_{fw_or_bw}", fid_value_test)
                                writer.add_scalar(iteration, f"test_l2_cost_{imf_num_iter}_mode_{fw_or_bw}", test_l2_cost)

                                netG_proj.train()

        x, y = condsampler.sample(args.batch_size)
        x, y = x.to(device), y.to(device)

        # print(f"rank = {rank}, bs = {x.shape}")

        # -----Discriminator Opt Step-----

        # Get D ready for optimization
        for p in netD_proj.parameters():
            p.requires_grad = True

        netD_proj.zero_grad()

        # sample from p(x_0)
        real_data = x.to(device, non_blocking=True)
        input_real_data = y.to(device, non_blocking=True)

        if args.posterior == "ddpm":
            t = torch.randint(0, args.num_timesteps, (1,), device=device).repeat(real_data.size(0))
            x_t, x_tp1 = q_sample_supervised_pairs(pos_coeff, real_data, t, input_real_data)
        elif args.posterior == "brownian_bridge":
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_supervised_pairs_brownian(pos_coeff, real_data, t, input_real_data)

        x_t.requires_grad = True

        # train with real
        D_real = netD_proj(x_t, t, x_tp1.detach()).view(-1)

        errD_real = F.softplus(-D_real)
        errD_real = errD_real.mean()

        errD_real.backward(retain_graph=True)

        if args.lazy_reg is None:
            grad_real = torch.autograd.grad(
                outputs=D_real.sum(), inputs=x_t, create_graph=True
            )[0]
            grad_penalty = (
                    grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()

            grad_penalty = args.r1_gamma / 2 * grad_penalty
            grad_penalty.backward()
        else:
            if iteration % args.lazy_reg == 0:
                grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t, create_graph=True
                )[0]
                grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()

                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()

        # train with fake
        latent_z = torch.randn(args.batch_size, args.nz, device=device)

        x_0_predict = netG_proj(x_tp1.detach(), t, latent_z)
        x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

        output = netD_proj(x_pos_sample, t, x_tp1.detach()).view(-1)

        errD_fake = F.softplus(output)
        errD_fake = errD_fake.mean()
        errD_fake.backward()

        errD = errD_real + errD_fake
        # Update D
        opt_D_proj.step()

        # -----Generator Opt Step-----

        if iteration % D_opt_steps == 0:

            # Get G ready for optimization
            for p in netD_proj.parameters():
                p.requires_grad = False
            netG_proj.zero_grad()

            if args.posterior == "brownian_bridge":
                t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
                x_t, x_tp1 = q_sample_supervised_pairs_brownian(pos_coeff, real_data, t, input_real_data)
            else:
                raise ValueError('ONLY Brownian Bridge posterior')

            latent_z = torch.randn(args.batch_size, args.nz, device=device)

            x_0_predict = netG_proj(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD_proj(x_pos_sample, t, x_tp1.detach()).view(-1)

            errG = F.softplus(-output)
            errG = errG.mean()

            errG.backward()
            opt_G_proj.step()
            ema_g.update()

            # iter_gloval = iteration + epoch * data_loader_size
            if rank == 0:
                writer.add_scalar(iteration, f"G_loss_imf_iter_{imf_num_iter}_mode_{fw_or_bw}", errG.item())
                writer.add_scalar(iteration, f"D_loss_imf_iter_{imf_num_iter}_mode_{fw_or_bw}", errD.item())

        if iteration % 100 == 0:
            if rank == 0:
                print('Markovain proj {} on IMF iter {}: Iter {}, G Loss: {}, D Loss: {}'.format(fw_or_bw,
                                                                                                 imf_num_iter,
                                                                                                 iteration,
                                                                                                 errG.item(),
                                                                                                 errD.item()))


# %%
def train(rank, gpu, args):
    from score_sde.models.discriminator import Discriminator_small, Discriminator_large
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp
    # from EMA import EMA

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size

    inner_imf_mark_proj_iters = args.inner_imf_mark_proj_iters
    D_opt_steps = args.D_opt_steps

    if args.dataset == 'cifar10':
        train_dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)

    elif args.dataset == 'stackmnist':
        train_transform, valid_transform = _data_transforms_stacked_mnist()
        train_dataset = StackedMNIST(root='./data', train=True, download=False, transform=train_transform)

    elif args.dataset == 'lsun':
        train_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_data = LSUN(root='/datasets/LSUN/', classes=['church_outdoor_train'], transform=train_transform)
        subset = list(range(0, 120000))
        train_dataset = torch.utils.data.Subset(train_data, subset)

    elif args.dataset == 'celeba_256':
        train_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = LMDBDataset(root='/datasets/celeba-lmdb/', name='celeba', train=True, transform=train_transform)

    elif args.dataset == 'paired_colored_mnist':
        print(f"loading paired colored mnist")
        train_dataset, test_dataset = load_paired_colored_mnist()

    elif args.dataset == 'faces2comics':
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

    elif args.dataset == 'celeba':
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
                                                               batch_size=args.test_batch_size,
                                                               shuffle=False,
                                                               drop_last=False,
                                                               num_workers=cpu_count())
        test_dataloader_target_fw = torch.utils.data.DataLoader(test_dataset_target_fw,
                                                                batch_size=args.test_batch_size,
                                                                shuffle=False,
                                                                drop_last=False,
                                                                num_workers=cpu_count())
        test_dataloader_input_bw = torch.utils.data.DataLoader(test_dataset_input_bw,
                                                               batch_size=args.test_batch_size,
                                                               shuffle=False,
                                                               drop_last=False,
                                                               num_workers=cpu_count())
        test_dataloader_target_bw = torch.utils.data.DataLoader(test_dataset_target_bw,
                                                                batch_size=args.test_batch_size,
                                                                shuffle=False,
                                                                drop_last=False,
                                                                num_workers=cpu_count())

    train_sampler_input = torch.utils.data.distributed.DistributedSampler(train_dataset_input,
                                                                          num_replicas=args.world_size,
                                                                          rank=rank)
    train_data_loader_input = torch.utils.data.DataLoader(train_dataset_input,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          num_workers=4,
                                                          pin_memory=True,
                                                          sampler=train_sampler_input,
                                                          drop_last=True)

    train_sampler_target = torch.utils.data.distributed.DistributedSampler(train_dataset_target,
                                                                           num_replicas=args.world_size,
                                                                           rank=rank)
    train_data_loader_target = torch.utils.data.DataLoader(train_dataset_target,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           num_workers=4,
                                                           pin_memory=True,
                                                           sampler=train_sampler_target,
                                                           drop_last=True)

    # data_loader_size = len(data_loader)
    # print(f'data_loader size = {data_loader_size}')
    netG_fw = NCSNpp(args).to(device)
    netG_bw = NCSNpp(args).to(device)

    if args.dataset == 'cifar10' or args.dataset == 'stackmnist' or args.dataset == "paired_colored_mnist":
        print("use small discriminator")
        netD_fw = Discriminator_small(nc=2 * args.num_channels, ngf=args.ngf,
                                      t_emb_dim=args.t_emb_dim,
                                      act=nn.LeakyReLU(0.2)).to(device)
        netD_bw = Discriminator_small(nc=2 * args.num_channels, ngf=args.ngf,
                                      t_emb_dim=args.t_emb_dim,
                                      act=nn.LeakyReLU(0.2)).to(device)
    else:
        print("use large discriminator")
        netD_fw = Discriminator_large(nc=2 * args.num_channels, ngf=args.ngf,
                                      t_emb_dim=args.t_emb_dim,
                                      act=nn.LeakyReLU(0.2)).to(device)
        netD_bw = Discriminator_large(nc=2 * args.num_channels, ngf=args.ngf,
                                      t_emb_dim=args.t_emb_dim,
                                      act=nn.LeakyReLU(0.2)).to(device)

    optimizerG_fw = optim.Adam(netG_fw.parameters(),
                               lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizerD_fw = optim.Adam(netD_fw.parameters(),
                               lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG_bw = optim.Adam(netG_bw.parameters(),
                               lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizerD_bw = optim.Adam(netD_bw.parameters(),
                               lr=args.lr_d, betas=(args.beta1, args.beta2))

    # ddp
    netG_fw = nn.parallel.DistributedDataParallel(netG_fw, device_ids=[gpu])
    netD_fw = nn.parallel.DistributedDataParallel(netD_fw, device_ids=[gpu])
    netG_bw = nn.parallel.DistributedDataParallel(netG_bw, device_ids=[gpu])
    netD_bw = nn.parallel.DistributedDataParallel(netD_bw, device_ids=[gpu])

    exp_forward = args.exp_forward
    exp_backward = args.exp_backward
    parent_dir_forward = "./saved_info/dd_gan/{}".format(args.dataset_forward)
    parent_dir_backward = "./saved_info/dd_gan/{}".format(args.dataset_backward)

    exp = args.exp
    exp_path = "./saved_info/dd_gan_imf/{}_{}/{}".format(args.dataset_forward, args.dataset_backward, exp)
    print(f"will save results in {exp_path}")

    exp_path_fw = os.path.join(parent_dir_forward, exp_forward)
    exp_path_bw = os.path.join(parent_dir_backward, exp_backward)

    coeff = Diffusion_Coefficients(args, device)
    if args.posterior == "ddpm":
        pos_coeff = Posterior_Coefficients(args, device)
    elif args.posterior == "brownian_bridge":
        pos_coeff = BrownianPosterior_Coefficients(args, device)

    T = get_time_schedule(args, device)

    writer = TensorBoardWriter(args, path_to_save="./saved_info/dd_gan_imf",
                               dir_name=f"{args.dataset_forward}_{args.dataset_backward}")
    
    resume = args.resume

    compute_fid = False
    print(f"Compute FID = {compute_fid}")

    if compute_fid:
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model_fid = InceptionV3([block_idx]).to(device)

        save_dir_gt_fid_statistics_forward = "./fid_results_train/{}/{}".format(args.dataset_forward, args.exp_forward)
        save_dir_gt_fid_statistics_backward = "./fid_results_train/{}/{}".format(args.dataset_backward, args.exp_backward)
        os.makedirs(save_dir_gt_fid_statistics_forward, exist_ok=True)
        os.makedirs(save_dir_gt_fid_statistics_backward, exist_ok=True)
        path_to_save_gt_fid_statistics_forward = os.path.join(save_dir_gt_fid_statistics_forward, "gt_statistics_forward.npz")
        path_to_save_gt_fid_statistics_backward = os.path.join(save_dir_gt_fid_statistics_backward, "gt_statistics_backward.npz")
        path_to_save_pred_fid_statistics_forward = os.path.join(exp_path, "init_pred_forward_statistics.npz")
        path_to_save_pred_fid_statistics_backward = os.path.join(exp_path, "init_pred_backward_statistics.npz")

        mode = "features_gt"
        # "features_gt" -> (x, y) in dataloader, compute features of x
        # "features_pred" -> (x, y) in dataloader, comput features of f(y)
        m1_test_gt_for, s1_test_gt_for, _ = compute_statistics_of_path_or_dataloader(path_to_save_gt_fid_statistics_forward,
                                                                        test_dataloader_target_fw, model_fid,
                                                                        args.test_batch_size, dims, device,
                                                                        mode, None, None, None)
        m1_test_gt_back, s1_test_gt_back, _ = compute_statistics_of_path_or_dataloader(path_to_save_gt_fid_statistics_backward,
                                                                        test_dataloader_target_bw, model_fid,
                                                                        args.test_batch_size, dims, device,
                                                                        mode, None, None, None)
        
        if not os.path.exists(path_to_save_gt_fid_statistics_forward):
            print(f"Forward, saving stats for gt test data to {path_to_save_gt_fid_statistics_forward}")
            np.savez(path_to_save_gt_fid_statistics_forward, mu=m1_test_gt_for, sigma=s1_test_gt_for)

        if not os.path.exists(path_to_save_gt_fid_statistics_backward):
            print(f"Backward, saving stats for gt test data to {path_to_save_gt_fid_statistics_backward}")
            np.savez(path_to_save_gt_fid_statistics_backward, mu=m1_test_gt_back, sigma=s1_test_gt_back)

    else:
        m1_test_gt_for = None 
        s1_test_gt_for = None
        m1_test_gt_back = None 
        s1_test_gt_back = None

    imf_iters = args.imf_iters

    if not resume:
        print(f"training IMF from beginning!")
        checkpoint_file_fw = os.path.join(exp_path_fw, args.exp_forward_model)
        checkpoint_fw = torch.load(checkpoint_file_fw, map_location=device)
        global_step_fw = checkpoint_fw['global_step']
        print(f"downloaded forward ckpt {checkpoint_file_fw} on iter {global_step_fw}")
        netG_fw.load_state_dict(checkpoint_fw['netG_dict'])
        # load G

        optimizerG_fw.load_state_dict(checkpoint_fw['optimizerG'])

        ema_g_fw = ExponentialMovingAverage(netG_fw.parameters(), decay=args.ema_decay)
        ema_g_fw.to(device)

        # load D
        netD_fw.load_state_dict(checkpoint_fw['netD_dict'])
        optimizerD_fw.load_state_dict(checkpoint_fw['optimizerD'])

        checkpoint_file_bw = os.path.join(exp_path_bw, args.exp_backward_model)
        checkpoint_bw = torch.load(checkpoint_file_bw, map_location=device)
        global_step_bw = checkpoint_bw['global_step']
        print(f"downloaded backward ckpt {checkpoint_file_bw} on iter {global_step_bw}")
        netG_bw.load_state_dict(checkpoint_bw['netG_dict'])
        # load G

        optimizerG_bw.load_state_dict(checkpoint_bw['optimizerG'])

        ema_g_bw = ExponentialMovingAverage(netG_bw.parameters(), decay=args.ema_decay)
        ema_g_bw.to(device)

        # load D
        netD_bw.load_state_dict(checkpoint_bw['netD_dict'])
        optimizerD_bw.load_state_dict(checkpoint_bw['optimizerD'])

        print(f"save_content_every = {args.save_content_every}, save_ckpt_every = {args.save_ckpt_every}")

        if compute_fid:
            if rank == 0:
                mode = "features_pred"
                m1_test_pred_fw, s1_test_pred_fw, test_l2_cost_forward = compute_statistics_of_path_or_dataloader(path_to_save_pred_fid_statistics_forward,
                                                                                    test_dataloader_input_fw, model_fid,
                                                                                    args.test_batch_size, dims, device,
                                                                                    mode, netG_fw, args,
                                                                                    pos_coeff)

                m1_test_pred_bw, s1_test_pred_bw, test_l2_cost_backward = compute_statistics_of_path_or_dataloader(path_to_save_pred_fid_statistics_backward,
                                                                                    test_dataloader_input_bw, model_fid,
                                                                                    args.test_batch_size, dims, device,
                                                                                    mode, netG_bw, args,
                                                                                    pos_coeff)

                fid_value_test_fw = calculate_frechet_distance(m1_test_pred_fw, s1_test_pred_fw, m1_test_gt_for, s1_test_gt_for)
                print(f"Forward FID for init model = {fid_value_test_fw}")
                fid_value_test_bw = calculate_frechet_distance(m1_test_pred_bw, s1_test_pred_bw, m1_test_gt_back, s1_test_gt_back)
                print(f"Backward FID for init model = {fid_value_test_bw}")

                path_to_save_cost_txt_fw = os.path.join(exp_path, "init_pred_forward_cost.txt")
                if not os.path.exists(path_to_save_cost_txt_fw):
                    with open(path_to_save_cost_txt_fw, "w") as f:
                        f.write(f"L2 cost = {test_l2_cost_forward}")

                    print(f"saving L2 cost to {test_l2_cost_forward}")
                else:
                    with open(path_to_save_cost_txt_fw, "r") as f:
                        line = f.readlines()[0]
                        test_l2_cost_forward = float(line.split(" ")[-1])
                print(f"Forward cost for init model = {test_l2_cost_forward}")

                path_to_save_cost_txt_bw = os.path.join(exp_path, "init_pred_backward_cost.txt")
                if not os.path.exists(path_to_save_cost_txt_bw):
                    with open(path_to_save_cost_txt_bw, "w") as f:
                        f.write(f"L2 cost = {test_l2_cost_backward}")

                    print(f"saving L2 cost to {path_to_save_cost_txt_bw}")
                else:
                    with open(path_to_save_cost_txt_bw, "r") as f:
                        line = f.readlines()[0]
                        test_l2_cost_backward = float(line.split(" ")[-1])
                print(f"Backward cost for init model = {test_l2_cost_backward}")

                if not os.path.exists(path_to_save_pred_fid_statistics_forward):
                    print(f"Forward, saving stats for pred test data to {path_to_save_gt_fid_statistics_forward}")
                    np.savez(path_to_save_pred_fid_statistics_forward, mu=m1_test_pred_fw, sigma=s1_test_pred_fw)

                if not os.path.exists(path_to_save_pred_fid_statistics_backward):
                    print(f"Backward, saving stats for pred test data to {path_to_save_gt_fid_statistics_backward}")
                    np.savez(path_to_save_pred_fid_statistics_backward, mu=m1_test_pred_bw, sigma=s1_test_pred_bw)

        print(f"Start to run IMF with {imf_iters} iterations")

        for imf_num_iter in range(imf_iters):
            # ----Forward model learning (y -> x)----

            bmgan_sample_bw = lambda x: sample_from_model(pos_coeff, netG_bw,
                                                        args.num_timesteps, x.to(device),
                                                        args, return_trajectory=True)[0]

            sampler_x = XSampler(CondLoaderSampler(train_data_loader_input))

            if args.use_ema:
                condsampler_fw = ModelCondSampler(sampler_x, bmgan_sample_bw, ema_model=ema_g_bw)
            else:
                condsampler_fw = ModelCondSampler(sampler_x, bmgan_sample_bw)

            markovian_projection(train_data_loader_input, inner_imf_mark_proj_iters, args, condsampler_fw, netG_fw, netD_fw,
                                optimizerG_fw, optimizerD_fw, ema_g_fw, rank, pos_coeff, device, exp_path, imf_num_iter,
                                T, writer, test_dataloader_input_fw, m1_test_gt_for, s1_test_gt_for,
                                D_opt_steps=D_opt_steps, fw_or_bw='fw', compute_fid=compute_fid)

            bmgan_sample_fw = lambda x: sample_from_model(pos_coeff, netG_fw,
                                                        args.num_timesteps, x.to(device),
                                                        args, return_trajectory=True)[0]

            sampler_x = XSampler(CondLoaderSampler(train_data_loader_target))

            if args.use_ema:
                condsampler_bw = ModelCondSampler(sampler_x, bmgan_sample_fw, ema_model=ema_g_fw)
            else:
                condsampler_bw = ModelCondSampler(sampler_x, bmgan_sample_fw)

            markovian_projection(train_data_loader_target, inner_imf_mark_proj_iters, args, condsampler_bw, netG_bw, netD_bw,
                                optimizerG_bw, optimizerD_bw, ema_g_bw, rank, pos_coeff, device, exp_path, imf_num_iter,
                                T, writer, test_dataloader_input_bw, m1_test_gt_back, s1_test_gt_back,
                                D_opt_steps=D_opt_steps, fw_or_bw='bw', compute_fid=compute_fid)
            
    else:
        print(f"Resuming from existing checkpoints!")
        all_ckpts_fw = sorted(glob.glob(os.path.join(exp_path, "content_fw_imf_num_iter_*_0.pth")))
        all_ckpts_bw = sorted(glob.glob(os.path.join(exp_path, "content_bw_imf_num_iter_*_0.pth")))
        all_ckpts_fw_basenames = [os.path.basename(path) for path in all_ckpts_fw]
        all_ckpts_bw_basenames = [os.path.basename(path) for path in all_ckpts_bw]
        all_ckpts_fw_iters = sorted([int(name.split(".")[0].split("_")[-2]) for name in all_ckpts_fw_basenames])
        all_ckpts_bw_iters = sorted([int(name.split(".")[0].split("_")[-2]) for name in all_ckpts_bw_basenames])
        print(f"maximum num IMF iter in forward = {all_ckpts_fw_iters[-1]}")
        print(f"maximum num IMF iter in backward = {all_ckpts_bw_iters[-1]}")
        iter_to_start = min(all_ckpts_fw_iters[-1], all_ckpts_bw_iters[-1])
        print(f"Will resume IMF from iter {iter_to_start}")
        print(f"training IMF from beginning!")
        checkpoint_file_fw = os.path.join(exp_path, f"content_fw_imf_num_iter_{iter_to_start}_0.pth")
        checkpoint_fw = torch.load(checkpoint_file_fw, map_location=device)
        print(f"downloaded forward ckpt {checkpoint_file_fw}")
        netG_fw.load_state_dict(checkpoint_fw['netG_dict'])
        # load G

        optimizerG_fw.load_state_dict(checkpoint_fw['optimizerG'])

        ema_g_fw = ExponentialMovingAverage(netG_fw.parameters(), decay=args.ema_decay)
        ema_g_fw.to(device)

        # load D
        netD_fw.load_state_dict(checkpoint_fw['netD_dict'])
        optimizerD_fw.load_state_dict(checkpoint_fw['optimizerD'])

        checkpoint_file_bw = os.path.join(exp_path, f"content_bw_imf_num_iter_{iter_to_start}_0.pth")
        checkpoint_bw = torch.load(checkpoint_file_bw, map_location=device)
        print(f"downloaded backward ckpt {checkpoint_file_bw}")
        netG_bw.load_state_dict(checkpoint_bw['netG_dict'])
        # load G

        optimizerG_bw.load_state_dict(checkpoint_bw['optimizerG'])

        ema_g_bw = ExponentialMovingAverage(netG_bw.parameters(), decay=args.ema_decay)
        ema_g_bw.to(device)

        # load D
        netD_bw.load_state_dict(checkpoint_bw['netD_dict'])
        optimizerD_bw.load_state_dict(checkpoint_bw['optimizerD'])

        print(f"Start to run IMF with {imf_iters} iterations from {iter_to_start} IMF iter")

        for imf_num_iter in range(iter_to_start, imf_iters):
            # ----Forward model learning (y -> x)----

            bmgan_sample_bw = lambda x: sample_from_model(pos_coeff, netG_bw,
                                                        args.num_timesteps, x.to(device),
                                                        args, return_trajectory=True)[0]

            sampler_x = XSampler(CondLoaderSampler(train_data_loader_input))

            if args.use_ema:
                condsampler_fw = ModelCondSampler(sampler_x, bmgan_sample_bw, ema_model=ema_g_bw)
            else:
                condsampler_fw = ModelCondSampler(sampler_x, bmgan_sample_bw)

            markovian_projection(train_data_loader_input, inner_imf_mark_proj_iters, args, condsampler_fw, netG_fw, netD_fw,
                                optimizerG_fw, optimizerD_fw, ema_g_fw, rank, pos_coeff, device, exp_path, imf_num_iter,
                                T, writer, test_dataloader_input_fw, m1_test_gt_for, s1_test_gt_for,
                                D_opt_steps=D_opt_steps, fw_or_bw='fw', compute_fid=compute_fid)

            bmgan_sample_fw = lambda x: sample_from_model(pos_coeff, netG_fw,
                                                        args.num_timesteps, x.to(device),
                                                        args, return_trajectory=True)[0]

            sampler_x = XSampler(CondLoaderSampler(train_data_loader_target))

            if args.use_ema:
                condsampler_bw = ModelCondSampler(sampler_x, bmgan_sample_fw, ema_model=ema_g_fw)
            else:
                condsampler_bw = ModelCondSampler(sampler_x, bmgan_sample_fw)

            markovian_projection(train_data_loader_target, inner_imf_mark_proj_iters, args, condsampler_bw, netG_bw, netD_bw,
                                optimizerG_bw, optimizerD_bw, ema_g_bw, rank, pos_coeff, device, exp_path, imf_num_iter,
                                T, writer, test_dataloader_input_bw, m1_test_gt_back, s1_test_gt_back,
                                D_opt_steps=D_opt_steps, fw_or_bw='bw', compute_fid=compute_fid)



def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')

    parser.add_argument('--posterior', type=str, default='ddpm',
                        help='type of posterior to use')

    # ddpm prior
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    # brownian bridge prior
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
    parser.add_argument('--exp_forward', type=str)
    parser.add_argument('--exp_backward', type=str)
    parser.add_argument('--exp_forward_model', type=str, default="content.pth")
    parser.add_argument('--exp_backward_model', type=str, default="content.pth")
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--dataset_forward', type=str)
    parser.add_argument('--dataset_backward', type=str)
    parser.add_argument('--data_root', default='')
    parser.add_argument('--paired', action='store_true', default=False)
    parser.add_argument('--use_minibatch_ot', action='store_true', default=False)

    parser.add_argument('--inner_imf_mark_proj_iters', type=int, default=20000)
    parser.add_argument('--imf_iters', type=int, default=20)
    parser.add_argument('--D_opt_steps', type=int, default=1)

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=50, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6022',
                        help='address for master')

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        print('starting in multiprocessing mode')
        torch.multiprocessing.set_start_method("spawn")
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('starting in debug mode')
        args.global_rank = 0
        init_processes(0, size, train, args)

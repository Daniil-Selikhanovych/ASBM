# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
import glob

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as tu

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from datasets_prep.lsun import LSUN
from datasets_prep.stackmnist_data import StackedMNIST, _data_transforms_stacked_mnist
from datasets_prep.lmdb_datasets import LMDBDataset
from datasets_prep.faces2comics import Faces2Comics
from datasets_prep.paired_colored_mnist import load_paired_colored_mnist
from datasets_prep.celeba import Celeba

from discrete_ot import OTPlanSampler
from multiprocessing import cpu_count

from torch.multiprocessing import Process
import torch.distributed as dist
import shutil

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

from sampling_utils import sample_from_model, q_sample_supervised_trajectory, q_sample_supervised_pairs, \
    q_sample_supervised_pairs_brownian, sample_posterior
from posteriors import Diffusion_Coefficients, Posterior_Coefficients, BrownianPosterior_Coefficients, \
    get_time_schedule
from metrics import compute_statistics_of_path_or_dataloader

from utils import TensorBoardWriter

from torch_ema import ExponentialMovingAverage

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))
            
def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def save_and_log_images(train_data_loader, device, use_minibatch_ot, ot_plan_sampler, args, use_ema, pos_coeff, netG,
                        batch_size, writer, num_of_samples_for_trajectories_to_visualize, exp_path,
                        global_step, num_trajectories, is_train):
    for (x, y) in train_data_loader:
        x_0 = x.to(device)
        x_t_1 = y.to(device)
        break

    print(f"save_and_log_images, save ema = {use_ema}, is train = {is_train}, use minibath ot = {use_minibatch_ot}")

    if use_minibatch_ot:
        with torch.no_grad():
            x_0, x_t_1 = ot_plan_sampler.sample_plan(x_0, x_t_1)

    # x_t_1.shape = (B, C, H, W)
    # print(f"x_0.shape ={x_0.shape}, x_t_1.shape = {x_t_1.shape}")

    # x_t_1 = torch.randn_like(real_data)
    if use_ema:
        print(f"Sampling for differents starts from EMA with decay = {args.ema_decay}")
    else:
        print(f"Sampling for differents starts from last G")
    # fake_sample in [-1, 1]
    # trajectory = [x_t_1, ...]
    # torchvision.utils.save_image(fake_sample, os.path.join(exp_path, 'sample_discrete_epoch_{}.png'.format(epoch)), normalize=True)
    fake_sample, trajectory = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, args,
                                                return_trajectory=True)
    if writer is not None:
        writer.add_image(global_step, f"sample_discrete_ema_{use_ema}_is_train_{is_train}", 
                         tu.make_grid(fake_sample, nrow=int(batch_size ** 0.5), normalize=True))
        print(f"saving sample_discrete_ema_{use_ema}_is_train_{is_train} at global step {global_step} to writer {writer}")


    trajectory_to_visualize = []
    for i in range(args.num_timesteps + 1):
        if writer is not None:
            writer.add_image(global_step, f"gen_x_{args.num_timesteps - i}_ema_{use_ema}_is_train_{is_train}",
                             tu.make_grid(trajectory[i], nrow=int(batch_size ** 0.5), normalize=True))
        # torchvision.utils.save_image(trajectory[i], os.path.join(exp_path, f"gen_x_{args.num_timesteps-i}_epoch_{epoch}.png"), normalize=True)
            print(f"saving gen_x_{args.num_timesteps - i}_ema_{use_ema}_is_train_{is_train} at global step {global_step} to writer {writer}")

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
                                        f'evolution_step_{global_step}_ema_{use_ema}_is_train_{is_train}.png')

    plt.savefig(path_to_save_figures)
    print(f"saving {path_to_save_figures}")
    img_to_tensorboard = Image.open(path_to_save_figures)
    img_to_tensorboard = transforms.ToTensor()(img_to_tensorboard)[:3]
    if writer is not None:
        writer.add_image(global_step, f"evolution_different_starts_trajectories_ema_{use_ema}_is_train_{is_train}", 
                         img_to_tensorboard)
        print(f"saving evolution_different_starts_trajectories_ema_{use_ema}_is_train_{is_train} at global step {global_step} to writer {writer}")

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
                ax[0][j].set_title(f'Real data')
            ax[k][j].xaxis.tick_top()
            # ax[k][j].get_xaxis().set_visible(False)
            ax[k][j].set_xticks([])
            ax[k][j].get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.001)
    path_to_save_figures = os.path.join(exp_path,
                                    f'evolution_many_trajectories_step_{global_step}_ema_{use_ema}_is_train_{is_train}.png')
    
    plt.savefig(path_to_save_figures)
    print(f"saving {path_to_save_figures}")
    img_to_tensorboard = Image.open(path_to_save_figures)
    img_to_tensorboard = transforms.ToTensor()(img_to_tensorboard)[:3]
    if writer is not None:
        writer.add_image(global_step, f"evolution_same_start_trajectories_ema_{use_ema}_is_train_{is_train}", 
                         img_to_tensorboard)
        print(f"sending image evolution_same_start_trajectories_ema_{use_ema}_is_train_{is_train} at global step {global_step} to writer {writer}")

    trajectory = q_sample_supervised_trajectory(pos_coeff, x_0, x_t_1)
    # x_start = x_0, x_end = x_t_1
    # trajectory = [x_t_1, ...]
    if writer is not None:
        for i in range(args.num_timesteps + 1):
            writer.add_image(global_step, f"x_{args.num_timesteps - i}_ema_{use_ema}_is_train_{is_train}",
                             tu.make_grid(trajectory[i], nrow=int(batch_size ** 0.5), normalize=True))
            print(f"sending image x_{args.num_timesteps - i}_ema_{use_ema}_is_train_{is_train} at global step {global_step} to writer {writer}")


def validate_metrics(test_dataloader_input, netG, args, exp_path, epoch, device, pos_coeff, T, writer,
                     num_of_samples_for_trajectories_to_visualize, global_step, num_trajectories,
                     m1_test_gt, s1_test_gt, use_ema):
    net_G_name = f'netG_{epoch}_ema_{use_ema}.pth'
    print(f"saving with EMA {net_G_name}, decay = {args.ema_decay}")
    torch.save(netG.state_dict(), os.path.join(exp_path, net_G_name))

    save_dir_pred_fid_statistics = "./fid_results_train/{}/{}/epoch_{}".format(args.dataset,
                                                                                args.exp,
                                                                                epoch)
    if not os.path.exists(save_dir_pred_fid_statistics):
        os.makedirs(save_dir_pred_fid_statistics)
    path_to_save_pred_fid_statistics = os.path.join(save_dir_pred_fid_statistics,
                                                    f"pred_statistics_ema_{use_ema}.npz")
    netG.eval()

    with torch.no_grad():
        save_and_log_images(test_dataloader_input, device, False, None, args, use_ema, pos_coeff,
                            netG, 64, writer,
                            num_of_samples_for_trajectories_to_visualize, save_dir_pred_fid_statistics,
                            global_step,
                            num_trajectories, False)

        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model_fid = InceptionV3([block_idx]).to(device)

        mode = "features_pred"
        m1_test_pred, s1_test_pred, test_l2_cost = compute_statistics_of_path_or_dataloader(
            path_to_save_pred_fid_statistics,
            test_dataloader_input, model_fid,
            args.batch_size, dims, device,
            mode, netG, args,
            pos_coeff)
        fid_value_test = calculate_frechet_distance(m1_test_pred, s1_test_pred,
                                                    m1_test_gt, s1_test_gt)

        path_to_save_fid_txt = os.path.join(save_dir_pred_fid_statistics, 
                                            f"fid_test_ema_{use_ema}.txt")
        with open(path_to_save_fid_txt, "w") as f:
            f.write(f"FID = {fid_value_test}")
        print(
            f'Test FID = {fid_value_test} on dataset {args.dataset} for exp {args.exp} and epoch = {epoch}')
        
        path_to_save_cost_txt = os.path.join(save_dir_pred_fid_statistics, 
                                             f"l2_cost_test_ema_{use_ema}.txt")
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
            f'Test l2 cost = {test_l2_cost} on dataset {args.dataset} for exp {args.exp} and epoch = {epoch}')

        del model_fid

        path_to_save_cost_txt = os.path.join(save_dir_pred_fid_statistics, 
                                             f"l2_cost_test_ema_{use_ema}.txt")
        
    writer.add_scalar(global_step, f"test_fid_ema_{use_ema}", fid_value_test)
    print(f"sending test_fid_ema_{use_ema} to tensorboard at global step {global_step} to {writer}!")
    writer.add_scalar(global_step, f"test_l2_cost_ema_{use_ema}", test_l2_cost)
    print(f"sending test_l2_cost_ema_{use_ema} to tensorboard at global step {global_step} to {writer}!")

    netG.train()

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
    
    nz = args.nz  # latent dimension
    
    if args.dataset == 'cifar10':
        train_dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)

    elif args.dataset == 'stackmnist':
        train_transform, valid_transform = _data_transforms_stacked_mnist()
        train_dataset = StackedMNIST(root='./data', train=True, download=False, transform=train_transform)
        
    elif args.dataset == 'lsun':
        train_transform = transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.CenterCrop(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ])

        train_data = LSUN(root='/datasets/LSUN/', classes=['church_outdoor_train'], transform=train_transform)
        subset = list(range(0, 120000))
        train_dataset = torch.utils.data.Subset(train_data, subset)

    elif args.dataset == 'celeba_256':
        train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
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
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
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
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        data_root = args.data_root
        train_dataset = Celeba(data_root, train=True, transform=transform, part_test=0.1, mode="male_to_female")
        print(f"Num train images = {len(train_dataset)}")
        test_dataset_input = Celeba(data_root, train=False, transform=transform, part_test=0.1, mode="male_to_female")
        # (x, y) in dataloader, (female_img, male_img), indexing for males
        test_dataset_target = Celeba(data_root, train=False, transform=transform, part_test=0.1, mode="female_to_male",
                                     input_is_second=False)
        # (x, y) in dataloader, (female_img, male_img), indexing for female
        print(
            f"Num of input test images = {len(test_dataset_input)}, num of target test images = {len(test_dataset_target)}")
        test_dataloader_input = torch.utils.data.DataLoader(test_dataset_input,
                                                            batch_size=args.batch_size,
                                                            shuffle=False,
                                                            drop_last=False,
                                                            num_workers=cpu_count())
        test_dataloader_target = torch.utils.data.DataLoader(test_dataset_target,
                                                             batch_size=args.batch_size,
                                                             shuffle=False,
                                                             drop_last=False,
                                                             num_workers=cpu_count())

    elif args.dataset == 'celeba_female_to_male':
        print(f"Preparing celeba female2male")
        transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        data_root = args.data_root
        train_dataset = Celeba(data_root, train=True, transform=transform, part_test=0.1, mode="female_to_male")
        print(f"Num train images = {len(train_dataset)}")
        test_dataset_input = Celeba(data_root, train=False, transform=transform, part_test=0.1, mode="female_to_male")
        # (x, y) in dataloader, (female_img, male_img), indexing for males
        test_dataset_target = Celeba(data_root, train=False, transform=transform, part_test=0.1, mode="male_to_female",
                                     input_is_second=False)
        # (x, y) in dataloader, (female_img, male_img), indexing for female
        print(
            f"Num of input test images = {len(test_dataset_input)}, num of target test images = {len(test_dataset_target)}")
        test_dataloader_input = torch.utils.data.DataLoader(test_dataset_input,
                                                            batch_size=args.batch_size,
                                                            shuffle=False,
                                                            drop_last=False,
                                                            num_workers=cpu_count())
        test_dataloader_target = torch.utils.data.DataLoader(test_dataset_target,
                                                             batch_size=args.batch_size,
                                                             shuffle=False,
                                                             drop_last=False,
                                                             num_workers=cpu_count())

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=4,
                                                    pin_memory=True,
                                                    sampler=train_sampler,
                                                    drop_last=True)

    # data_loader_size = len(data_loader)
    # print(f'data_loader size = {data_loader_size}')

    print(f"NCSNpp with ch_mult = {args.ch_mult}")
    
    netG = NCSNpp(args).to(device)

    if args.dataset == 'cifar10' or args.dataset == 'stackmnist' or args.dataset == "paired_colored_mnist":    
        print("use small discriminator")
        netD = Discriminator_small(nc=2*args.num_channels, ngf=args.ngf,
                                   t_emb_dim=args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)
    else:
        print("use large discriminator")
        netD = Discriminator_large(nc=2*args.num_channels, ngf=args.ngf,
                                   t_emb_dim=args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)
    
    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())
    
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)

    # ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    if args.use_ema:
        # optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
        ema_g = ExponentialMovingAverage(netG.parameters(), decay=args.ema_decay)
        ema_g.to(device)
        print(f"use EMA from torch_ema with decay = {args.ema_decay}!")
        print(f"ema_g = {ema_g}")
    else:
        print(f"don't use EMA!")

    exp = args.exp
    parent_dir = "./saved_info/dd_gan/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models', os.path.join(exp_path, 'score_sde/models'))

    coeff = Diffusion_Coefficients(args, device)
    if args.posterior == "ddpm":
        pos_coeff = Posterior_Coefficients(args, device)
    elif args.posterior == "brownian_bridge":
        pos_coeff = BrownianPosterior_Coefficients(args, device)
        
    T = get_time_schedule(args, device)

    writer = TensorBoardWriter(args)

    if args.resume:
        all_contents = sorted(glob.glob(os.path.join(exp_path, 'content_*.pth')))
        all_contents_basenames = [os.path.basename(path) for path in all_contents]
        print(f"Found saved contents = {all_contents_basenames}")
        all_contents_iters = sorted([int(path.split("_")[1].split(".")[0]) for path in all_contents_basenames])
        used_content = f'content_{all_contents_iters[-1]}.pth'
        print(f"used content {used_content}")
        checkpoint_file = os.path.join(exp_path, used_content)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        netG.load_state_dict(checkpoint['netG_dict'])
        # load G
        
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    num_of_samples_for_trajectories_to_visualize = 8

    num_trajectories = 8

    print(f"save_content_every = {args.save_content_every}, save_ckpt_every = {args.save_ckpt_every}")

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model_fid = InceptionV3([block_idx]).to(device)

    save_dir_gt_fid_statistics = "./fid_results_train/{}/{}".format(args.dataset, args.exp)
    os.makedirs(save_dir_gt_fid_statistics, exist_ok=True)
    path_to_save_gt_fid_statistics = os.path.join(save_dir_gt_fid_statistics, "gt_statistics.npz")

    mode = "features_gt"
    # "features_gt" -> (x, y) in dataloader, compute features of x
    # "features_pred" -> (x, y) in dataloader, comput features of f(y)
    if rank == 0:
        m1_test_gt, s1_test_gt, _ = compute_statistics_of_path_or_dataloader(path_to_save_gt_fid_statistics,
                                                                        test_dataloader_target, model_fid,
                                                                        args.batch_size, dims, device,
                                                                        mode, None, None, None)

        if not os.path.exists(path_to_save_gt_fid_statistics):
            print(f"saving stats for gt test data to {path_to_save_gt_fid_statistics}")
            np.savez(path_to_save_gt_fid_statistics, mu=m1_test_gt, sigma=s1_test_gt)

    use_minibatch_ot = args.use_minibatch_ot
    if use_minibatch_ot:
        ot_plan_sampler = OTPlanSampler('exact')
        print(f"Use minibatch OT!")
    else:
        print(f"Don't use minibatch OT!")
        ot_plan_sampler = None

    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)
        print(f"epoch {epoch}")

        if rank == 0:
            if args.use_ema:
                with ema_g.average_parameters():
                    save_and_log_images(train_data_loader, device, use_minibatch_ot, ot_plan_sampler, args, True, pos_coeff,
                                        netG, batch_size, writer,
                                        num_of_samples_for_trajectories_to_visualize, exp_path, global_step,
                                        num_trajectories, True)
            save_and_log_images(train_data_loader, device, use_minibatch_ot, ot_plan_sampler, args, False, pos_coeff,
                                netG, batch_size, writer,
                                num_of_samples_for_trajectories_to_visualize, exp_path, global_step,
                                num_trajectories, True)
            
            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content and do validation')
                    if args.use_ema:
                        with ema_g.average_parameters():
                            validate_metrics(test_dataloader_input, netG, args, exp_path, epoch, device, pos_coeff, T, writer,
                                                num_of_samples_for_trajectories_to_visualize, global_step, num_trajectories,
                                                m1_test_gt, s1_test_gt, True)
                            content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                                       'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                                       'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                                       'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}

                            torch.save(content, os.path.join(exp_path, f'content_{epoch}_ema_True.pth'))

                    print(f"-----------")
                    validate_metrics(test_dataloader_input, netG, args, exp_path, epoch, device, pos_coeff, T, writer,
                                    num_of_samples_for_trajectories_to_visualize, global_step, num_trajectories,
                                    m1_test_gt, s1_test_gt, False)
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                                'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                                'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                                'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}

                    torch.save(content, os.path.join(exp_path, f'content_{epoch}_ema_False.pth'))


        for iteration, (x, y) in enumerate(train_data_loader):
            for p in netD.parameters():  
                p.requires_grad = True  

            netD.zero_grad()
            
            # sample from p(x_0)
            real_data = x.to(device, non_blocking=True)
            input_real_data = y.to(device, non_blocking=True)

            if use_minibatch_ot:
                with torch.no_grad():
                    real_data, input_real_data = ot_plan_sampler.sample_plan(real_data, input_real_data)
                        
            if args.posterior == "ddpm":
                t = torch.randint(0, args.num_timesteps, (1,), device=device).repeat(real_data.size(0))
                x_t, x_tp1 = q_sample_supervised_pairs(pos_coeff, real_data, t, input_real_data)
            elif args.posterior == "brownian_bridge":
                t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
                x_t, x_tp1 = q_sample_supervised_pairs_brownian(pos_coeff, real_data, t, input_real_data)
                
            # x_t, x_tp1 = q_sample_supervised_pairs(pos_coeff, real_data, t, input_real_data)
            x_t.requires_grad = True

            # train with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            
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
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                            )[0]
                    grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()

                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

            # train with fake
            latent_z = torch.randn(batch_size, nz, device=device)

            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

            errD_fake = F.softplus(output)
            errD_fake = errD_fake.mean()
            errD_fake.backward()

            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            
            # x_t, x_tp1 = q_sample_supervised_pairs(pos_coeff, real_data, t, input_real_data)
            if args.posterior == "ddpm":
                t = torch.randint(0, args.num_timesteps, (1,), device=device).repeat(real_data.size(0))
                x_t, x_tp1 = q_sample_supervised_pairs(pos_coeff, real_data, t, input_real_data)
            elif args.posterior == "brownian_bridge":
                t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
                x_t, x_tp1 = q_sample_supervised_pairs_brownian(pos_coeff, real_data, t, input_real_data)

            latent_z = torch.randn(batch_size, nz, device=device)

            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

            errG = F.softplus(-output)
            errG = errG.mean()
            
            errG.backward()
            optimizerG.step()
            if args.use_ema:
                ema_g.update()

            global_step += 1
            if iteration % 10 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(epoch, iteration, errG.item(),
                                                                                errD.item()))
                    # iter_gloval = iteration + epoch * data_loader_size
                    writer.add_scalar(global_step, "G_loss", errG.item())
                    print(f"sending G_loss at global step {global_step} to {writer}!")
                    writer.add_scalar(global_step, "D_loss", errD.item())
                    print(f"sending D_loss at global step {global_step} to {writer}!")
        
        if not args.no_lr_decay:
            
            schedulerG.step()
            schedulerD.step()



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
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    
    parser.add_argument('--resume', action='store_true',default=False)
    
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')

    parser.add_argument('--posterior', type=str, default='ddpm',
                        help='type of posterior to use')

    # ddpm prior
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
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
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--data_root', default='')
    parser.add_argument('--paired', action='store_true', default=False)
    parser.add_argument('--use_minibatch_ot', action='store_true', default=False)

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    
    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    
    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true',default=False)
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
        processes = []
        torch.multiprocessing.set_start_method("spawn")
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
   
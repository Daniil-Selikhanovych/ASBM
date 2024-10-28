import os
import shutil

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tqdm
from functools import partial

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import TensorDataset

from torch.multiprocessing import Process
import torch.distributed as dist

from posteriors import Diffusion_Coefficients, BrownianPosterior_Coefficients, \
    get_time_schedule

from train_asbm_swissroll import MyGenerator, MyDiscriminator, dotdict, my_swiss_roll, GaussianDist, OneDistSampler, \
    q_sample_supervised_pairs_brownian, sample_posterior
from test_swiss_roll import sample_from_model_bb, sample_from_model

from utils import TensorBoardWriter
from torch_ema import ExponentialMovingAverage

from sklearn import datasets


class XSampler:
    def __init__(self, sampler):
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


def save_and_log_images(x_gt_sampler, y_gt_sampler, device, args, pos_coeff, netG, 
                        iteration, writer, exp_path, imf_num_iter, fw_or_bw):
    if fw_or_bw == 'fw':
        x = x_gt_sampler.sample(args.batch_size)
        y = y_gt_sampler.sample(args.batch_size)

        x_0, x_t_1 = x.to(device), y.to(device)
        print(f"x_0.shape = {x_0.shape}, x_t_1.shape = {x_t_1.shape}")

        fig, ax = plt.subplots(1, 1, figsize=(4., 4.), dpi=200)

        ax.grid(zorder=-20)
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

        tr_samples_init = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
        tr_samples_init_device = tr_samples_init.to(torch.float32).to(device, non_blocking=True)
        tr_samples_init_device_randn = torch.randn_like(tr_samples_init_device)

        tr_samples = tr_samples_init[None].repeat(3, 1, 1).reshape(12, 2)

        y_pred, _ = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, args)
        y_pred = y_pred.detach().cpu().numpy()

        n_inter_points = 100
        trajectory, x_traj_prediction = sample_from_model_bb(pos_coeff, netG, args.num_timesteps, tr_samples_init_device,
                                                             args, n_inter_points)
        trajectory = trajectory.detach().cpu().numpy()
        x_traj_prediction = x_traj_prediction.detach().cpu().numpy()
        print(f"trajectory.shape = {trajectory.shape}")

        print(f"min y_pred_x = {np.min(y_pred[:, 0])}, max y_pred_x = {np.max(y_pred[:, 0])}")
        print(f"min y_pred_y = {np.min(y_pred[:, 1])}, max y_pred_y = {np.max(y_pred[:, 1])}")

        y_pred_randn, _ = sample_from_model(pos_coeff, netG, args.num_timesteps, tr_samples_init_device_randn, args)

        ax.scatter(y_pred[:, 0], y_pred[:, 1],
                   c="salmon", s=64, edgecolors="black", label="Fitted distribution", zorder=1)

        ax.scatter(tr_samples[:, 0], tr_samples[:, 1],
                   c="lime", s=128, edgecolors="black", label=r"Trajectory start ($x \sim p_0$)", zorder=3)

        ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1],
                   c="yellow", s=64, edgecolors="black", label=r"Trajectory end (fitted)", zorder=3)

        num_samples = 4
        for i in range(num_samples):
            if i == 0:
                ax.plot(trajectory[::1, i, 0], trajectory[::1, i, 1], "black", markeredgecolor="black",
                        linewidth=1.5, zorder=2, label=r"Trajectory of $S_{\theta}$")
            else:
                ax.plot(trajectory[::1, i, 0], trajectory[::1, i, 1], "black", markeredgecolor="black",
                        linewidth=1.5, zorder=2)

        print(f"x_traj_prediction.shape = {x_traj_prediction.shape}")
        for i in range(num_samples):
            if i == 0:
                ax.plot(x_traj_prediction[:, i, 0], x_traj_prediction[:, i, 1], color="blue",
                        label="Intermediate predictions")
            else:
                ax.plot(x_traj_prediction[:, i, 0], x_traj_prediction[:, i, 1], color="blue")

        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])

        ax.legend(loc="lower left")

        fig.tight_layout(pad=0.1)

        fig_name = f'evolution_step_imf_iter_{imf_num_iter}_mode_{fw_or_bw}.png'
        fig_path = os.path.join(exp_path, fig_name)
        print(f"saving {fig_path}")
        plt.savefig(fig_path)
        img_to_tensorboard = Image.open(fig_path)
        img_to_tensorboard = transforms.ToTensor()(img_to_tensorboard)[:3]
        if writer is not None:
            writer.add_image(iteration, f"trajectories_imf_iter_{imf_num_iter}_mode_{fw_or_bw}",
                             img_to_tensorboard)


def markovian_projection(x_gt_sampler, y_gt_sampler, max_iter, args, condsampler, netG_proj, netD_proj,
                         opt_G_proj, opt_D_proj, ema_g, pos_coeff, device, exp_path, imf_num_iter, T,
                         writer, D_opt_steps=1, fw_or_bw='fw'):
    print(f"Mode {fw_or_bw}, start markovian projection on iteration = {imf_num_iter}")
    print(f"Will save EMA every {args.save_content_every} iteration")

    for iteration in tqdm.tqdm(range(max_iter)):
        if args.save_content:
            if iteration % args.save_content_every == 0:
                if args.use_ema:
                    with ema_g.average_parameters():
                        print('Saving content in EMA.')
                        content = {'iteration': iteration, 'imf_num_iter': imf_num_iter, 'fw_or_bw': fw_or_bw,
                                   'netG_dict': netG_proj.state_dict(),
                                   'optimizerG': opt_G_proj.state_dict(),
                                   'netD_dict': netD_proj.state_dict(),
                                   'optimizerD': opt_D_proj.state_dict()}

                        path_to_save_ckpt = os.path.join(exp_path,
                                                         f'content_{fw_or_bw}_imf_num_iter_{imf_num_iter}_{iteration}.pth')

                        torch.save(content, path_to_save_ckpt)
                        print(f"Saving {path_to_save_ckpt}")

                        netG_proj.eval()

                        with torch.no_grad():
                            exp_path_mode = os.path.join(exp_path, fw_or_bw)
                            os.makedirs(exp_path_mode, exist_ok=True)
                            exp_path_mode_num_iter = os.path.join(exp_path_mode, str(imf_num_iter))
                            os.makedirs(exp_path_mode_num_iter, exist_ok=True)
                            if args.use_ema:
                                with ema_g.average_parameters():
                                    save_and_log_images(x_gt_sampler, y_gt_sampler, device, args, pos_coeff,
                                                        netG_proj, iteration, writer, exp_path, imf_num_iter,
                                                        fw_or_bw)
                            else:
                                save_and_log_images(x_gt_sampler, y_gt_sampler, device, args, pos_coeff,
                                                    netG_proj, iteration, writer, exp_path, imf_num_iter,
                                                    fw_or_bw)

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

            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_supervised_pairs_brownian(pos_coeff, real_data, t, input_real_data)

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
            writer.add_scalar(iteration, f"G_loss_imf_iter_{imf_num_iter}_mode_{fw_or_bw}", errG.item())
            writer.add_scalar(iteration, f"D_loss_imf_iter_{imf_num_iter}_mode_{fw_or_bw}", errD.item())

        if iteration % 100 == 0:
            print('Markovain proj {} on IMF iter {}: Iter {}, G Loss: {}, D Loss: {}'.format(fw_or_bw,
                                                                                             imf_num_iter,
                                                                                             iteration,
                                                                                             errG.item(),
                                                                                             errD.item()))


class SwissDist(OneDistSampler):
    def __init__(self, noise) -> None:
        super().__init__()
        self.noise = noise

    def sample(self, batch_size):
        sampled_batch = datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=self.noise
        )[0].astype('float32')[:, [0, 2]] / 7.5
        return torch.tensor(sampled_batch)


# %%
def train(gpu, args):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device('cuda:{}'.format(gpu))

    inner_imf_mark_proj_iters = args.inner_imf_mark_proj_iters

    exp = args.exp
    exp_path = "./saved_info/dd_gan_imf/{}_{}/{}".format(args.dataset_forward, args.dataset_backward, exp)
    print(f"will save results in {exp_path}")

    exp_forward = args.exp_forward
    exp_backward = args.exp_backward
    parent_dir_forward = "./saved_info/dd_gan/{}".format(args.dataset_forward)
    parent_dir_backward = "./saved_info/dd_gan/{}".format(args.dataset_backward)

    exp_path_fw = os.path.join(parent_dir_forward, exp_forward)
    exp_path_bw = os.path.join(parent_dir_backward, exp_backward)

    dir_name = f"{args.dataset_forward}_{args.dataset_backward}"

    checkpoint_file_fw = os.path.join(exp_path_fw, args.exp_forward_model)
    checkpoint_file_bw = os.path.join(exp_path_bw, args.exp_backward_model)

    num_iterations = 200000  # 200000
    opt = {
        'nz': 1,
        'num_timesteps': 4,
        'x_dim': 2,
        't_dim': 2,
        'out_dim': 2,
        'beta_min': 0.1,
        'beta_max': 20.,
        'layers_G': [256, 256, 256],
        'layers_D': [256, 256, 256],
        'num_iterations': num_iterations,
        'rank': 0,
        'global_rank': 0,
        'batch_size': 512,
        'lr_d': 1e-4,
        'lr_g': 1e-4,
        'beta1': 0.5,
        'beta2': 0.9,
        'r1_gamma': 0.01,
        'lazy_reg': 1,
        'use_ema': True,
        'ema_decay': 0.999,
        'epsilon': args.epsilon,
        'sampler_precalc': 1000,
        'sample_func': partial(
            my_swiss_roll,
            noise=0.8
        ),
        'dataset_forward': args.dataset_forward,
        'dataset_backward': args.dataset_backward,
        'imf_iters': args.imf_iters,
        'exp': exp,
        'exp_path': exp_path,
        'save_ckpt': True,
        'save_ckpt_every': 5000,
        'save_content': True,
        'save_content_every': 5000,
        'visualize': True,
        'visualize_every': 1000,
        'print': True,
        'print_every': 100,
        'resume': False,
        'D_opt_steps': 1
    }
    args = dotdict(opt)

    print(f"Training, exp = {args.exp}")

    writer = TensorBoardWriter(args, path_to_save="./saved_info/dd_gan_imf",
                               dir_name=dir_name)

    x_gt_sampler = SwissDist(noise=0.8)
    y_gt_sampler = GaussianDist(dim=2)

    # data_loader_size = len(data_loader)
    # print(f'data_loader size = {data_loader_size}')
    nz = args.nz  # latent dimension

    netG_fw = MyGenerator(
        x_dim=args.x_dim,
        t_dim=args.t_dim,
        n_t=args.num_timesteps,
        out_dim=args.out_dim,
        z_dim=nz,
        layers=args.layers_G
    ).to(device)
    netG_bw = MyGenerator(
        x_dim=args.x_dim,
        t_dim=args.t_dim,
        n_t=args.num_timesteps,
        out_dim=args.out_dim,
        z_dim=nz,
        layers=args.layers_G
    ).to(device)

    netD_fw = MyDiscriminator(
        x_dim=args.x_dim,
        t_dim=args.t_dim,
        n_t=args.num_timesteps,
        layers=args.layers_D
    ).to(device)
    netD_bw = MyDiscriminator(
        x_dim=args.x_dim,
        t_dim=args.t_dim,
        n_t=args.num_timesteps,
        layers=args.layers_D
    ).to(device)

    optimizerG_fw = optim.Adam(netG_fw.parameters(),
                               lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizerD_fw = optim.Adam(netD_fw.parameters(),
                               lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG_bw = optim.Adam(netG_bw.parameters(),
                               lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizerD_bw = optim.Adam(netD_bw.parameters(),
                               lr=args.lr_d, betas=(args.beta1, args.beta2))

    # ddp
    # netG_fw = nn.parallel.DistributedDataParallel(netG_fw, device_ids=[gpu])
    # netD_fw = nn.parallel.DistributedDataParallel(netD_fw, device_ids=[gpu])
    # netG_bw = nn.parallel.DistributedDataParallel(netG_bw, device_ids=[gpu])
    # netD_bw = nn.parallel.DistributedDataParallel(netD_bw, device_ids=[gpu])

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = BrownianPosterior_Coefficients(args, device)

    T = get_time_schedule(args, device)

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

    imf_iters = args.imf_iters

    print(f"Start to run IMF with {imf_iters} iterations")

    save_and_log_images(x_gt_sampler, y_gt_sampler, device, args, pos_coeff, netG_fw, 
                        -1, writer, exp_path, -1, 'fw')

    for imf_num_iter in range(imf_iters):
        # ----Forward model learning (y -> x)----

        bmgan_sample_bw = lambda x: sample_from_model(pos_coeff, netG_bw,
                                                      args.num_timesteps, x.to(device),
                                                      args)[0]

        if args.use_ema:
            condsampler_fw = ModelCondSampler(x_gt_sampler, bmgan_sample_bw, ema_model=ema_g_bw)
        else:
            condsampler_fw = ModelCondSampler(x_gt_sampler, bmgan_sample_bw)


        markovian_projection(x_gt_sampler, y_gt_sampler, inner_imf_mark_proj_iters, args, condsampler_fw, netG_fw, netD_fw,
                             optimizerG_fw, optimizerD_fw, ema_g_fw, pos_coeff, device, exp_path, imf_num_iter,
                             T, writer, D_opt_steps=args.D_opt_steps, fw_or_bw='fw')

        bmgan_sample_fw = lambda x: sample_from_model(pos_coeff, netG_fw,
                                                      args.num_timesteps, x.to(device),
                                                      args)[0]

        if args.use_ema:
            condsampler_bw = ModelCondSampler(y_gt_sampler, bmgan_sample_fw, ema_model=ema_g_fw)
        else:
            condsampler_bw = ModelCondSampler(y_gt_sampler, bmgan_sample_fw)

        markovian_projection(y_gt_sampler, x_gt_sampler, inner_imf_mark_proj_iters, args, condsampler_bw, netG_bw, netD_bw,
                             optimizerG_bw, optimizerD_bw, ema_g_bw, pos_coeff, device, exp_path, imf_num_iter,
                             T, writer, D_opt_steps=args.D_opt_steps, fw_or_bw='bw')


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
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--dataset_name', type=str, default="swiss_roll")

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

    parser.add_argument('--inner_imf_mark_proj_iters', type=int, default=20000)
    parser.add_argument('--imf_iters', type=int, default=20)

    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=50, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')

    args = parser.parse_args()
    train(0, args)

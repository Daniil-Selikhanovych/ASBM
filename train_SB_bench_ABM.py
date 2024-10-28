import os
from functools import partial
import argparse

from copy import deepcopy
import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from PIL import Image
import wandb

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from train_asbm import Diffusion_Coefficients, BrownianPosterior_Coefficients, get_time_schedule,\
    q_sample_pairs, extract

from utils import TensorBoardWriter
from torch_ema import ExponentialMovingAverage
from sklearn import datasets

from eot_bench import EOTGMMSampler, MLPTimeDiscriminator, MLPTimeGenerator
from eot_benchmark.metrics import (
    compute_BW_UVP_by_gt_samples
)

from discrete_ot import OTPlanSampler
ot_plan_sampler = OTPlanSampler('exact')


from bench_utils import compute_condBWUVP, pca_plot

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MyGenerator(nn.Module):
    def __init__(
        self, x_dim=2, t_dim=2, n_t=4, z_dim=1, out_dim=2, layers=[128, 128, 128],
        active=partial(nn.LeakyReLU, 0.2),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.t_dim = t_dim
        self.z_dim = z_dim

        self.model = []
        ch_prev = x_dim + t_dim + z_dim

        self.t_transform = nn.Embedding(n_t, t_dim,)

        for ch_next in layers:
            self.model.append(nn.Linear(ch_prev, ch_next))
            self.model.append(active())
            ch_prev = ch_next

        self.model.append(nn.Linear(ch_prev, out_dim))
        self.model = nn.Sequential(*self.model)

    def forward(self, x, t, z):
        batch_size = x.shape[0]

        if z.shape != (batch_size, self.z_dim):
            z = z.reshape((batch_size, self.z_dim))

        return self.model(
            torch.cat([
                x,
                self.t_transform(t),
                z,
            ], dim=1)
        )


class MyDiscriminator(nn.Module):
    def __init__(
        self, x_dim=2, t_dim=2, n_t=4, layers=[128, 128, 128],
        active=partial(nn.LeakyReLU, 0.2),
    ):
        super().__init__()

        self.x_dim = x_dim
        self.t_dim = t_dim

        self.model = []
        ch_prev = 2 * x_dim + t_dim

        self.t_transform = nn.Embedding(n_t, t_dim,)

        for ch_next in layers:
            self.model.append(nn.Linear(ch_prev, ch_next))
            self.model.append(active())
            ch_prev = ch_next

        self.model.append(nn.Linear(ch_prev, 1))
        self.model = nn.Sequential(*self.model)

    def forward(self, x_t, t, x_tp1,):
        transform_t = self.t_transform(t)
        # print(f"x_t.shape = {x_t.shape}, transform_t = {transform_t.shape}, x_tp1 = {x_tp1.shape}")

        return self.model(
            torch.cat([
                x_t,
                transform_t,
                x_tp1,
            ], dim=1)
        ).squeeze()


def my_swiss_roll(n_samples=100, noise=0.8):
    sampled_batch = datasets.make_swiss_roll(
        n_samples=n_samples,
        noise=noise
    )[0].astype('float32')[:, [0, 2]] / 7.5

    return sampled_batch


class MySampler:
    def __init__(self, batch_size, sample_func=my_swiss_roll, precalc=None):
        self.precalc = precalc
        self.batch_size = batch_size
        self.sample_func = sample_func
        print(f"sample_func = {sample_func}")

        if self.precalc is not None:
            self.regenerate()

    def regenerate(self):
        self.generated = self.sample_func(self.precalc * self.batch_size,)
        self.idx = 0

    def sample(self):
        if self.precalc is None:
            return self.sample_func(self.batch_size,)
        if self.idx == self.precalc:
            self.regenerate()
        ret = self.generated[self.idx * self.batch_size: (self.idx + 1) * self.batch_size]
        self.idx += 1
        return ret


def sample_posterior(coefficients, x_0, x_t, t):

    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped


    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))
        while len(nonzero_mask.shape) < len(mean.shape):
            nonzero_mask = nonzero_mask.unsqueeze(-1)

        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x


class OneDistSampler:
    def __init__(self) -> None:
        pass

    def __call__(self, batch_size):
        return self.sample(batch_size)

    def sample(self, batch_size):
        raise NotImplementedError('Abstract Class')

    def __str__(self):
        return self.__class__.__name__


class GaussianDist(OneDistSampler):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def sample(self, batch_size):
        return torch.randn([batch_size, self.dim])


def q_sample_supervised_pairs_brownian(pos_coeff, x_start, t, x_end):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    num_steps = pos_coeff.posterior_mean_coef1.shape[0]
    t_plus_one_tensor = ((t + 1) / num_steps)[:, None]

    x_t_plus_one = t_plus_one_tensor * x_end + (1.0 - t_plus_one_tensor) * x_start + torch.sqrt(
        pos_coeff.epsilon * t_plus_one_tensor * (1 - t_plus_one_tensor)) * noise

    x_t = sample_posterior(pos_coeff, x_start, x_t_plus_one, t)

    return x_t, x_t_plus_one


def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = args.batch_size
    nz = args.nz  # latent dimension

    if args.pos_enc == 0:
        netG = MyGenerator(
            x_dim=args.dim,
            t_dim=args.t_dim,
            n_t=args.num_timesteps,
            out_dim=args.dim,
            z_dim=nz,
            layers=args.layers_G
        ).to(device)

        netD = MyDiscriminator(
            x_dim=args.dim,
            t_dim=args.t_dim,
            n_t=args.num_timesteps,
            layers=args.layers_D
        ).to(device)

    dim = args.dim
    eps = args.epsilon
    fb = args.fb
    eval_freq = args.eval_freq

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        # optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
        ema_g = ExponentialMovingAverage(netG.parameters(), decay=args.ema_decay)
        ema_g.to(device)
        print(f"use EMA from torch_ema with decay = {args.ema_decay}!")
        print(f"ema_g = {ema_g}")
    else:
        print(f"don't use EMA!")

    coeff = Diffusion_Coefficients(args, device)
    # change here to brownian

    pos_coeff = BrownianPosterior_Coefficients(args, device)
    #     pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    from  datetime import datetime
    datetime_marker_str = datetime.now().strftime("%d:%m:%y_%H:%M:%S")

    save_dir_name = os.path.join('sb_bench_results', f'dim_{dim}_eps_{eps}')

    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)
        
    save_dir_name = os.path.join(save_dir_name, f'T_{args.num_timesteps}_{fb}_{args.plan}')
    
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)

    save_dir_name = os.path.join(save_dir_name, datetime_marker_str + '_exp' + f'_{args.D_opt_steps}')
    os.mkdir(save_dir_name)

    args.global_rank = 0
    
    wandb_config = {'dim': dim, 'eps': eps, 'fb': fb, 'ema': args.use_ema, 'num_iter': args.num_iterations,
                     'dataset': 'SB_bench', 'T': args.num_timesteps, 'plan': args.plan, 'D_opt_steps': args.D_opt_steps, 'pos_enc': args.pos_enc}

    wandb.init(project="BM_GAN", name=f"BMGAN_SB", config=wandb_config)

    global_iteration, init_iteration = 0, 0

    sampler = EOTGMMSampler(dim=dim, eps=eps, batch_size=128, download=False)
    
    if fb == 'b':
        x_0_sampler, x_1_sampler = sampler.x_sample, sampler.y_sample
    else:
        x_1_sampler, x_0_sampler = sampler.x_sample, sampler.y_sample

    history = {
        'D_loss': [],
        'G_loss': [],
    }

    history = dotdict(history)

    print(f"start to train for experiment {save_dir_name}")

    for iteration in tqdm(range(init_iteration, args.num_iterations + 1)):
        #########################
        # Discriminator training
        #########################
        for p in netD.parameters():
            p.requires_grad = True

        netD.zero_grad()
        
        ###################################
        # Sample timestep
        
        # x_0 = gen(x_1)
        
        x_0_samples = x_0_sampler(args.batch_size).to(device, non_blocking=True)
        x_1_samples = x_1_sampler(args.batch_size).to(device, non_blocking=True)

        if args.plan == 'mbot':
            x_0_samples, x_1_samples = ot_plan_sampler.sample_plan(x_0_samples, x_1_samples)

        t = torch.randint(0, args.num_timesteps, (x_0_samples.size(0),), device=device)

        x_t, x_tp1 = q_sample_supervised_pairs_brownian(pos_coeff, x_0_samples, t, x_1_samples)
        
        x_t.requires_grad = True

        ###################################
        # Optimizing loss on real data
        D_real = netD(x_t, t, x_tp1.detach()).view(-1)

        errD_real = F.softplus(-D_real)
        errD_real = errD_real.mean()

        errD_real.backward(retain_graph=True)

        ###################################
        # R_1(\phi) regularization
        if args.lazy_reg is None or iteration % args.lazy_reg == 0:
            grad_real = torch.autograd.grad(
                outputs=D_real.sum(), inputs=x_t, create_graph=True,
            )[0]
            grad_penalty = (
                    grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()

            grad_penalty = args.r1_gamma / 2 * grad_penalty
            grad_penalty.backward()

        ###################################
        # Sample vector from latent space
        # for generation
        
        latent_z = torch.randn(batch_size, nz, device=device)

        ###################################
        # Sample fake output
        # (x_tp1 -> x_0 -> x_t)

        x_0_predict = netG(x_tp1.detach(), t, latent_z)
        x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
        # print(f"x_pos_sample.shape = {x_pos_sample.shape}, x_tp1.shape = {x_tp1.shape}")

        ###################################
        # Optimize loss on fake data
        output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

        errD_fake = F.softplus(output)
        errD_fake = errD_fake.mean()
        errD_fake.backward()

        errD = errD_real + errD_fake

        history.D_loss.append(errD.item())

        ###################################
        # Update weights of netD
        optimizerD.step()

        #############################################################

        #########################
        # Generator training
        #########################
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        ###################################
        # Sample pairs for training

        
        if iteration % args.D_opt_steps == 0:

            x_0_samples = x_0_sampler(args.batch_size).to(device, non_blocking=True)
            x_1_samples = x_1_sampler(args.batch_size).to(device, non_blocking=True)

            if args.plan == 'mbot':
                x_0_samples, x_1_samples = ot_plan_sampler.sample_plan(x_0_samples, x_1_samples)
            
            t = torch.randint(0, args.num_timesteps, (x_0_samples.size(0),), device=device)

            x_t, x_tp1 = q_sample_supervised_pairs_brownian(pos_coeff, x_0_samples, t, x_1_samples)

            ###################################
            # Sample vector from latent space
            # for generation
            latent_z = torch.randn(batch_size, nz, device=device)

            ###################################
            # Sample fake output
            # (x_tp1 -> x_0 -> x_t)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            ###################################
            # Optimize loss on fake data
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

            ###################################
            # Update weights of netG
            errG = F.softplus(-output)
            errG = errG.mean()

            errG.backward()
            optimizerG.step()
            if args.use_ema:
                ema_g.update()
            history.G_loss.append(errG.item())

        if wandb.run:
            wandb.log({'loss_G': errG.item(), 'loss_D': errD.item()})

        if iteration % eval_freq == 100:
            if args.use_ema:
                with ema_g.average_parameters():
                    
                    x_0_samples = x_0_sampler(args.batch_size).to(device, non_blocking=True)
                    x_1_samples = x_1_sampler(args.batch_size).to(device, non_blocking=True)
                    
                    x_fake = sample_from_model(pos_coeff, netG, args.num_timesteps, x_1_samples, T, args).detach()

                    if wandb.run:
                        pca_plot(x_1_samples, x_0_samples, x_fake, n_plot=512, save_name='plot_pca_samples.png', wandb_save_postfix=fb, is_wandb=wandb.run)
                    else:
                        pca_plot(x_1_samples, x_0_samples, x_fake, n_plot=512, save_name=os.path.join(save_dir_name, 'plot_pca_samples.png'), wandb_save_postfix=fb, is_wandb=wandb.run)


                    bmgan_sample = lambda x: sample_from_model(pos_coeff, netG,
                                                                    args.num_timesteps, x.to(device),
                                                                    T, args)
                    
                    condBW_Uvp = compute_condBWUVP(bmgan_sample, dim, eps, n_samples=1000, device='cpu')
                    
                    print(f'CondBW-UVP: {condBW_Uvp}')

                    y_samples = x_1_sampler(10000)
                    x_samples = x_0_sampler(10000)
                    x_pred = bmgan_sample(y_samples)
                    bw_uvp = compute_BW_UVP_by_gt_samples(x_samples, x_pred.cpu())
                    print(f'BW-UVP f: {bw_uvp}')

                if wandb.run:
                    wandb.log({f'condBW_Uvp': condBW_Uvp, f'BW-UVP': bw_uvp})


        if args.print and (iteration + 1) % args.print_every == 0:
            print('iteration: {} | G Loss: {} | D Loss: {}'.format(iteration, errG.item(), errD.item()))

    from copy import deepcopy

    if args.save_ckpt:
        # Save model in the end
        if args.use_ema:

            with ema_g.average_parameters():
                content = {'global_step': args.num_iterations,  # 'args': args,
                           'netG_dict_ema': deepcopy(netG.state_dict()), 'optimizerG': optimizerG.state_dict(),
                           'netD_dict': netD.state_dict(),
                           'optimizerD': optimizerD.state_dict()}
            
            content['netG_dict'] = netG.state_dict()
            
            path_to_save_content = os.path.join(save_dir_name, f'content_{args.num_iterations}.pth')
            print(f"saving {path_to_save_content}")
            torch.save(content, path_to_save_content)
        else:
            content = {'global_step': args.num_iterations,  # 'args': args,
                       'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                       'netD_dict': netD.state_dict(),
                       'optimizerD': optimizerD.state_dict()}
            print(f"saving {os.path.join(save_dir_name, f'content_{args.num_iterations}.pth')}")
            torch.save(content, os.path.join(save_dir_name, f'content_{args.num_iterations}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--dataset_name', type=str, default="swiss_roll")
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--fb', type=str, default='f')
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--num_iterations', type=int, default=50000)
    parser.add_argument('--plan', type=str, default='ind')
    parser.add_argument('--D_opt_steps', type=int, default=1)

    parser.add_argument('--pos_enc', type=int, default=0)

    args = parser.parse_args()

    dataset_name = args.dataset_name

    exp_path = f"./saved_info/dd_gan/{dataset_name}/{args.exp_name}"
    os.makedirs(exp_path, exist_ok=True)

    opt = {
        'nz': 1,
        'num_timesteps': args.num_timesteps,
        'x_dim': 2,
        't_dim': 2,
        'out_dim': 2,
        'beta_min': 0.1,
        'beta_max': 20.,
        'layers_G': [256, 256, 256],
        'layers_D': [256, 256, 256],
        'num_iterations': args.num_iterations,
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
        'dir_name': dataset_name,
        'exp': args.exp_name,
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
        'dim': args.dim,
        'fb': args.fb,
        'eval_freq': args.eval_freq,
        'plan': args.plan,
        'D_opt_steps': args.D_opt_steps,
        'pos_enc': args.pos_enc
    }

    opt = dotdict(opt)

    train_model(opt)

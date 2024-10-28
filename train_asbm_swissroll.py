import os
from functools import partial
import argparse

import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from posteriors import Diffusion_Coefficients, BrownianPosterior_Coefficients, get_time_schedule
from sampling_utils import q_sample_pairs, extract

from utils import TensorBoardWriter
from torch_ema import ExponentialMovingAverage
from sklearn import datasets

from eot_bench import EOTGMMSampler


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


def sample_posterior(coefficients, x_0,x_t, t):

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


def sample_from_model(coefficients, generator, n_time, x_init, opt):
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

    netG = MyGenerator(
        x_dim=args.x_dim,
        t_dim=args.t_dim,
        n_t=args.num_timesteps,
        out_dim=args.out_dim,
        z_dim=nz,
        layers=args.layers_G
    ).to(device)

    netD = MyDiscriminator(
        x_dim=args.x_dim,
        t_dim=args.t_dim,
        n_t=args.num_timesteps,
        layers=args.layers_D
    ).to(device)

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

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_iterations, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_iterations, eta_min=1e-5)

    coeff = Diffusion_Coefficients(args, device)
    # change here to brownian

    pos_coeff = BrownianPosterior_Coefficients(args, device)
    #     pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    args.global_rank = 0
    writer = TensorBoardWriter(args, dir_name=args.dir_name)

    global_iteration, init_iteration = 0, 0
    if args.resume:
        checkpoint_file = os.path.join(args.exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_iteration = checkpoint['iteration']
        global_iteration = init_iteration

        # load G
        netG.load_state_dict(checkpoint['netG_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])

        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])

        print("=> loaded checkpoint (iteration {})"
              .format(init_iteration))

    print(f"sample_func = {args.sample_func}")

    swiss_sampler = MySampler(
        batch_size=args.batch_size,
        sample_func=args.sample_func,
        precalc=args.precalc,
    )
    gaussian_sampler = GaussianDist(dim=2)
    if args.dir_name in ['benchmark_dim_2_forward', 'benchmark_dim_2_backward']:
        sampler = EOTGMMSampler(dim=2, eps=args.epsilon, batch_size=128, download=False)
        x_0_sampler, x_1_sampler = sampler.x_sample, sampler.y_sample

    history = {
        'D_loss': [],
        'G_loss': [],
    }
    history = dotdict(history)

    print(f"start to train for experiment {args.dir_name}")

    for iteration in tqdm(range(init_iteration, args.num_iterations + 1)):
        #########################
        # Discriminator training
        #########################
        for p in netD.parameters():
            p.requires_grad = True

        netD.zero_grad()

        ###################################
        # Sample real data
        if args.dir_name in ['swiss_roll', 'swiss_roll_to_gaussian']:
            x_real = swiss_sampler.sample()
            real_data = torch.from_numpy(x_real).to(torch.float32).to(device, non_blocking=True)
        elif args.dir_name == 'benchmark_dim_2_forward':
            x_real = x_1_sampler(args.batch_size).to(device, non_blocking=True)
            real_data = x_real.clone()
        elif args.dir_name == 'benchmark_dim_2_backward':
            x_real = x_0_sampler(args.batch_size).to(device, non_blocking=True)
            real_data = x_real.clone()

        ###################################
        # Sample timesteps
        t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

        if args.dir_name == 'swiss_roll':
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
        elif args.dir_name == 'swiss_roll_to_gaussian':
            # print(f"x_t.shape = {x_t.shape}, x_tp1 = {x_tp1.shape}")
            target_real_data = gaussian_sampler.sample(args.batch_size).to(torch.float32).to(device, non_blocking=True)
            # print(f"target_real_data.shape = {target_real_data.shape}, real_data = {real_data.shape}")
            x_t, x_tp1 = q_sample_supervised_pairs_brownian(pos_coeff, target_real_data, t, real_data)
            # print(f"x_t.shape = {x_t.shape}, x_tp1 = {x_tp1.shape}")
        elif args.dir_name == 'benchmark_dim_2_forward':
            target_real_data = x_0_sampler(args.batch_size).to(device, non_blocking=True)
            # print(f"target_real_data.shape = {target_real_data.shape}, real_data = {real_data.shape}")
            x_t, x_tp1 = q_sample_supervised_pairs_brownian(pos_coeff, target_real_data, t, real_data)
        elif args.dir_name == 'benchmark_dim_2_backward':
            target_real_data = x_1_sampler(args.batch_size).to(device, non_blocking=True)
            # print(f"target_real_data.shape = {target_real_data.shape}, real_data = {real_data.shape}")
            x_t, x_tp1 = q_sample_supervised_pairs_brownian(pos_coeff, target_real_data, t, real_data)

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
        # Sample timesteps
        t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

        ###################################
        # Sample pairs for training
        if args.dir_name == 'swiss_roll':
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
        elif args.dir_name == 'swiss_roll_to_gaussian':
            x_t, x_tp1 = q_sample_supervised_pairs_brownian(pos_coeff, target_real_data, t, real_data)
        elif args.dir_name in ['benchmark_dim_2_forward', 'benchmark_dim_2_backward']:
            x_t, x_tp1 = q_sample_supervised_pairs_brownian(pos_coeff, target_real_data, t, real_data)

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

        # LR-Scheduling step
        schedulerG.step()
        schedulerD.step()

        if args.visualize and iteration % args.visualize_every == 0:
            if args.use_ema:
                with ema_g.average_parameters():
                    if args.dir_name == 'swiss_roll':
                        x_t_1 = torch.randn_like(real_data)
                    elif args.dir_name == 'swiss_roll_to_gaussian':
                        x_real = swiss_sampler.sample()
                        x_t_1 = torch.from_numpy(x_real).to(torch.float32).to(device, non_blocking=True)
                    elif args.dir_name == 'benchmark_dim_2_forward':
                        x_real = x_1_sampler(args.batch_size).to(device, non_blocking=True)
                        x_t_1 = x_real.clone()
                    elif args.dir_name == 'benchmark_dim_2_backward':
                        x_real = x_0_sampler(args.batch_size).to(device, non_blocking=True)
                        x_t_1 = x_real.clone()
                    x_fake = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, args).detach().cpu().numpy()

                    print(f"min y_pred_x = {np.min(x_fake[:, 0])}, max y_pred_x = {np.max(x_fake[:, 0])}")
                    print(f"min y_pred_y = {np.min(x_fake[:, 1])}, max y_pred_y = {np.max(x_fake[:, 1])}")

                    print(f"G_loss = {history.G_loss[-1]}")
                    print(f"D_loss = {history.D_loss[-1]}")

                    writer.add_scalar(iteration, "G_loss", history.G_loss[-1])
                    writer.add_scalar(iteration, "D_loss", history.D_loss[-1])

                    fig, ax = plt.subplots(2, 1, figsize=(8, 2 * 6))

                    if args.dir_name == 'swiss_roll':
                        ax[0].scatter(x_real[:, 0], x_real[:, 1], s=1, label='real')
                    elif args.dir_name == 'swiss_roll_to_gaussian':
                        x_plot = torch.randn_like(real_data).detach().cpu().numpy()
                        ax[0].scatter(x_plot[:, 0], x_plot[:, 1], s=1, label='real')
                    elif args.dir_name == 'benchmark_dim_2_forward':
                        x_plot = x_0_sampler(args.batch_size).detach().cpu().numpy()
                        ax[0].scatter(x_plot[:, 0], x_plot[:, 1], s=1, label='real')
                    elif args.dir_name == 'benchmark_dim_2_backward':
                        x_plot = x_1_sampler(args.batch_size).detach().cpu().numpy()
                        ax[0].scatter(x_plot[:, 0], x_plot[:, 1], s=1, label='real')
                    ax[0].set_aspect('equal', adjustable='box')
                    xlim = ax[0].get_xlim()
                    ylim = ax[0].get_ylim()
                    ax[1].scatter(x_fake[:, 0], x_fake[:, 1], s=1, label='fake')
                    ax[1].set_aspect('equal', adjustable='box')
                    ax[1].set_xlim(xlim)
                    ax[1].set_ylim(ylim)
                    ax[0].set_title('Real data')
                    ax[1].set_title('Fake data')

                    path_to_save_figures = os.path.join(args.exp_path,
                                                        f'visualization_{iteration}.png')
                    plt.savefig(path_to_save_figures)
                    print(f"saving {path_to_save_figures}")
                    img_to_tensorboard = Image.open(path_to_save_figures)
                    img_to_tensorboard = transforms.ToTensor()(img_to_tensorboard)[:3]
                    writer.add_image(iteration, "visualization", img_to_tensorboard)

        if args.print and (iteration + 1) % args.print_every == 0:
            print('iteration: {} | G Loss: {} | D Loss: {}'.format(iteration, errG.item(), errD.item()))

        if args.exp_path is not None and args.save_content and (iteration + 1) % args.save_content_every == 0:
            print('Saving content.')
            if args.use_ema:
                with ema_g.average_parameters():
                    content = {
                        'iteration': iteration,
                        # 'args': args,
                        'netG_dict': netG.state_dict(),
                        'optimizerG': optimizerG.state_dict(),
                        'schedulerG': schedulerG.state_dict(),
                        'netD_dict': netD.state_dict(),
                        'optimizerD': optimizerD.state_dict(),
                        'schedulerD': schedulerD.state_dict(),
                    }
                torch.save(content, os.path.join(args.exp_path, f'content_{iteration}.pth'))
            else:
                content = {
                    'iteration': iteration,
                    # 'args': args,
                    'netG_dict': netG.state_dict(),
                    'optimizerG': optimizerG.state_dict(),
                    'schedulerG': schedulerG.state_dict(),
                    'netD_dict': netD.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    'schedulerD': schedulerD.state_dict(),
                }
                torch.save(content, os.path.join(args.exp_path, f'content_{iteration}.pth'))

            print(f"saving {os.path.join(args.exp_path, f'content_{iteration}.pth')}")

        if args.save_ckpt and args.exp_path is not None and (iteration + 1) % args.save_ckpt_every == 0:
            if args.use_ema:
                with ema_g.average_parameters():
                    torch.save(netG.state_dict(), os.path.join(args.exp_path, 'netG_{}.pth'.format(iteration)))
                    torch.save(netD.state_dict(), os.path.join(args.exp_path, 'netD_{}.pth'.format(iteration)))
            else:
                torch.save(netG.state_dict(), os.path.join(args.exp_path, 'netG_{}.pth'.format(iteration)))
                torch.save(netD.state_dict(), os.path.join(args.exp_path, 'netD_{}.pth'.format(iteration)))

    if args.save_ckpt and args.exp_path is not None:
        # Save model in the end
        if args.use_ema:
            with ema_g.average_parameters():
                content = {'global_step': args.num_iterations,  # 'args': args,
                           'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                           'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                           'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
                path_to_save_content = os.path.join(args.exp_path, f'content_{args.num_iterations}.pth')
                print(f"saving {path_to_save_content}")
                torch.save(content, path_to_save_content)
        else:
            content = {'global_step': args.num_iterations,  # 'args': args,
                       'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                       'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                       'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
            print(f"saving {os.path.join(args.exp_path, f'content_{args.num_iterations}.pth')}")
            torch.save(content, os.path.join(args.exp_path, f'content_{args.num_iterations}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--dataset_name', type=str, default="swiss_roll")
    args = parser.parse_args()

    num_iterations = 200000  # 200000

    dataset_name = args.dataset_name

    exp_path = f"./saved_info/dd_gan/{dataset_name}/{args.exp_name}"
    os.makedirs(exp_path, exist_ok=True)

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
    }
    opt = dotdict(opt)

    train_model(opt)

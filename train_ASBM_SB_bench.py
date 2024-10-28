import os
import shutil

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tqdm
from functools import partial
import wandb

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import TensorDataset

from torch.multiprocessing import Process
import torch.distributed as dist

from train_asbm import Diffusion_Coefficients, BrownianPosterior_Coefficients, \
    get_time_schedule

from train_asbm_swissroll import MyGenerator, MyDiscriminator, dotdict, my_swiss_roll, GaussianDist, OneDistSampler, \
    q_sample_supervised_pairs_brownian, sample_posterior
from test_swiss_roll import sample_from_model_bb, sample_from_model

from utils import TensorBoardWriter
from torch_ema import ExponentialMovingAverage

from sklearn import datasets

from eot_benchmark.metrics import (
    compute_BW_UVP_by_gt_samples
)

from eot_bench import EOTGMMSampler
from eot_benchmark.metrics import (
    compute_BW_UVP_by_gt_samples
)
from bench_utils import compute_condBWUVP, pca_plot

# import logging
# logger = logging.getLogger(__name__)

class XSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def sample(self, size=5):
        return self.sampler.sample(size)[0]


class ModelCondSampler:
    def __init__(self, sample_fn: XSampler, model_sample_fn, ema_model=None):
        self.model_sample_fn = model_sample_fn
        self.sample_fn = sample_fn
        self.ema_model = ema_model
        if self.ema_model is not None:
            print(f"Will use EMA for ModelCondSampler sampling")
        else:
            print(f"Will not use EMA for ModelCondSampler sampling")

    def sample(self, size=5):
        sample_x = self.sample_fn(size)
        if self.ema_model is not None:
            with self.ema_model.average_parameters():
                sample_y = self.model_sample_fn(sample_x)
        else:
            sample_y = self.model_sample_fn(sample_x)
        return sample_x, sample_y


def markovian_projection(x_gt_sampler, y_gt_sampler, max_iter, args, condsampler, netG_proj, netD_proj,
                         opt_G_proj, opt_D_proj, ema_g, pos_coeff, device, save_dir_name, imf_num_iter, T, dim, eps,
                         D_opt_steps=1, fw_or_bw='fw'):
    
    # gen(y) = x

    print(f"Mode {fw_or_bw}, start markovian projection on iteration = {imf_num_iter}")
    print(f"Will save EMA every {args.save_content_every} iteration")

    for iteration in tqdm.tqdm(range(max_iter)):
        
        if iteration % args.eval_freq == 0:

            netG_proj.eval()

            with torch.no_grad():
                
                if args.use_ema:
                    with ema_g.average_parameters():
                        
                        x_0_samples = x_gt_sampler(args.batch_size).to(device, non_blocking=True)
                        x_1_samples = y_gt_sampler(args.batch_size).to(device, non_blocking=True)
                        
                        x_fake = sample_from_model(pos_coeff, netG_proj, args.num_timesteps, x_1_samples, T, args)[0].detach()
                        
                        if wandb.run:
                            pca_plot(x_1_samples, x_0_samples, x_fake, n_plot=512, save_name='plot_pca_samples.png', wandb_save_postfix=fw_or_bw, is_wandb=wandb.run)
                        else:
                            pca_plot(x_1_samples, x_0_samples, x_fake, n_plot=512, save_name=os.path.join(save_dir_name, 'plot_pca_samples.png'), wandb_save_postfix=fw_or_bw, is_wandb=wandb.run)


                        bmgan_sample = lambda x: sample_from_model(pos_coeff, netG_proj,
                                                                        args.num_timesteps, x.to(device),
                                                                        T, args)[0]
                        
                        condBW_Uvp = compute_condBWUVP(bmgan_sample, dim, eps, n_samples=1000, device='cpu')
                        
                        print(f'CondBW-UVP: {condBW_Uvp}')

                        y_samples = y_gt_sampler(10000)
                        x_samples = x_gt_sampler(10000)
                        x_pred = bmgan_sample(y_samples)
                        bw_uvp = compute_BW_UVP_by_gt_samples(x_samples, x_pred.cpu())
                        print(f'BW-UVP f: {bw_uvp}')

                    if wandb.run:
                        wandb.log({f'condBW_Uvp_{fw_or_bw}': condBW_Uvp, f'BW-UVP_{fw_or_bw}': bw_uvp})

                    x_0_samples = x_gt_sampler(args.batch_size).to(device, non_blocking=True)
                    x_1_samples = y_gt_sampler(args.batch_size).to(device, non_blocking=True)
                    
                    x_fake = sample_from_model(pos_coeff, netG_proj, args.num_timesteps, x_1_samples, T, args)[0].detach()
                    
                    if wandb.run:
                        pca_plot(x_1_samples, x_0_samples, x_fake, n_plot=512, save_name='plot_pca_samples.png', wandb_save_postfix=fw_or_bw + '_no_ema', is_wandb=wandb.run)
                    else:
                        pca_plot(x_1_samples, x_0_samples, x_fake, n_plot=512, save_name=os.path.join(save_dir_name, 'plot_pca_samples.png'), wandb_save_postfix=fw_or_bw+ '_no_ema', is_wandb=wandb.run)


                    bmgan_sample = lambda x: sample_from_model(pos_coeff, netG_proj,
                                                                    args.num_timesteps, x.to(device),
                                                                    T, args)[0]
                    
                    condBW_Uvp = compute_condBWUVP(bmgan_sample, dim, eps, n_samples=1000, device='cpu')
                    
                    print(f'CondBW-UVP: {condBW_Uvp}')

                    y_samples = y_gt_sampler(10000)
                    x_samples = x_gt_sampler(10000)
                    x_pred = bmgan_sample(y_samples)
                    bw_uvp = compute_BW_UVP_by_gt_samples(x_samples, x_pred.cpu())
                    print(f'BW-UVP f: {bw_uvp}')

                    if wandb.run:
                        wandb.log({f'condBW_Uvp_{fw_or_bw}_no_ema': condBW_Uvp, f'BW-UVP_{fw_or_bw}_no_ema': bw_uvp})

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
            if wandb.run:
                wandb.log({'loss_G': errG.item(), 'loss_D': errD.item()})
                
        if iteration % 100 == 0:
            print('Markovain proj {} on IMF iter {}: Iter {}, G Loss: {}, D Loss: {}'.format(fw_or_bw,
                                                                                             imf_num_iter,
                                                                                             iteration,
                                                                                             errG.item(),
                                                                                             errD.item()))

# %%
def train(gpu, args):
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # torch.cuda.manual_seed_all(42)
    device = torch.device('cuda:{}'.format(gpu))

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
        'imf_iters': args.imf_iters,
        'save_ckpt': True,
        'save_ckpt_every': 5000,
        'save_content': True,
        'save_content_every': 5000,
        'visualize': True,
        'visualize_every': 1000,
        'print': True,
        'print_every': 100,
        'resume': False,
        'D_opt_steps': args.D_opt_steps,
        'dim': args.dim,
        'epsilon': args.epsilon,
        'eval_freq': args.eval_freq,
        'fw_ckpt': args.fw_ckpt,
        'bw_ckpt': args.bw_ckpt,
        'inner_imf_mark_proj_iters': args.inner_imf_mark_proj_iters,
        'is_wandb': args.is_wandb
    }
    

    dim = args.dim
    eps = args.epsilon
    eval_freq = args.eval_freq
    
    
    args = dotdict(opt)

    
    inner_imf_mark_proj_iters = args.inner_imf_mark_proj_iters
    
    from  datetime import datetime
    datetime_marker_str = datetime.now().strftime("%d:%m:%y_%H:%M:%S")
    
    save_dir_name = os.path.join('sb_bench_results', f'dim_{dim}_eps_{eps}_imf')

    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)
    
    save_dir_name = os.path.join(save_dir_name, f'T_{args.num_timesteps}')
    
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)

    save_dir_name = os.path.join(save_dir_name, datetime_marker_str + '_exp')

    if os.path.exists(save_dir_name):
        save_dir_name += '_1'
    
    os.mkdir(save_dir_name)

    forward_model_ckpt = args.fw_ckpt
    backward_model_ckpt = args.bw_ckpt

    wandb_config = {'dim': dim, 'eps': eps, 'imf': True, 'ema': args.use_ema,
                     'num_iter': args.num_iterations, 'dataset': 'SB_bench',
                      'T': args.num_timesteps, 'reset_weights': args.reset_weights, 
                      'fw_ckpt': forward_model_ckpt, 'bw_ckpt': backward_model_ckpt}

    # logging.basicConfig(filename=os.oath.join(save_dir_name, ''), level=logging.INFO)
    # logger.info('Started')

    # if args.is_wandb > 0:

    wandb.init(project="BM_GAN", name=f"BMGAN_SB", config=wandb_config)
    
    # data_loader_size = len(data_loader)
    # print(f'data_loader size = {data_loader_size}')
    nz = args.nz  # latent dimension

    netG_fw = MyGenerator(
        x_dim=args.dim,
        t_dim=args.t_dim,
        n_t=args.num_timesteps,
        out_dim=args.dim,
        z_dim=nz,
        layers=args.layers_G
    ).to(device)

    netG_bw = MyGenerator(
        x_dim=args.dim,
        t_dim=args.t_dim,
        n_t=args.num_timesteps,
        out_dim=args.dim,
        z_dim=nz,
        layers=args.layers_G
    ).to(device)

    netD_fw = MyDiscriminator(
        x_dim=args.dim,
        t_dim=args.t_dim,
        n_t=args.num_timesteps,
        layers=args.layers_D
    ).to(device)

    netD_bw = MyDiscriminator(
        x_dim=args.dim,
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
    
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = BrownianPosterior_Coefficients(args, device)

    T = get_time_schedule(args, device)

    print(forward_model_ckpt)

    checkpoint_fw = torch.load(forward_model_ckpt)

    global_step_fw = checkpoint_fw['global_step']
    print(f"downloaded forward ckpt {forward_model_ckpt} on iter {global_step_fw}")
    
    # load G
    from copy import deepcopy

    optimizerG_fw.load_state_dict(checkpoint_fw['optimizerG'])

    netG_fw.load_state_dict(checkpoint_fw['netG_dict_ema'])
    
    ema_g_fw = ExponentialMovingAverage(netG_fw.parameters(), decay=args.ema_decay)
    # ema_g_fw.shadow_params = deepcopy(list(netG_fw.parameters()))
    ema_g_fw.to(device)
    
    # netG_fw.load_state_dict(checkpoint_fw['netG_dict'])

    # load D
    netD_fw.load_state_dict(checkpoint_fw['netD_dict'])
    optimizerD_fw.load_state_dict(checkpoint_fw['optimizerD'])

    checkpoint_bw = torch.load(backward_model_ckpt)
    global_step_bw = checkpoint_bw['global_step']
    print(f"downloaded backward ckpt {backward_model_ckpt} on iter {global_step_bw}")
    # load G

    optimizerG_bw.load_state_dict(checkpoint_bw['optimizerG'])

    netG_bw.load_state_dict(checkpoint_bw['netG_dict_ema'])

    ema_g_bw = ExponentialMovingAverage(netG_bw.parameters(), decay=args.ema_decay)
    # ema_g_bw.shadow_params = deepcopy(list(netG_bw.parameters()))
    ema_g_bw.to(device)
    
    # netG_bw.load_state_dict(checkpoint_bw['netG_dict'])
    
    # load D
    netD_bw.load_state_dict(checkpoint_bw['netD_dict'])
    optimizerD_bw.load_state_dict(checkpoint_bw['optimizerD'])

    imf_iters = args.imf_iters
    
    sampler = EOTGMMSampler(dim=dim, eps=eps, batch_size=128, download=False)
    
    # fw x -> y

    x_gt_sampler = sampler.x_sample
    y_gt_sampler = sampler.y_sample

    print(f"Start to run IMF with {imf_iters} iterations")

    bmgan_sample_bw = lambda x: sample_from_model(pos_coeff, netG_bw,
                                                    args.num_timesteps, x.to(device),
                                                    T, args)[0]

    for imf_num_iter in range(imf_iters):
        # ----Forward model learning (y -> x)----

        if args.use_ema:
            condsampler_fw = ModelCondSampler(y_gt_sampler, bmgan_sample_bw, ema_model=ema_g_bw)
        else:
            condsampler_fw = ModelCondSampler(y_gt_sampler, bmgan_sample_bw)

        # reset model weights
        
        if args.reset_weights:
            print('\n\nReset Weigths\n\n')
            netG_fw = MyGenerator(
                x_dim=args.dim,
                t_dim=args.t_dim,
                n_t=args.num_timesteps,
                out_dim=args.dim,
                z_dim=nz,
                layers=args.layers_G
            ).to(device)

            netD_fw = MyDiscriminator(
                x_dim=args.dim,
                t_dim=args.t_dim,
                n_t=args.num_timesteps,
                layers=args.layers_D
            ).to(device)

            optimizerG_fw = optim.Adam(netG_fw.parameters(),
                                    lr=args.lr_g, betas=(args.beta1, args.beta2))
            optimizerD_fw = optim.Adam(netD_fw.parameters(),
                                    lr=args.lr_d, betas=(args.beta1, args.beta2))
            
            ema_g_fw = ExponentialMovingAverage(netG_fw.parameters(), decay=args.ema_decay)
            
        markovian_projection(y_gt_sampler, x_gt_sampler, inner_imf_mark_proj_iters, args, condsampler_fw, netG_fw, netD_fw,
                             optimizerG_fw, optimizerD_fw, ema_g_fw, pos_coeff, device, save_dir_name, imf_num_iter, dim=dim, eps=eps,
                             T=T, D_opt_steps=args.D_opt_steps, fw_or_bw='fw')

        # condBW compute

        bmgan_sample_fw = lambda x: sample_from_model(pos_coeff, netG_fw,
                                                      args.num_timesteps, x.to(device),
                                                      T, args)[0]
        
        with ema_g_fw.average_parameters():
            condBW_Uv_fw = compute_condBWUVP(bmgan_sample_fw, dim, eps, n_samples=1000, device='cpu')

        print(f'CondBW-UVP: {condBW_Uv_fw}')

        from eot_benchmark.metrics import (
            compute_BW_UVP_by_gt_samples
        )

        with ema_g_fw.average_parameters():
            x_1_samples = y_gt_sampler(10000)
            x_0_samples = x_gt_sampler(10000)
            x_1_pred = bmgan_sample_fw(x_0_samples)
            bw_uvp = compute_BW_UVP_by_gt_samples(x_1_pred.cpu(), x_1_samples)
            print(f'BW-UVP f: {bw_uvp}')

        if wandb.run:
            wandb.log({'condBW_Uvp_fw': condBW_Uv_fw, 'BW-UVP_fw': bw_uvp})
        
        # save model
        path_to_save_ckpt = os.path.join(save_dir_name, 
                                                f'content_fw_imf_num_iter_{imf_num_iter}.pth')
        
        with ema_g_fw.average_parameters():
        
            content = {'imf_num_iter': imf_num_iter, 'fw_or_bw': 'fw',
                        'netG_dict': netG_fw.state_dict(),
                        'optimizerG': optimizerG_fw.state_dict(),
                        'netD_dict': netD_fw.state_dict(),
                        'optimizerD': optimizerD_fw.state_dict()}

            torch.save(content, path_to_save_ckpt)

        if args.use_ema:
            condsampler_bw = ModelCondSampler(x_gt_sampler, bmgan_sample_fw, ema_model=ema_g_fw)
        else:
            condsampler_bw = ModelCondSampler(x_gt_sampler, bmgan_sample_fw)
            
        # reset model weights
        if args.reset_wights:
            print('\n\nReset Weigths\n\n')
            netG_bw = MyGenerator(
                x_dim=args.dim,
                t_dim=args.t_dim,
                n_t=args.num_timesteps,
                out_dim=args.dim,
                z_dim=nz,
                layers=args.layers_G
            ).to(device)

            netD_bw = MyDiscriminator(
                x_dim=args.dim,
                t_dim=args.t_dim,
                n_t=args.num_timesteps,
                layers=args.layers_D
            ).to(device)

            optimizerG_bw = optim.Adam(netG_bw.parameters(),
                                    lr=args.lr_g, betas=(args.beta1, args.beta2))
            optimizerD_bw = optim.Adam(netD_bw.parameters(),
                                    lr=args.lr_d, betas=(args.beta1, args.beta2))
            
            ema_g_bw = ExponentialMovingAverage(netG_bw.parameters(), decay=args.ema_decay)

        markovian_projection(x_gt_sampler, y_gt_sampler, inner_imf_mark_proj_iters, args, condsampler_bw, netG_bw, netD_bw,
                             optimizerG_bw, optimizerD_bw, ema_g_bw, pos_coeff, device, save_dir_name, imf_num_iter, dim=dim, eps=eps,
                             T=T, D_opt_steps=args.D_opt_steps, fw_or_bw='bw')
        
            
        bmgan_sample_bw = lambda x: sample_from_model(pos_coeff, netG_bw,
                                                        args.num_timesteps, x.to(device),
                                                        T, args)[0]

        # with ema_g_bw.average_parameters():
        #     condBW_Uv_bw = compute_condBWUVP(bmgan_sample_bw, dim, eps, n_samples=1000, device='cpu')
            
        # if wandb.run:
        #     wandb.log({'condBW_Uvp_bw': condBW_Uv_bw})

        with ema_g_bw.average_parameters():
            x_1_samples = x_gt_sampler(10000)
            x_0_samples = y_gt_sampler(10000)
            x_1_pred = bmgan_sample_bw(x_0_samples)
            bw_uvp = compute_BW_UVP_by_gt_samples(x_1_pred.cpu(), x_1_samples)
            print(f'BW-UVP bw: {bw_uvp}')

        if wandb.run:
            wandb.log({'BW-UVP_bw': bw_uvp})

        path_to_save_ckpt = os.path.join(save_dir_name,
                                                f'content_bw_imf_num_iter_{imf_num_iter}.pth')
        
        with ema_g_bw.average_parameters():

            content = {'imf_num_iter': imf_num_iter, 'fw_or_bw': 'bw',
                        'netG_dict': netG_bw.state_dict(),
                        'optimizerG': optimizerG_bw.state_dict(),
                        'netD_dict': netD_bw.state_dict(),
                        'optimizerD': optimizerD_bw.state_dict()}

            torch.save(content, path_to_save_ckpt)

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--is_wandb', type=int, default=0)
    
    parser.add_argument('--D_opt_steps', type=int, default=1)
    parser.add_argument('--reset_weights', type=int, default=0)

    # generator and training
    
    parser.add_argument('--fw_ckpt', type=str)
    parser.add_argument('--bw_ckpt', type=str)

    parser.add_argument('--inner_imf_mark_proj_iters', type=int, default=20000)
    parser.add_argument('--imf_iters', type=int, default=20)
    parser.add_argument('--eval_freq', type=int, default=1000)

    args = parser.parse_args()

    if args.reset_weights > 0:
        args.reset_weights = True
    else:
        args.reset_weights = False

    train(0, args)

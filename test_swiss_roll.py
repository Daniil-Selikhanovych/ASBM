import os
import glob
import argparse
import math
import numpy as np
from functools import partial

import matplotlib.pyplot as plt

from train_asbm_swissroll import MySampler, my_swiss_roll, dotdict, MyGenerator, sample_posterior
from posteriors import BrownianPosterior_Coefficients, get_time_schedule

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sample_traj(self, x, device):
    x_t = x
    traj = []
    for t in range(1, self.T + 1):
        traj.append(x_t)

        z = torch.randn([x.shape[0], self.latent_dim]).to(device)

        x_1_predict = self.g_net(x_t, z, (t - 1) * torch.ones([x.shape[0], 1]).to(device))

        t_prev, t_next = (t - 1) / self.T, t / self.T

        x_t = x_1_predict * (1 - (1 - t_next) / (1 - t_prev)) + x_t * ((1 - t_next) / (1 - t_prev))
        x_t = x_t + torch.randn_like(x_t) * math.sqrt(self.eps * (t_next - t_prev) * (1 - t_next) / (1 - t_prev))
    traj.append(x_t)
    traj = torch.stack(traj, dim=1)
    return traj


def sample_from_model(coefficients, generator, n_time, x_init, opt):
    x = x_init
    x_traj = [x_init]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
            x_traj.append(x)

    traj = torch.stack(x_traj, dim=0)
    return x, traj


def sample_from_model_bb(coefficients, generator, n_time, x_init, opt, n_inter_points=100):
    x = x_init
    x_traj = [x_init]
    x_traj_prediction = [x_init]
    print(f"n_inter_points = {n_inter_points}")
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_prev, t_next = (i + 1) / n_time, i / n_time

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)

            x_traj_prediction.append(x_new)

            x_t_prev = x
            # print(f"t_next, t_prev = {t_next, t_prev}")

            for t_prime in torch.linspace(t_prev + (t_next - t_prev) * 1 / n_inter_points, t_next, n_inter_points):
                # print(f"t_prime = {t_prime}, t_prev = {t_prev}, t_next = {t_next}, epsilon = {opt.epsilon}")
                # print(f"multiplier = {(t - t_prev) * (t_next - t) / (t_next - t_prev)}")
                x_t_prime = brownian_bridge(x_new, x_t_prev, t_next, t_prev, t_prime, opt.epsilon)
                # print(f"x_t_prime = {x_t_prime}")
                x_traj.append(x_t_prime)
                x_t_prev = x_t_prime
                t_prev = t_prime

            x = x_new.detach()
            # x_traj.append(x)

    x_traj_prediction = torch.stack(x_traj_prediction, dim=0)
    traj = torch.stack(x_traj, dim=0)
    return traj, x_traj_prediction


def brownian_bridge(x_t_prev, x_t_next, t_prev, t_next, t, eps):
    # assert t_next > t_prev
    mean = x_t_prev + (x_t_next - x_t_prev) * (t - t_prev) / (t_next - t_prev)
    multiplier = (t - t_prev) * (t_next - t) / (t_next - t_prev)
    # print(f"multiplier = {multiplier}, t - t_prev = {t - t_prev}, t_next - t - {t_next - t}, t_next - t_prev = {t_next - t_prev}")
    std = torch.sqrt(eps * multiplier)
    return mean + torch.randn_like(x_t_prev) * std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--epsilon', type=float, default=1.0)
    # parser.add_argument('--content_name', type=str, default="content_200000.pth")
    parser.add_argument('--dataset_name', type=str, default="swiss_roll")
    parser.add_argument('--root', type=str, default="dd_gan")

    args = parser.parse_args()

    # content_name = args.content_name
    # num_iterations = 200000
    # content_name = f"content_{num_iterations}.pth"
    dataset_name = args.dataset_name
    exp_path = f"./saved_info/{args.root}/{dataset_name}/{args.exp_name}"

    checkpoint_files = os.path.join(exp_path, "content_fw_imf_num_iter_*_0.pth")
    all_checkpoints = sorted(glob.glob(checkpoint_files))
    checkpoints_basenames = [os.path.basename(checkpoint) for checkpoint in all_checkpoints]
    num_iters = sorted([int(checkpoint_basename.split(".")[0].split("_")[-2])
                        for checkpoint_basename in checkpoints_basenames])

    print(f"will evaluated checkpoints {checkpoints_basenames}")

    l2_cost_vals = []

    fig_l2_cost, ax_l2_cost = plt.subplots(figsize=(12, 12))

    for iter_num in num_iters:
        content_name = f"content_fw_imf_num_iter_{iter_num}_0.pth"
        print(f"evaluate experiment {content_name}")

        device = 'cuda:0'

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
            'sampler_gen_params': {},
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

        nz = opt.nz  # latent dimension

        netG = MyGenerator(
            x_dim=opt.x_dim,
            t_dim=opt.t_dim,
            n_t=opt.num_timesteps,
            out_dim=opt.out_dim,
            z_dim=nz,
            layers=opt.layers_G
        ).to(device)

        checkpoint_file = os.path.join(opt.exp_path, content_name)
        print(f"loading {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        # init_iteration = checkpoint['global_step']
        # global_iteration = init_iteration

        # load G
        netG.load_state_dict(checkpoint['netG_dict'])
        # netG.load_state_dict(checkpoint)

        pos_coeff = BrownianPosterior_Coefficients(opt, device)
        T = get_time_schedule(opt, device)

        sample_func = opt.sample_func

        batch_size = 512
        precalc = None

        sampler = MySampler(
            batch_size=batch_size,
            sample_func=sample_func,
            precalc=precalc,
        )

        x_samples = sampler.sample()
        real_data = torch.from_numpy(x_samples).to(torch.float32).to(device, non_blocking=True)
        # print(f"min real_x = {torch.min(real_data[:, 0])}, max real_x = {torch.max(real_data[:, 0])}")
        # print(f"min rea_y = {torch.min(real_data[:, 1])}, max real_y = {torch.max(real_data[:, 1])}")

        x_t_1 = torch.randn_like(real_data)

        fig, ax = plt.subplots(1, 1, figsize=(4., 4.), dpi=200)

        titles = [None]

        ax.grid(zorder=-20)
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

        n_inter_points = 100

        tr_samples_init = torch.tensor([[0.0, 0.0], [1.75, -1.75], [-1.5, 1.5], [2, 2]])
        tr_samples = tr_samples_init[None].repeat(3, 1, 1).reshape(12, 2)
        tr_samples_init_device = tr_samples.to(torch.float32).to(device, non_blocking=True)
        tr_samples_init_device_randn = torch.randn_like(tr_samples_init_device)
        tr_samples_init_device_randn_numpy = tr_samples_init_device_randn.detach().cpu().numpy()

        # Sampling
        # y_pred = model.sample(x_samples.to(device)).cpu()

        # your sampling function

        # y_pred = None
        y_pred, _ = sample_from_model(pos_coeff, netG, opt.num_timesteps, x_t_1, opt)
        y_pred = y_pred.detach().cpu().numpy()

        # y_pred_given_starts, trajectory = sample_from_model(pos_coeff, netG, opt.num_timesteps, tr_samples_init_device,
        #                                                     T, opt)
        # trajectory = trajectory.detach().cpu().numpy()

        n_inter_points = 100
        trajectory, x_traj_prediction = sample_from_model_bb(pos_coeff, netG, opt.num_timesteps, tr_samples_init_device,
                                          opt, n_inter_points)
        trajectory = trajectory.detach().cpu().numpy()
        x_traj_prediction = x_traj_prediction.detach().cpu().numpy()
        # print(f"trajectory.shape = {trajectory.shape}")

        # print(f"min y_pred_x = {np.min(y_pred[:, 0])}, max y_pred_x = {np.max(y_pred[:, 0])}")
        # print(f"min y_pred_y = {np.min(y_pred[:, 1])}, max y_pred_y = {np.max(y_pred[:, 1])}")

        ax.scatter(y_pred[:, 0], y_pred[:, 1],
                   c="salmon", s=64, edgecolors="black", label="Fitted distribution", zorder=1)

        num_points_for_cost = 1000

        np.random.seed(42)
        random_input = np.random.randn(num_points_for_cost, 2)
        x_t_1_for_cost = torch.from_numpy(random_input).to(torch.float32).to(device, non_blocking=True)
        # new_seed = torch.randint(low=1, high=10000, size=(1,))[0]
        # print(f"new seed = {new_seed}")
        y_pred_for_cost, _ = sample_from_model(pos_coeff, netG, opt.num_timesteps, x_t_1_for_cost, opt)

        size = num_points_for_cost
        cost = (F.mse_loss(x_t_1_for_cost, y_pred_for_cost) * x_t_1_for_cost.shape[0]).item()
        cost = cost / size

        l2_cost_vals.append(cost)
        print(f"L2 cost = {cost}")

        # ax.scatter(y_pred_randn[:, 0], y_pred_randn[:, 1],
        #            c="pink", s=64, edgecolors="black", label="Random endings", zorder=1)

        # ax.scatter(tr_samples_init_device_randn_numpy[:, 0], tr_samples_init_device_randn_numpy[:, 1],
        #            c="blue", s=64, edgecolors="black", label="Random starts", zorder=1)

        #ax.scatter(y_pred_given_starts[:, 0], y_pred_given_starts[:, 1],
        #           c="black", s=64, edgecolors="black", label="Endings from given starts", zorder=1)

        # trajectory = sample_traj(model, tr_samples.to(device)).detach().cpu()
        # trajectory = sample_traj_bb(netG, tr_samples.to(device), opt.epsilon, opt.num_timesteps, device).detach().cpu()

        ax.scatter(tr_samples[:, 0], tr_samples[:, 1],
                   c="lime", s=128, edgecolors="black", label=r"Trajectory start ($x \sim p_0$)", zorder=3)

        # ax.scatter(trajectory[:, -1, 0], trajectory[:, -1, 1],
        #            c="yellow", s=64, edgecolors="black", label=r"Trajectory end (fitted)", zorder=3)
        ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1],
                   c="yellow", s=64, edgecolors="black", label=r"Trajectory end (fitted)", zorder=3)

        # num_samples = 12
        # num_samples = opt.num_timesteps + 1
        num_samples = trajectory.shape[1]
        for i in range(num_samples):
            # ax.scatter(trajectory[i, ::1, 0], trajectory[i, ::1, 1], "black", markeredgecolor="black", )
            #         linewidth=1.5, zorder=2)
            # ax.scatter(trajectory[i, ::1, 0], trajectory[i, ::1, 1], color="black")
            if i == 0:
                ax.plot(trajectory[::1, i, 0], trajectory[::1, i, 1], "black", markeredgecolor="black",
                        linewidth=1.5, zorder=2, label=r"Trajectory of $S_{\theta}$")
            else:
                ax.plot(trajectory[::1, i, 0], trajectory[::1, i, 1], "black", markeredgecolor="black",
                        linewidth=1.5, zorder=2)
            # if i == 0:
            #     ax.plot(trajectory[i, ::1, 0], trajectory[i, ::1, 1], "grey", markeredgecolor="black",
            #             linewidth=0.5, zorder=2, label=r"Trajectory of $S_{\theta}$")
            # else:
            #     ax.plot(trajectory[i, ::1, 0], trajectory[i, ::1, 1], "grey", markeredgecolor="black",
            #             linewidth=0.5, zorder=2)

        # print(f"x_traj_prediction.shape = {x_traj_prediction.shape}")
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

        fig_name = f"trajectories_{content_name}.png"
        fig_path = os.path.join(opt.exp_path, fig_name)
        print(f"saving {fig_path}")
        plt.savefig(fig_path)

    l2_cost_vals = np.array(l2_cost_vals)

    ax_l2_cost.plot(num_iters, l2_cost_vals)
    ax_l2_cost.set_title(f'L2 cost, epsilon = {args.epsilon}')
    ax_l2_cost.set_xlabel("IMF iteration")
    ax_l2_cost.set_ylabel("Cost")
    ax_l2_cost.legend()
    ax_l2_cost.grid(True)

    path_to_save_figures = os.path.join(exp_path,
                                        f"test_l2_cost_imf_evolution_epsilon_{args.epsilon}.png")
    fig_l2_cost.savefig(path_to_save_figures)
    print(f"saving fid graph in {path_to_save_figures}")

import numpy as np
import torch
import torch.nn as nn
from eot_benchmark.gaussian_mixture_benchmark import (
    get_guassian_mixture_benchmark_sampler,
    get_guassian_mixture_benchmark_ground_truth_sampler,
)


class GroundTruthSDE(nn.Module):

    def __init__(self, potential_params, eps, n_steps, is_diagonal=False):
        super().__init__()
        probs, mus, sigmas = potential_params
        self.eps = eps
        self.n_steps = n_steps
        self.is_diagonal = is_diagonal

        self.register_buffer("potential_probs", probs)
        self.register_buffer("potential_mus", mus)
        self.register_buffer("potential_sigmas", sigmas)

    def forward(self, x):
        t_storage = [torch.zeros([1])]
        trajectory = [x.cpu()]
        for i in range(self.n_steps):
            delta_t = 1 / self.n_steps
            t = torch.tensor([i / self.n_steps])
            drift = self.get_drift(x, t)

            rand = np.sqrt(self.eps) * np.sqrt(delta_t) * torch.randn(*x.shape).to(x.device)

            x = (x + drift * delta_t + rand).detach()

            trajectory.append(x.cpu())
            t_storage.append(t)

        return torch.stack(trajectory, dim=0).transpose(0, 1), torch.stack(t_storage, dim=0).unsqueeze(1).repeat(
            [1, x.shape[0], 1])

    def sample(self, x):
        return self.forward(x)


# class EOTGMMSampler(Sampler):
class EOTGMMSampler:
    def __init__(self, dim, eps, batch_size=64, download=False) -> None:
        super().__init__()
        eps = eps if int(eps) < 1 else int(eps)

        self.dim = dim
        self.eps = eps
        self.x_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="input", dim=dim, eps=eps,
                                                                batch_size=batch_size, device=f"cpu", download=download)
        self.y_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="target", dim=dim, eps=eps,
                                                                batch_size=batch_size, device=f"cpu", download=download)
        self.gt_sampler = get_guassian_mixture_benchmark_ground_truth_sampler(dim=dim, eps=eps,
                                                                              batch_size=batch_size, device=f"cpu",
                                                                              download=download)

    def x_sample(self, batch_size):
        return self.x_sampler.sample(batch_size)

    def y_sample(self, batch_size):
        return self.y_sampler.sample(batch_size)

    def gt_sample(self, batch_size):
        return self.gt_sampler.sample(batch_size)

    def conditional_y_sample(self, x):
        return self.gt_sampler.conditional_plan.sample(x)

    def gt_sde_path_sampler(self):
        mus = self.y_sampler.conditional_plan.potential_mus
        probs = self.y_sampler.conditional_plan.potential_probs
        sigmas = self.y_sampler.conditional_plan.potential_sigmas
        potential_params = (probs, mus, sigmas)

        n_em_steps = 99
        gt_sde = GroundTruthSDE(potential_params, self.eps, n_em_steps)

        def return_fn(batch_size):
            x_samples = self.x_sample(batch_size)
            x_t, t = gt_sde.sample(x_samples)
            t = t.transpose(0, 1)
            return x_t, t

        return return_fn

    def brownian_bridge_sampler(self):
        def return_fn(batch_size):
            x_0 = self.x_sample(batch_size)

            x_1 = self.gt_sampler.conditional_plan.sample(x_0)
            t_0 = 0
            t_1 = 1
            n_timesteps = 100

            t = torch.arange(n_timesteps).reshape([-1, 1]).repeat([1, batch_size]).transpose(0, 1) / n_timesteps

            x_0 = x_0.unsqueeze(1).repeat([1, n_timesteps, 1])
            x_1 = x_1.unsqueeze(1).repeat([1, n_timesteps, 1])

            # N x T x D

            mean = x_0 + ((t - t_0) / (t_1 - t_0)).reshape([x_0.shape[0], x_0.shape[1], 1]) * (x_1 - x_0)

            std = torch.sqrt(self.eps * (t - t_0) * (t_1 - t) / (t_1 - t_0))

            x_t = mean + std.reshape([std.shape[0], std.shape[1], 1]) * torch.randn_like(mean)

            return x_t, t

        return return_fn

    def path_sampler(self):
        mus = self.y_sampler.conditional_plan.potential_mus
        probs = self.y_sampler.conditional_plan.potential_probs
        sigmas = self.y_sampler.conditional_plan.potential_sigmas
        potential_params = (probs, mus, sigmas)

        n_em_steps = 99
        gt_sde = GroundTruthSDE(potential_params, self.eps, n_em_steps)

        def return_fn(batch_size):
            x_samples = self.x_sample(batch_size)
            x_t, t = gt_sde.sample(x_samples)
            t = t.transpose(0, 1)
            return x_t, t

        return return_fn

    def __str__(self) -> str:
        return f'EOTSampler_D_{self.dim}_eps_{self.eps}'

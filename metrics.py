import os
from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d

from sampling_utils import sample_from_model, sample_from_generator


def get_activations_paired_dataloader(paired_dataloader, pos_coeff, netG, args, model_fid, batch_size=50,
                                      dims=2048, device='cpu', mode="features_gt", cost_type='mse'):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model_fid.eval()

    assert mode in ["features_gt", "features_pred"]

    len_dataset = len(paired_dataloader.dataset)

    if batch_size > len_dataset:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        paired_dataloader.batch_size = len_dataset

    pred_arr = np.empty((len_dataset, dims))
    to_range_0_1 = lambda x: (x + 1.) / 2.

    print(f"Compute activations for mode {mode}")

    cost = 0
    size = len(paired_dataloader.dataset)

    start_idx = 0

    for (x, y) in tqdm(paired_dataloader):
        # batch = batch.to(device)
        x_0 = x.to(device)
        x_t_1 = y.to(device)

        with torch.no_grad():
            if mode == "features_pred":
                fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, args)
                fake_sample_range = to_range_0_1(fake_sample)
                # print(f"min fake sample = {fake_sample_range.min()}, max fake sample = {fake_sample_range.max()}")
                pred = model_fid(fake_sample_range)[0]

                if cost_type == 'mse':
                    cost += (F.mse_loss(x_t_1, fake_sample) * x_t_1.shape[0]).item()
                elif cost_type == 'l1':
                    cost += (F.l1_loss(x_t_1, fake_sample) * x_t_1.shape[0]).item()
                else:
                    raise Exception('Unknown COST')
            elif mode == "features_gt":
                x_0_range = to_range_0_1(x_0)
                # print(f"min gt sample = {x_0_range.min()}, max fake sample = {x_0_range.max()}")
                pred = model_fid(x_0_range)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    cost = cost / size
    torch.cuda.empty_cache()

    return pred_arr, cost


def get_activations_paired_dataloader_all_generator_steps(paired_dataloader, pos_coeff, netG, args, model_fid, batch_size=50,
                                      dims=2048, device='cpu', cost_type='mse'):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model_fid.eval()


    len_dataset = len(paired_dataloader.dataset)

    if batch_size > len_dataset:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        paired_dataloader.batch_size = len_dataset

    pred_arr = np.empty((args.num_timesteps, len_dataset, dims))
    to_range_0_1 = lambda x: (x + 1.) / 2.

    print(f"Compute activations for generator trajectory mode")

    cost_all_steps = [0 for _ in range(args.num_timesteps)]
    size = len(paired_dataloader.dataset)

    start_idx = 0

    for (_, y) in tqdm(paired_dataloader):
        # batch = batch.to(device)\
        x_t_1 = y.to(device)

        with torch.no_grad():
            fake_sample_all_steps = sample_from_generator(pos_coeff, netG, args.num_timesteps, x_t_1, args, append_first=False)
            fake_sample_range_all_steps = to_range_0_1(fake_sample_all_steps)
            # print(f"min fake sample = {fake_sample_range.min()}, max fake sample = {fake_sample_range.max()}")
            for i in range(args.num_timesteps):
                pred = model_fid(fake_sample_range_all_steps[i])[0]

                if cost_type == 'mse':
                    cost_all_steps[i] += (F.mse_loss(x_t_1, fake_sample_all_steps[i]) * x_t_1.shape[0]).item()
                elif cost_type == 'l1':
                    cost_all_steps[i] += (F.l1_loss(x_t_1, fake_sample_all_steps[i]) * x_t_1.shape[0]).item()
                else:
                    raise Exception('Unknown COST')

                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

                pred = pred.squeeze(3).squeeze(2).cpu().numpy()

                pred_arr[i, start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    cost_all_steps = np.array(cost_all_steps)
    cost_all_steps = cost_all_steps / size
    torch.cuda.empty_cache()

    return pred_arr, cost_all_steps


def calculate_activation_statistics_paired_dataloader_all_generator_steps(paired_dataloader, model_fid, batch_size=50, dims=2048,
                                                      device='cpu', netG=None, args=None,
                                                      pos_coeff=None):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- resize      : resize image to this shape

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    # act = get_activations(files, model, batch_size, dims, device, resize)
    act_all_steps, cost_all_steps = get_activations_paired_dataloader_all_generator_steps(paired_dataloader, pos_coeff, netG, 
                                                                      args, model_fid, batch_size,
                                                                      dims, device, cost_type='mse')
    num_steps = args.num_timesteps
    mu_all_steps = []
    sigma_all_steps = []
    for i in range(num_steps):
        mu = np.mean(act_all_steps[i], axis=0)
        sigma = np.cov(act_all_steps[i], rowvar=False)
        mu_all_steps.append(mu)
        sigma_all_steps.append(sigma)
    return mu_all_steps, sigma_all_steps, cost_all_steps


def calculate_activation_statistics_paired_dataloader(paired_dataloader, model_fid, batch_size=50, dims=2048,
                                                      device='cpu', mode="features_gt", netG=None, args=None,
                                                      pos_coeff=None):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- resize      : resize image to this shape

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    # act = get_activations(files, model, batch_size, dims, device, resize)
    act, cost = get_activations_paired_dataloader(paired_dataloader,
                                                  pos_coeff, netG, args, model_fid, batch_size, dims, device,
                                                  mode, cost_type='mse')
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, cost


def compute_statistics_of_path_or_dataloader(path, paired_dataloader, model_fid, batch_size, dims, device,
                                             mode, netG, args,
                                             pos_coeff):
    if (path.endswith('.npz') or path.endswith('.npy')) and os.path.exists(path):
        f = np.load(path, allow_pickle=True)
        try:
            m, s = f['mu'][:], f['sigma'][:]
        except:
            m, s = f.item()['mu'][:], f.item()['sigma'][:]
        print(f"read stats from {path}")
        cost = 0
    else:
        # print(f"Compute stats for paired dataloader in the mode {mode}")
        m, s, cost = calculate_activation_statistics_paired_dataloader(paired_dataloader, model_fid, batch_size, dims,
                                                                       device, mode, netG, args,
                                                                       pos_coeff)
    return m, s, cost


def calculate_cost(pos_coeff, netG, args, T, loader, cost_type='mse', verbose=False):
    size = len(loader.dataset)

    cost = 0
    for step, (_, y) in tqdm(enumerate(loader)) if verbose else enumerate(loader):
        x_t_1 = y.cuda()

        with torch.no_grad():
            fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, args)

        if cost_type == 'mse':
            cost += (F.mse_loss(x_t_1, fake_sample) * x_t_1.shape[0]).item()
        elif cost_type == 'l1':
            cost += (F.l1_loss(x_t_1, fake_sample) * x_t_1.shape[0]).item()
        else:
            raise Exception('Unknown COST')

    cost = cost / size
    torch.cuda.empty_cache()
    return cost

import os
import glob
import numpy as np
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from multiprocessing import cpu_count

class CelebaDataset(Dataset):
    def __init__(self, size, root, root_with_results, path_to_celeba_attr):
        self.root = root
        self.size = size
        self.root_with_results = root_with_results
        self.path_to_celeba_attr = path_to_celeba_attr

        self.transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        self.part_test = 0.1

        self.mode = "male_to_female" # opt.mode
        # assert self.mode in ["male_to_female", "female_to_male"]

        self.path_to_celeba_attr = path_to_celeba_attr
        with open(self.path_to_celeba_attr, 'r') as f:
            lines = f.readlines()[2:]

        self.female_imgs = [lines[i].replace('  ', ' ').split(' ')[0] for i in list(range(len(lines)))
                            if lines[i].replace('  ', ' ').split(' ')[21] == '-1']
        self.male_imgs = [lines[i].replace('  ', ' ').split(' ')[0] for i in list(range(len(lines)))
                          if lines[i].replace('  ', ' ').split(' ')[21] != '-1']

        self.female_paths = sorted([os.path.join(self.root, img) for img in self.female_imgs])
        self.male_paths = sorted([os.path.join(self.root, img) for img in self.male_imgs])

        self.all_images = sorted(glob.glob(os.path.join(self.root, "*.jpg")))
        self.num_all_paths = len(self.all_images)
        self.num_female_paths = len(self.female_imgs)
        self.num_male_paths = len(self.male_imgs)
        assert self.num_male_paths + self.num_female_paths == self.num_all_paths

        print(f"num celeba male paths = {self.num_male_paths}, num celeba female = {self.num_female_paths}, num all images = {self.num_all_paths}")

        self.num_test_male_paths = int(self.part_test * self.num_male_paths)
        self.num_test_female_paths = int(self.part_test * self.num_female_paths)

        self.num_train_male_paths = self.num_male_paths - self.num_test_male_paths
        self.num_train_female_paths = self.num_female_paths - self.num_test_female_paths

        self.train_male_paths = self.male_paths[:self.num_train_male_paths]
        self.train_female_paths = self.female_paths[:self.num_train_female_paths]

        self.test_male_paths = self.male_paths[self.num_train_male_paths:]
        self.test_female_paths = self.female_paths[self.num_train_female_paths:]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        male_path = self.test_male_paths[index]
        basename = os.path.basename(male_path)
        path_to_fake_image = os.path.join(self.root_with_results, basename)

        A_img = Image.open(male_path).convert('RGB')
        B_img = Image.open(path_to_fake_image).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)

        return A, B

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.num_test_male_paths


def get_activations_paired_dataloader(dataloader, model_fid, batch_size=50,
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

    len_dataset = len(dataloader.dataset)

    if batch_size > len_dataset:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        dataloader.batch_size = len_dataset

    pred_arr = np.empty((len_dataset, dims))
    to_range_0_1 = lambda x: ((x + 1.) / 2.).clamp(0.0, 1.0)

    # print(f"Compute activations for mode {mode}")

    cost = 0
    size = len(dataloader.dataset)

    start_idx = 0

    for (x, y) in tqdm(dataloader):
        # batch = batch.to(device)
        fake_sample = y.to(device)
        input_sample = x.to(device)

        with torch.no_grad():
            # if mode == "features_pred":
            fake_sample_range = to_range_0_1(fake_sample)
            # print(f"min fake sample = {fake_sample_range.min()}, max fake sample = {fake_sample_range.max()}")
            pred = model_fid(fake_sample_range)[0]

            if cost_type == 'mse':
                cost += (F.mse_loss(input_sample, fake_sample) * input_sample.shape[0]).item()
            elif cost_type == 'l1':
                cost += (F.l1_loss(input_sample, fake_sample) * input_sample.shape[0]).item()
            else:
                raise Exception('Unknown COST')
            # elif mode == "features_gt":
            #     x_0_range = to_range_0_1(x_0)
            #     # print(f"min gt sample = {x_0_range.min()}, max fake sample = {x_0_range.max()}")
            #     pred = model_fid(x_0_range)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    cost = cost / size
    torch.cuda.empty_cache()

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma, cost

path_to_results = "./results"
exp_name = "celeba_male_to_female_128"

celeba_root = "/trinity/home/daniil.selikhanovych/data/img_align_celeba_all/img_align_celeba"
path_to_celeba_attr = "/trinity/home/daniil.selikhanovych/NeuralOptimalTransport/datasets/list_attr_celeba.txt"
size = 128

path_to_statistics = "/trinity/home/daniil.selikhanovych/ASBM_reproduce/ASBM/fid_results_test_imf/celeba_celeba_female_to_male/ddgan_unpaired_celeba_ema_bigger_nets_T_4_imf_eps_1_celeba_128_ema_sampling/gt_statistics_forward.npz"

f = np.load(path_to_statistics, allow_pickle=True)
try:
    m1_test_gt_for, s1_test_gt_for = f['mu'][:], f['sigma'][:]
except:
    m1_test_gt_for, s1_test_gt_for = f.item()['mu'][:], f.item()['sigma'][:]
print(f"read stats from {path_to_statistics}")

device = 'cuda'
dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model_fid = InceptionV3([block_idx]).to(device)

bs = 256
list_of_nfe = [1, 2, 3, 4, 5]
iterations = [50000]
path_to_save_metrics = "./fid_metrics"
os.makedirs(path_to_save_metrics, exist_ok=True)

for iteration in iterations:
    path_to_images = os.path.join(path_to_results, exp_name, str(iteration), "images")
    path_to_fid_iteration = os.path.join(path_to_save_metrics, str(iteration))
    os.makedirs(path_to_fid_iteration, exist_ok=True)
    for nfe in list_of_nfe:
        path_to_fid_iteration_nfe = os.path.join(path_to_fid_iteration, str(nfe))
        os.makedirs(path_to_fid_iteration_nfe, exist_ok=True)
        final_path_to_images = os.path.join(path_to_images, f"fake_{nfe}")
        celeba_dataset_fake = CelebaDataset(size, celeba_root, final_path_to_images, path_to_celeba_attr)
        test_dataloader_input_fw = torch.utils.data.DataLoader(celeba_dataset_fake,
                                                               batch_size=bs,
                                                               shuffle=False,
                                                               drop_last=False,
                                                               num_workers=cpu_count())
        mu_fake, sigma_fake, cost = get_activations_paired_dataloader(test_dataloader_input_fw, model_fid, batch_size=50,
                                                                      dims=dims, device=device)
        fid_value_test = calculate_frechet_distance(mu_fake, sigma_fake, m1_test_gt_for, s1_test_gt_for)
        print(f"iteration = {iteration}, nfe = {nfe}, FID = {fid_value_test}, cost = {cost}")

        path_to_save_fid_txt = os.path.join(path_to_fid_iteration_nfe, "fid.txt")
        path_to_save_cost_txt = os.path.join(path_to_fid_iteration_nfe, "cost.txt")

        with open(path_to_save_fid_txt, "w") as f:
            f.write(f"FID = {fid_value_test}")

        with open(path_to_save_cost_txt, "w") as f:
            f.write(f"cost = {cost}")


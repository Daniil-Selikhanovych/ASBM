import numpy as np
import torch
import torchvision
from torchvision import datasets
from torch.utils.data import TensorDataset


def get_random_colored_images(images, seed=0x000000):
    np.random.seed(seed)

    images = 0.5 * (images + 1)
    size = images.shape[0]
    colored_images = []
    hues = 360 * np.random.rand(size)

    for V, H in zip(images, hues):
        V_min = 0

        a = (V - V_min) * (H % 60) / 60
        V_inc = a
        V_dec = V - a

        colored_image = torch.zeros((3, V.shape[1], V.shape[2]))
        H_i = round(H / 60) % 6

        if H_i == 0:
            colored_image[0] = V
            colored_image[1] = V_inc
            colored_image[2] = V_min
        elif H_i == 1:
            colored_image[0] = V_dec
            colored_image[1] = V
            colored_image[2] = V_min
        elif H_i == 2:
            colored_image[0] = V_min
            colored_image[1] = V
            colored_image[2] = V_inc
        elif H_i == 3:
            colored_image[0] = V_min
            colored_image[1] = V_dec
            colored_image[2] = V
        elif H_i == 4:
            colored_image[0] = V_inc
            colored_image[1] = V_min
            colored_image[2] = V
        elif H_i == 5:
            colored_image[0] = V
            colored_image[1] = V_min
            colored_image[2] = V_dec

        colored_images.append(colored_image)

    colored_images = torch.stack(colored_images, dim=0)
    colored_images = 2 * colored_images - 1

    return colored_images


def load_paired_colored_mnist():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: 2 * x - 1)
    ])

    train_set = datasets.MNIST("./", train=True, transform=transform, download=True)
    test_set = datasets.MNIST("./", train=False, transform=transform, download=True)

    digits_2_train = torch.stack(
        [train_set[i][0] for i in range(len(train_set.targets)) if train_set.targets[i] == 2],
        dim=0
    )
    digits_2_train = digits_2_train.reshape(-1, 1, 32, 32)  # (N, 1, 32, 32)
    digits_2_colored_train = get_random_colored_images(digits_2_train)

    digits_3_train = torch.stack(
        [train_set[i][0] for i in range(len(train_set.targets)) if train_set.targets[i] == 3],
        dim=0
    )
    digits_3_train = digits_3_train.reshape(-1, 1, 32, 32)
    digits_3_colored_train = get_random_colored_images(digits_3_train)

    train_size = min(digits_2_colored_train.shape[0], digits_3_colored_train.shape[0])

    print(f"train size = {train_size}")
    print(
        f"digits_2_colored_train.shape = {digits_2_colored_train.shape}, digits_3_colored_train.shape = {digits_3_colored_train.shape}")
    train_dataset = TensorDataset(digits_2_colored_train[:train_size], digits_3_colored_train[:train_size])
    # train_dataset in [-1, 1]

    digits_2_test = torch.stack(
        [test_set[i][0] for i in range(len(test_set.targets)) if test_set.targets[i] == 2],
        dim=0
    )

    digits_2_test = digits_2_test.reshape(-1, 1, 32, 32)
    digits_2_colored_test = get_random_colored_images(digits_2_test)
    digits_3_test = torch.stack(
        [test_set[i][0] for i in range(len(test_set.targets)) if test_set.targets[i] == 3],
        dim=0
    )
    digits_3_test = digits_3_test.reshape(-1, 1, 32, 32)
    digits_3_colored_test = get_random_colored_images(digits_3_test)
    test_size = min(digits_2_colored_test.shape[0], digits_3_colored_test.shape[0])
    print(f"test size = {test_size}")
    print(
        f"digits_2_colored_test.shape = {digits_2_colored_test.shape}, digits_3_colored_test.shape = {digits_3_colored_test.shape}")
    test_dataset = TensorDataset(digits_2_colored_test[:test_size], digits_3_colored_test[:test_size])

    return train_dataset, test_dataset

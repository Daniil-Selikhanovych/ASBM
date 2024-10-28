import os
import numpy as np
import glob
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


class Celeba(data.Dataset):
    def __init__(self, root, train=True, transform=None, part_test=0.1, mode="male_to_female", input_is_second=True):
        self.root = root
        self.train = train
        self.transform = transform
        self.part_test = part_test

        self.mode = mode
        assert self.mode in ["male_to_female", "female_to_male"]

        self.path_to_celeba_attr = os.path.join(root, "..", "list_attr_celeba.txt")
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

        self.num_test_male_paths = int(part_test * self.num_male_paths)
        self.num_test_female_paths = int(part_test * self.num_female_paths)

        self.num_train_male_paths = self.num_male_paths - self.num_test_male_paths
        self.num_train_female_paths = self.num_female_paths - self.num_test_female_paths

        self.train_male_paths = self.male_paths[:self.num_train_male_paths]
        self.train_female_paths = self.female_paths[:self.num_train_female_paths]

        self.test_male_paths = self.male_paths[self.num_train_male_paths:]
        self.test_female_paths = self.female_paths[self.num_train_female_paths:]

        self.generator_for_flip = torch.distributions.Bernoulli(torch.tensor([0.5]))
        self.input_is_second = input_is_second

    def __getitem__(self, index):
        if self.mode == "male_to_female":
            if self.train:
                path_to_male = self.train_male_paths[index]
                path_to_female = np.random.choice(self.train_female_paths, 1)[0]
            else:
                path_to_male = self.test_male_paths[index]
                path_to_female = np.random.choice(self.test_female_paths, 1)[0]

            male_img = Image.open(path_to_male)
            male_img = male_img.convert('RGB')
            female_img = Image.open(path_to_female)
            female_img = female_img.convert('RGB')
            if self.transform is not None:
                male_img = self.transform(male_img)
                female_img = self.transform(female_img)

            if self.train:
                is_flip = self.generator_for_flip.sample()
                if is_flip > 0.5:
                    male_img = transforms.functional.hflip(male_img)
                    female_img = transforms.functional.hflip(female_img)

            if self.input_is_second:
                return female_img, male_img  # due to the code in DDGAN, x = x0, y = x_t_1
            else:
                return male_img, female_img

        elif self.mode == "female_to_male":
            if self.train:
                path_to_female = self.train_female_paths[index]
                path_to_male = np.random.choice(self.train_male_paths, 1)[0]
            else:
                path_to_female = self.test_female_paths[index]
                path_to_male = np.random.choice(self.test_male_paths, 1)[0]

            male_img = Image.open(path_to_male)
            male_img = male_img.convert('RGB')
            female_img = Image.open(path_to_female)
            female_img = female_img.convert('RGB')
            if self.transform is not None:
                male_img = self.transform(male_img)
                female_img = self.transform(female_img)

            if self.train:
                is_flip = self.generator_for_flip.sample()
                if is_flip > 0.5:
                    male_img = transforms.functional.hflip(male_img)
                    female_img = transforms.functional.hflip(female_img)

            if self.input_is_second:
                return male_img, female_img  # due to the code in DDGAN, x = x0, y = x_t_1
            else:
                return female_img, male_img

    def __len__(self):
        if self.train:
            if self.mode == "female_to_male":
                return self.num_train_female_paths
            elif self.mode == "male_to_female":
                return self.num_train_male_paths
        else:
            if self.mode == "female_to_male":
                return self.num_test_female_paths
            elif self.mode == "male_to_female":
                return self.num_test_male_paths

import os
import numpy as np
import glob
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


class Faces2Comics(data.Dataset):
    def __init__(self, root, train=True, transform=None, part_test=0.1, paired=True):
        self.root = root
        self.train = train
        self.transform = transform
        self.paired = paired
        self.part_test = part_test

        if not self.train:
            assert self.paired

        self.path_to_comics = os.path.join(root, "comics")
        self.path_to_faces = os.path.join(root, "faces")

        self.comics_paths = sorted(glob.glob(os.path.join(self.path_to_comics, "*.jpg")))
        self.faces_paths = sorted(glob.glob(os.path.join(self.path_to_faces, "*.jpg")))
        self.num_all_paths = len(self.comics_paths)
        print(f"num comics paths = {len(self.comics_paths)}, num faces images = {len(self.faces_paths)}")
        assert len(self.comics_paths) == len(self.faces_paths)

        self.num_test_paths = int(part_test * self.num_all_paths)
        self.num_train_paths = self.num_all_paths - self.num_test_paths

        self.train_comics_paths = self.comics_paths[:self.num_train_paths]
        self.train_faces_paths = self.faces_paths[:self.num_train_paths]

        self.test_comics_paths = self.comics_paths[self.num_train_paths:]
        self.test_faces_paths = self.faces_paths[self.num_train_paths:]

        self.generator_for_flip = torch.distributions.Bernoulli(torch.tensor([0.5]))

    def __getitem__(self, index):
        if self.train:
            path_to_faces = self.train_faces_paths[index]
        else:
            path_to_faces = self.test_faces_paths[index]

        if self.paired:
            path_to_faces_basename = os.path.basename(path_to_faces)
            path_to_comics = os.path.join(self.path_to_comics, path_to_faces_basename)
        else:
            path_to_comics = np.random.choice(self.train_comics_paths, 1)[0]

        face_img = Image.open(path_to_faces)
        face_img = face_img.convert('RGB')
        comic_img = Image.open(path_to_comics)
        comic_img = comic_img.convert('RGB')
        if self.transform is not None:
            face_img = self.transform(face_img)
            comic_img = self.transform(comic_img)

        if self.train:
            is_flip = self.generator_for_flip.sample()
            if is_flip > 0.5:
                face_img = transforms.functional.hflip(face_img)
                comic_img = transforms.functional.hflip(comic_img)

        return comic_img, face_img  # due to the code in DDGAN, x = x0, y = x_t_1

    def __len__(self):
        if self.train:
            return self.num_train_paths
        else:
            return self.num_test_paths

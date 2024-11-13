import os
import random
import warnings
from os.path import isfile, join

import pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from typing import List, Optional

warnings.filterwarnings("ignore")


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, img):
        length = np.random.randint(1, self.max_length + 1)
        h = img.size(2)
        w = img.size(3)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


class NCaltech101(Dataset):
    def __init__(
        self,
        data_path="data/n-caltech/frames_number_10_split_by_number",
        data_type="train",
        transform=False,
    ):

        self.filepath = os.path.join(data_path)
        self.clslist = os.listdir(self.filepath)
        self.clslist.sort()

        self.dvs_filelist = []
        self.targets = []
        self.resize = transforms.Resize(
            size=(48, 48),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )

        for i, cls in enumerate(self.clslist):
            # print (i, cls)
            file_list = os.listdir(os.path.join(self.filepath, cls))
            num_file = len(file_list)

            cut_idx = int(num_file * 0.9)
            train_file_list = file_list[:cut_idx]
            test_split_list = file_list[cut_idx:]
            for file in file_list:
                if data_type == "train":
                    if file in train_file_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)
                else:
                    if file in test_split_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)

        self.data_num = len(self.dvs_filelist)
        self.data_type = data_type
        if data_type != "train":
            counts = np.unique(np.array(self.targets), return_counts=True)[1]
            class_weights = counts.sum() / (counts * len(counts))
            self.class_weights = torch.Tensor(class_weights)
        self.classes = range(101)
        self.transform = transform
        self.rotate = transforms.RandomRotation(degrees=15)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-15, 15))

    def __getitem__(self, index):
        file_pth = self.dvs_filelist[index]
        label = self.targets[index]
        data = torch.from_numpy(np.load(file_pth)["frames"]).float()
        data = self.resize(data)

        if self.transform:

            choices = ["roll", "rotate", "shear"]
            aug = np.random.choice(choices)
            if aug == "roll":
                off1 = random.randint(-3, 3)
                off2 = random.randint(-3, 3)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            if aug == "rotate":
                data = self.rotate(data)
            if aug == "shear":
                data = self.shearx(data)

        return data, label

    def __len__(self):
        return self.data_num


def build_ncaltech(transform=False):
    train_dataset = NCaltech101(transform=transform)
    val_dataset = NCaltech101(data_type="test", transform=False)

    return train_dataset, val_dataset


class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, indices: Optional[List[int]] = None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(
            size=(48, 48),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.rotate = transforms.RandomRotation(degrees=30)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-30, 30))
        self.cutout = Cutout(16)
        self.indices = indices
        # now 0, 1, 2, ... are mapped to indices[0], indices[1], indices[2], ...

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.indices is not None:
            index = self.indices[index]
        data, target = torch.load(self.root + "/{}.pt".format(index))
        data = self.resize(data.permute([3, 0, 1, 2]))

        if self.transform:

            data = self.flip(data)

            choices = ["roll", "rotate", "shear", "cutout"]
            aug = np.random.choice(choices)
            if aug == "roll":
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            if aug == "rotate":
                data = self.rotate(data)
            if aug == "shear":
                data = self.shearx(data)
            if aug == "cutout":
                data = self.cutout(data)

        return data, target.long().squeeze(-1)

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(os.listdir(self.root))


def build_dvscifar(path="/p/scratch/eelsaisdc/bania1/Cifar10DVS_frames", transform=False):
    train_path = path + "/train"
    val_path = path + "/test"
    train_dataset = DVSCifar10(root=train_path, transform=transform)
    val_dataset = DVSCifar10(root=val_path, transform=False)

    return train_dataset, val_dataset


def build_dvscifar_10_fold(path="/p/scratch/eelsaisdc/bania1/Cifar10DVS_frames", transform=False, fold_idx=0):
    if "indices.pkl" in os.listdir(path + "/combined"):
        # load the indices, this happens when we run adter the first time
        indices = pickle.load(open(path + "/combined/indices.pkl", "rb"))
        train_inds = indices[fold_idx][0]
        val_inds = indices[fold_idx][1]
        train_dataset = DVSCifar10(root=path + "/combined", transform=transform, indices=train_inds)
        val_dataset = DVSCifar10(root=path + "/combined", transform=False, indices=val_inds)
    else:
        print("Creating indices for the 10-fold experiment, it is advisable to launch other runs after a small delay.")
        # create the indices, this happens the first time
        total = 10000
        indices = list(range(total))
        random.shuffle(indices)
        index_tuples = []
        for i in range(10):
            train_indices = indices[: i * 1000] + indices[(i + 1) * 1000 :]
            val_indices = indices[i * 1000 : (i + 1) * 1000]
            index_tuples.append((train_indices, val_indices))

        pickle.dump(index_tuples, open(path + "/combined/indices.pkl", "wb"))
        train_inds = index_tuples[fold_idx][0]
        val_inds = index_tuples[fold_idx][1]
        train_dataset = DVSCifar10(root=path + "/combined", transform=transform, indices=train_inds)
        val_dataset = DVSCifar10(root=path + "/combined", transform=False, indices=val_inds)

    return train_dataset, val_dataset


def dvscifar10_collate_fn(batch):
    # need to apply cutmix here
    data = torch.stack([x[0] for x in batch])
    target = torch.stack([x[1] for x in batch])
    data, target_a, target_b, lam = cutmix_data(data, target)
    lam = torch.tensor(lam).float()
    return data, target_a, target_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W = size[3]
    H = size[4]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(input, target, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(input.size()[0])

    target_a = target
    target_b = target[rand_index]

    # generate mixed sample
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    return input, target_a, target_b, lam


if __name__ == "__main__":
    choices = ["roll", "rotate", "shear"]
    aug = np.random.choice(choices)
    print(aug)

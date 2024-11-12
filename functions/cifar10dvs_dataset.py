"""
Modified from: https://tonic.readthedocs.io/en/latest/_modules/tonic/datasets/cifar10dvs.html#CIFAR10DVS
"""

import os
import random
from concurrent.futures import ThreadPoolExecutor
from time import time

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from tonic.transforms import ToFrame
from torch.utils.data import Dataset
from tqdm import tqdm


class CIFAR10DVS(Dataset):
    """
    Parameters:
        save_to: str = local path to the CIFAR10DVS dataset
        train: bool = whether to load the training or testing set
    """

    dtype = np.dtype([("t", np.uint64), ("x", np.int32), ("y", np.int32), ("p", bool)])
    sensor_size = (128, 128, 2)
    classes = {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,
    }

    def __init__(
        self,
        save_to: str = "/p/project1/eelsaisdc/bhisikar1/datasets/CIFAR10DVS",
        train: bool = True,
    ):
        super().__init__()
        self.save_to = save_to
        self.train = train

        # Load the file paths and labels
        self.data = []
        self.targets = []
        for path, dirs, files in os.walk(save_to):
            dirs.sort()
            for file in files:
                if file.endswith("npy"):
                    self.data.append(os.path.join(path, file))
                    label_number = self.classes[os.path.basename(path)]
                    self.targets.append(label_number)

        # Initialize the frame converter, needed for the buffer loading
        self.to_frame = ToFrame(n_event_bins=10, sensor_size=(128, 128, 2), overlap=0)

        # Initialize the augmentations
        self.resize = transforms.Resize(
            size=(48, 48), interpolation=transforms.InterpolationMode.NEAREST
        )
        self.rotate = transforms.RandomRotation(degrees=30)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-30, 30))
        self.choices = ["roll", "rotate", "shear"]

        # For quick testing
        # self.data = self.data[:1000]
        # self.targets = self.targets[:1000]

        # Initialize the buffer
        self.buffer = [None] * len(self.data)
        self.initalize_memory()

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events, target = self.buffer[index]

        if self.train:
            aug = np.random.choice(self.choices)
            if aug == "roll":
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                events = torch.roll(events, shifts=(off1, off2), dims=(2, 3))
            elif aug == "rotate":
                events = self.rotate(events)
            elif aug == "shear":
                events = self.shearx(events)

        return events, target

    def __len__(self) -> int:
        return len(self.data)

    def read_index(self, index: int) -> None:
        # Load the events and clip them to the frame
        events = np.load(self.data[index]).astype(self.dtype)
        # Clip the events to the frame
        events = np.clip(self.to_frame(events), 0, 1)
        # Convert the events to a tensor
        events = torch.from_numpy(events).float()
        # Resize the events
        events = self.resize(events)

        target = self.targets[index]

        self.buffer[index] = (events, target)

    def initalize_memory(self) -> None:
        start_time = time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(
                tqdm(
                    executor.map(self.read_index, range(len(self.data))),
                    total=len(self.data),
                )
            )

        total_time = time() - start_time

        print(f"Time taken to load the dataset: {total_time:.2f} seconds")


def build_dvscifar_v2(fold_idx: int):
    train_data = CIFAR10DVS(train=True)
    val_data = CIFAR10DVS(train=False)

    kfold = KFold(n_splits=10, shuffle=True, random_state=99)
    fold_indices = list(kfold.split(np.arange(len(train_data))))
    train_indices, val_indices = fold_indices[fold_idx]

    train_data = torch.utils.data.Subset(train_data, train_indices)
    val_data = torch.utils.data.Subset(val_data, val_indices)

    return train_data, val_data

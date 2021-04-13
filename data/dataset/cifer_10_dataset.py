import numpy as np
import torch

from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset


class CIFAR10Dataset:

    def __init__(self, root: str, transform, validation_size: float) -> None:
        self.set_train_and_validation_data(
            train=True,
            download=True,
            root=root,
            transform=transform,
            validation_size=validation_size,
        )

        self.set_test_data(
            train=False, download=True, root=root, transform=transform
        )
    
    def set_train_and_validation_data(
        self,
        download: bool,
        root: str,
        train: bool,
        transform,
        validation_size: float,
    ) -> None:
        train_dataset = CIFAR10(download=download, root=root, train=train, transform=transform)
        validation_dataset = CIFAR10(download=download, root=root, train=train, transform=transform)
        # get all training targets and count the number of class instances
        targets = np.array(train_dataset.targets)
        classes, class_counts = np.unique(targets, return_counts=True)
        num_classes = len(classes)
        # create artificial imbalanced class count
        # [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, track]
        imbal_class_counts = [5000, 5000, 2500, 5000, 2500, 5000, 5000, 5000, 5000, 2500]
        # get class indices
        class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
        # get imbalanced number of instances
        imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]

        # [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, track]
        train_class_counts = [4500, 4500, 2000, 4500, 2000, 4500, 4500, 4500, 4500, 2000]
        val_class_counts = [imbal - train for (imbal, train) in zip(imbal_class_counts, train_class_counts)]
        train_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(imbal_class_indices, train_class_counts)]
        val_class_indices = [class_idx[class_count:] for class_idx, class_count in zip(imbal_class_indices, train_class_counts)]

        train_class_indices = np.hstack(train_class_indices)
        val_class_indices = np.hstack(val_class_indices)

        # set target and data to dataset
        train_dataset.targets = targets[train_class_indices]
        train_dataset.data = train_dataset.data[train_class_indices]
        validation_dataset.targets = targets[val_class_indices]
        validation_dataset.data = validation_dataset.data[val_class_indices]
        train_num = len(train_dataset)
        validation_num = len(validation_dataset)
        # print('Training data is {}, {} in toatal'.format(train_class_counts, train_num))
        # print('Validation data is {}, {} in toatal'.format(val_class_counts, validation_num))

        # split train and val
        indices = list(range(train_num))
        split = int(np.floor(validation_size * train_num))
        train_indices, validation_indices = indices[split:], indices[:split]
        self.train_data_dict = {
            'train_dataset': train_dataset,
            'val_dataset': validation_dataset
        }

    def set_test_data(
        self,
        download: bool,
        root: str,
        train: bool,
        transform,
    ) -> None:
        self.test_data_dict = {
            'dataset': CIFAR10(
                download=download, root=root, train=train, transform=transform
            )
        }

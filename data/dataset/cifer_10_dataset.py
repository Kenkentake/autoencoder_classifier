import numpy as np

from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler


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
        imbal_class_indices = np.hstack(imbal_class_indices)
        # set target and data to dataset
        train_dataset.targets = targets[imbal_class_indices]
        train_dataset.data = train_dataset.data[imbal_class_indices]
        train_num = len(train_dataset)
        print('Train and Val data is {}, {} in toatal'.format(imbal_class_counts, train_num))

        # split train and val
        indices = list(range(train_num))
        split = int(np.floor(validation_size * train_num))
        train_indices, validation_indices = indices[split:], indices[:split]
        self.train_data_dict = {
            'dataset': train_dataset,
            'train_sampler': SubsetRandomSampler(train_indices),
            'validation_sampler': SubsetRandomSampler(validation_indices),
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

import torchvision.transforms as transforms

from data.dataset.cifer_10_dataset import CIFAR10Dataset


def get_transform_from_list(transform_list: list, img_size: int):
    sequence = []

    for t in transform_list:
        if t == 'random_rotation':
            sequence.append(transforms.RandomRotation(degrees=15))
        if t == 'random_horizontal_flip':
            sequence.append(transforms.RandomHorizontalFlip(p=0.5))
        if t == 'random_vertical_flip':
            sequence.append(transforms.RandomVerticalFlip(p=0.5))
        if t == 'resize':
            sequence.append(transforms.Resize((img_size, img_size)))
        if t == 'color_jitter':
            sequence.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
        if t == 'to_tensor':
            sequence.append(transforms.ToTensor())
        if t == 'normalize':
            sequence.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(sequence)


def get_dataset(args):
    transform = get_transform_from_list(args.DATA.TRANSFORM_LIST, args.DATA.IMG_SIZE)
    dataset_type = args.DATA.DATASET_TYPE.lower()

    if dataset_type == 'cifer10':
        dataset = CIFAR10Dataset(
            root=args.DATA.CACHE_DIR,
            sampling_class_counts = args.DATA.SAMPLING_CLASS_COUNTS,
            train_class_counts = args.DATA.TRAIN_CLASS_COUNTS,
            transform=transform,
            validation_size=args.DATA.VALIDATION_SIZE
        )

    else:
        raise NotImplementedError()

    return dataset

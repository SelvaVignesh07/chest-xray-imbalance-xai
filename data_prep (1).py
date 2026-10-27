"""
data_prep.py
============
Phase 1: Data Preparation & Augmentation

Responsibilities:
    1. Build torchvision transform pipelines (train vs. val/test).
    2. Load the dataset via ImageFolder from a standard data/train, data/val,
       data/test directory layout.
    3. Compute class weights (Inverse Class Frequency) from the TRAINING SET
       ONLY, which is the correct methodology -- weights must never be derived
       from validation/test data, or you leak distributional information and
       bias your evaluation.

Expected folder structure (ImageFolder infers class labels from subfolder
names, and sorts them alphabetically -- so "NORMAL" -> 0, "PNEUMONIA" -> 1):

    data/
        train/
            NORMAL/
                img001.jpeg
                ...
            PNEUMONIA/
                img001.jpeg
                ...
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/
"""

import os
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet mean/std -- required because we use ImageNet-pretrained backbones
# (ResNet18/50). The pretrained convolutional filters were learned under this
# specific normalization; feeding data normalized any other way shifts the
# input distribution the network expects and quietly degrades transfer
# learning performance.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_SIZE = 224  # Standard input resolution for ResNet architectures


def get_transforms():
    """
    Build the train / eval transform pipelines.

    Why these specific augmentations and NOT others:
        - RandomHorizontalFlip: A chest X-ray flipped left-right is still a
          medically plausible chest X-ray (heart position is the main
          asymmetry, and models can still learn robust lung-field features).
        - RandomRotation(10): Small rotations simulate slight patient
          positioning variance during acquisition. Kept SMALL (10 degrees)
          because large rotations would create anatomically implausible
          images and could destroy diagnostically relevant spatial cues.
        - We deliberately AVOID RandomVerticalFlip, aggressive color jitter,
          or large-angle rotations/shears -- chest X-rays are grayscale
          anatomical images with a fixed "up/down" orientation (apex of lungs
          up, diaphragm down). Flipping vertically or heavily distorting
          color/contrast produces medically nonsensical training examples
          ("garbage augmentation") and can actively hurt generalization.
        - Validation/Test transforms contain NO augmentation -- only resize +
          normalize. Evaluation must reflect the real, un-augmented data
          distribution the model will see in deployment.

    Returns:
        dict: {'train': Compose, 'val': Compose, 'test': Compose}
    """
    train_transform = transforms.Compose([
        # Chest X-rays are often grayscale (1-channel) DICOM/JPEG exports.
        # ResNet expects 3-channel input (it was pretrained on RGB ImageNet),
        # so we replicate the single channel across 3 channels.
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        # Slight affine jitter (translation only) further simulates
        # positioning variance without distorting anatomy.
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


def get_datasets(data_dir):
    """
    Load train/val/test datasets using ImageFolder.

    Args:
        data_dir (str): path containing 'train', 'val', 'test' subfolders.

    Returns:
        dict: {'train': ImageFolder, 'val': ImageFolder, 'test': ImageFolder}
    """
    transforms_dict = get_transforms()
    image_datasets = {}
    for split in ["train", "val", "test"]:
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(
                f"Expected split directory not found: {split_path}\n"
                f"Make sure your data folder follows data/train, data/val, data/test."
            )
        image_datasets[split] = datasets.ImageFolder(
            root=split_path, transform=transforms_dict[split]
        )
    return image_datasets


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Wrap the datasets in DataLoaders.

    Args:
        data_dir (str): root data directory.
        batch_size (int): batch size for all loaders.
        num_workers (int): parallel data-loading workers (set to 0 on
            Windows/some notebook environments if you hit multiprocessing
            errors).

    Returns:
        (dict of DataLoader, dict of ImageFolder, list of class names)
    """
    image_datasets = get_datasets(data_dir)

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        ),
        "val": DataLoader(
            image_datasets["val"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            image_datasets["test"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }

    class_names = image_datasets["train"].classes  # e.g. ['NORMAL', 'PNEUMONIA']
    return dataloaders, image_datasets, class_names


def calculate_class_weights(train_dataset):
    """
    Compute Inverse Class Frequency weights from the TRAINING split.

    Rationale (this is the mathematical core of the "Imbalance-Aware"
    experiment):
        In datasets like the classic chest X-ray Pneumonia dataset, PNEUMONIA
        images heavily outnumber NORMAL images (roughly 3:1 or worse). A
        vanilla nn.CrossEntropyLoss() treats every training example equally,
        so the optimizer finds it "cheap" to just lean toward predicting the
        majority class -- this is the Accuracy Paradox: the model can score
        >85% accuracy while still missing a meaningful fraction of the
        minority class, which in a medical context could mean missing a
        NORMAL misclassified as PNEUMONIA (false positive) or, more
        dangerously in other imbalance directions, missing real disease
        (false negative).

        Inverse Class Frequency weighting assigns:
            w_c = N_total / (num_classes * N_c)
        where N_c is the number of training samples in class c. This means
        the *minority* class gets a proportionally LARGER weight, so
        misclassifying it contributes more to the loss. The optimizer is now
        mathematically penalized more heavily for errors on the minority
        class, discouraging it from collapsing to majority-class predictions.

    Args:
        train_dataset (ImageFolder): the training ImageFolder dataset.

    Returns:
        torch.FloatTensor: class weights ordered to match
            train_dataset.classes / train_dataset.class_to_idx, ready to be
            passed directly as nn.CrossEntropyLoss(weight=...).
    """
    # train_dataset.targets is a list of integer class indices, one per image,
    # in the SAME order as train_dataset.classes / class_to_idx. Counting
    # these gives us N_c for every class without touching val/test data.
    targets = train_dataset.targets
    class_counts = Counter(targets)
    num_classes = len(train_dataset.classes)
    total_samples = len(targets)

    weights = []
    for class_idx in range(num_classes):
        n_c = class_counts[class_idx]
        w_c = total_samples / (num_classes * n_c)
        weights.append(w_c)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print("Class distribution (training set):")
    for class_idx, class_name in enumerate(train_dataset.classes):
        print(f"  {class_name:12s} (idx {class_idx}): "
              f"{class_counts[class_idx]:5d} images  ->  weight = {weights[class_idx]:.4f}")

    return weights_tensor


if __name__ == "__main__":
    # Quick standalone sanity check: run `python data_prep.py` after placing
    # data under ./data/{train,val,test}/{NORMAL,PNEUMONIA}/
    dataloaders, image_datasets, class_names = get_dataloaders("data", batch_size=32)
    print(f"Classes found: {class_names}")
    for split in ["train", "val", "test"]:
        print(f"{split}: {len(image_datasets[split])} images")
    class_weights = calculate_class_weights(image_datasets["train"])
    print("Class weights tensor:", class_weights)

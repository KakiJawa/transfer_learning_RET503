import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


DATASET_DIR = "dataset_split"

BATCH_SIZE = 16
NUM_WORKERS = 0


# Transform untuk data training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Transform untuk validation dan test
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Dataset
train_dataset = datasets.ImageFolder(
    f"{DATASET_DIR}/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    f"{DATASET_DIR}/val",
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    f"{DATASET_DIR}/test",
    transform=val_test_transform
)


# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


if __name__ == "__main__":
    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)

    print("Classes:", train_dataset.classes)
    print("Class mapping:", train_dataset.class_to_idx)

    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    images, labels = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)

    print("=" * 50)
    print("DataLoader: OK")
    print("=" * 50)
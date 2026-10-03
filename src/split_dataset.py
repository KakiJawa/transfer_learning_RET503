import os
import shutil
import random

SOURCE_DIR = "dataset_raw"
OUTPUT_DIR = "dataset_split"

CLASSES = ["plastic", "paper", "metal", "glass"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

random.seed(SEED)

for class_name in CLASSES:
    source_class_dir = os.path.join(SOURCE_DIR, class_name)

    files = [
        f for f in os.listdir(source_class_dir)
        if os.path.splitext(f)[1].lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    files.sort()
    random.shuffle(files)

    total = len(files)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    split_files = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split_name, split_list in split_files.items():

        output_class_dir = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_name
        )

        os.makedirs(output_class_dir, exist_ok=True)

        for filename in split_list:
            source = os.path.join(source_class_dir, filename)
            destination = os.path.join(output_class_dir, filename)

            shutil.copy2(source, destination)

        print(
            f"{class_name:8s} | "
            f"{split_name:5s} | "
            f"{len(split_list)} gambar"
        )

print("\n" + "=" * 50)
print("SPLIT DATASET SELESAI")
print("=" * 50)
print(f"Random seed : {SEED}")
print("Train       : 70%")
print("Validation  : 15%")
print("Test        : 15%")
print("=" * 50)
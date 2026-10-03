from PIL import Image
import os

DATASET_DIR = "dataset_raw"
CLASSES = ["plastic", "paper", "metal", "glass"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total = 0
bad = []

for class_name in CLASSES:
    class_dir = os.path.join(DATASET_DIR, class_name)

    for filename in os.listdir(class_dir):
        filepath = os.path.join(class_dir, filename)

        if os.path.splitext(filename)[1].lower() not in EXTENSIONS:
            continue

        total += 1

        try:
            with Image.open(filepath) as img:
                img.verify()
        except Exception:
            bad.append(filepath)

print("=" * 50)
print("DATASET CHECK")
print("=" * 50)
print(f"Total gambar : {total}")
print(f"Gambar rusak : {len(bad)}")

if bad:
    print("\nFile bermasalah:")
    for filepath in bad:
        print(filepath)
else:
    print("\nSemua gambar dapat dibaca dengan baik.")

print("=" * 50)
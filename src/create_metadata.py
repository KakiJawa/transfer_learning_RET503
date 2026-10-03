import os
import csv
from PIL import Image

DATASET_DIR = "dataset_raw"
OUTPUT_FILE = "metadata.csv"

CLASSES = ["plastic", "paper", "metal", "glass"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

rows = []

for class_name in CLASSES:
    class_dir = os.path.join(DATASET_DIR, class_name)

    for filename in sorted(os.listdir(class_dir)):
        filepath = os.path.join(class_dir, filename)

        if os.path.splitext(filename)[1].lower() not in EXTENSIONS:
            continue

        try:
            with Image.open(filepath) as img:
                width, height = img.size

            rows.append({
                "filename": filename,
                "label": class_name,
                "source": "TrashNet",
                "width": width,
                "height": height
            })

        except Exception as e:
            print(f"Gagal membaca: {filepath}")
            print(e)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["filename", "label", "source", "width", "height"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("=" * 50)
print("METADATA BERHASIL DIBUAT")
print("=" * 50)
print(f"Total data : {len(rows)}")
print(f"Output     : {OUTPUT_FILE}")
print("=" * 50)
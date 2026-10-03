import os
import time

import torch
from PIL import Image
from torchvision import transforms

from model import create_model


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/partial_lr1e-3_best.pth"

IMAGE_PATH = "dataset_split/test/plastic/"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

WARMUP_RUNS = 20
MEASURE_RUNS = 100


# ==========================================
# Find one test image
# ==========================================

image_file = None

for root, dirs, files in os.walk(IMAGE_PATH):
    for file in files:
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            image_file = os.path.join(
                root,
                file
            )
            break

    if image_file is not None:
        break


if image_file is None:
    raise FileNotFoundError(
        "Tidak ditemukan gambar test."
    )


print("=" * 60)
print("LATENCY MEASUREMENT")
print("=" * 60)
print(f"Device       : {DEVICE}")
print(f"Model        : {MODEL_PATH}")
print(f"Test image   : {image_file}")
print(f"Image size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Warm-up runs : {WARMUP_RUNS}")
print(f"Measure runs : {MEASURE_RUNS}")
print("=" * 60)


# ==========================================
# Transform
# ==========================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# Load image
# ==========================================

image = Image.open(
    image_file
).convert("RGB")

input_tensor = transform(
    image
).unsqueeze(0)

input_tensor = input_tensor.to(
    DEVICE
)


# ==========================================
# Load model
# ==========================================

model = create_model(
    "partial"
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model = model.to(DEVICE)
model.eval()


# ==========================================
# Warm-up
# ==========================================

print()
print("Melakukan warm-up...")

with torch.no_grad():

    for _ in range(WARMUP_RUNS):

        _ = model(
            input_tensor
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()


# ==========================================
# Measure inference latency
# ==========================================

print("Mengukur latency...")

times = []


with torch.no_grad():

    for _ in range(MEASURE_RUNS):

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        _ = model(
            input_tensor
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        times.append(
            latency_ms
        )


# ==========================================
# Statistics
# ==========================================

average_latency = sum(times) / len(times)

min_latency = min(times)

max_latency = max(times)

fps = 1000 / average_latency


# ==========================================
# Result
# ==========================================

print()
print("=" * 60)
print("LATENCY RESULT")
print("=" * 60)

print(
    f"Average latency : {average_latency:.3f} ms"
)

print(
    f"Minimum latency : {min_latency:.3f} ms"
)

print(
    f"Maximum latency : {max_latency:.3f} ms"
)

print(
    f"Estimated FPS   : {fps:.2f}"
)

print("=" * 60)
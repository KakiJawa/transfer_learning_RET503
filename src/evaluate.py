import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import create_model


# ==========================================
# Configuration
# ==========================================

DATASET_DIR = "dataset_split"
MODEL_DIR = "models"
RESULTS_DIR = "results"

BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


EXPERIMENTS = [
    {
        "name": "feature_lr1e-3",
        "mode": "feature"
    },
    {
        "name": "partial_lr1e-4",
        "mode": "partial"
    },
    {
        "name": "partial_lr1e-3",
        "mode": "partial"
    },
    {
        "name": "scratch_lr1e-3",
        "mode": "scratch"
    }
]


# ==========================================
# Test transform
# ==========================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# Load test dataset
# ==========================================

test_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "test"),
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

class_names = test_dataset.classes


print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"Device : {DEVICE}")
print(f"Classes: {class_names}")
print(f"Test   : {len(test_dataset)} images")
print("=" * 60)


all_results = []


# ==========================================
# Evaluate models
# ==========================================

for experiment in EXPERIMENTS:

    name = experiment["name"]
    mode = experiment["mode"]

    model_path = os.path.join(
        MODEL_DIR,
        f"{name}_best.pth"
    )

    print()
    print("=" * 60)
    print(f"EVALUATING: {name}")
    print("=" * 60)

    # Create model
    model = create_model(mode)
    

    # Load trained weights
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model = model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    # ======================================
    # Prediction
    # ======================================

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    # ======================================
    # Accuracy
    # ======================================

    accuracy = (
        sum(
            true == pred
            for true, pred in zip(y_true, y_pred)
        )
        / len(y_true)
        * 100
    )

    print(f"Test Accuracy: {accuracy:.2f}%")

    # ======================================
    # Classification Report
    # ======================================

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    print()
    print("Classification Report:")
    print(report_text)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_path = os.path.join(
        RESULTS_DIR,
        f"{name}_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    # ======================================
    # Confusion Matrix
    # ======================================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        f"Confusion Matrix - {name}"
    )

    plt.tight_layout()

    cm_path = os.path.join(
        RESULTS_DIR,
        f"{name}_confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=150
    )

    plt.close()

    # ======================================
    # Summary
    # ======================================

    all_results.append({
        "experiment": name,
        "mode": mode,
        "test_accuracy": accuracy,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"]
    })

    print(f"Confusion matrix : {cm_path}")
    print(f"Classification   : {report_path}")


# ==========================================
# Save evaluation summary
# ==========================================

summary_df = pd.DataFrame(
    all_results
)

summary_path = os.path.join(
    RESULTS_DIR,
    "evaluation_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


print()
print("=" * 60)
print("EVALUATION SELESAI")
print("=" * 60)
print(f"Summary: {summary_path}")
print("=" * 60)
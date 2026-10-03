import os
import csv
import time

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, val_loader, test_loader
from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 20

RESULTS_DIR = "results"
MODELS_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


EXPERIMENTS = [
    {
        "name": "feature_lr1e-3",
        "mode": "feature",
        "lr": 1e-3
    },
    {
        "name": "partial_lr1e-4",
        "mode": "partial",
        "lr": 1e-4
    },
    {
        "name": "partial_lr1e-3",
        "mode": "partial",
        "lr": 1e-3
    },
    {
        "name": "scratch_lr1e-3",
        "mode": "scratch",
        "lr": 1e-3
    }
]


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# TEST FUNCTION
# ============================================================

def test_model(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(history, filepath):

    with open(filepath, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy"
        ])

        for row in history:
            writer.writerow(row)


# ============================================================
# MAIN TRAINING
# ============================================================

def run_experiment(experiment):

    name = experiment["name"]
    mode = experiment["mode"]
    lr = experiment["lr"]

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}")
    print(f"MODE      : {mode}")
    print(f"LEARNING RATE: {lr}")
    print(f"DEVICE    : {DEVICE}")
    print("=" * 70)

    # Create model
    model = create_model(mode)
    model = model.to(DEVICE)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Only optimize trainable parameters
    trainable_parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = optim.Adam(
        trainable_parameters,
        lr=lr
    )

    best_val_accuracy = 0.0

    history = []

    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion
        )

        history.append([
            epoch,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy
        ])

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            model_path = os.path.join(
                MODELS_DIR,
                f"{name}_best.pth"
            )

            torch.save(
                model.state_dict(),
                model_path
            )

    end_time = time.perf_counter()

    training_time = end_time - start_time

    # Load best model
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    test_accuracy = test_model(
        model,
        test_loader
    )

    # Save history
    history_path = os.path.join(
        RESULTS_DIR,
        f"{name}_history.csv"
    )

    save_history(
        history,
        history_path
    )

    print("\nRESULT")
    print("-" * 50)
    print(f"Best Validation Accuracy : {best_val_accuracy * 100:.2f}%")
    print(f"Test Accuracy            : {test_accuracy * 100:.2f}%")
    print(f"Training Time            : {training_time:.2f} seconds")
    print(f"Model                    : {model_path}")
    print(f"History                  : {history_path}")
    print("-" * 50)

    return {
        "experiment": name,
        "mode": mode,
        "learning_rate": lr,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "training_time_seconds": training_time
    }


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TRANSFER LEARNING - RESNET-18")
    print("=" * 70)
    print(f"Device : {DEVICE}")
    print(f"Epochs : {EPOCHS}")

    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    results = []

    for experiment in EXPERIMENTS:

        result = run_experiment(experiment)

        results.append(result)

    # Save comparison
    comparison_path = os.path.join(
        RESULTS_DIR,
        "comparison.csv"
    )

    with open(
        comparison_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "mode",
                "learning_rate",
                "best_val_accuracy",
                "test_accuracy",
                "training_time_seconds"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(f"Comparison file: {comparison_path}")
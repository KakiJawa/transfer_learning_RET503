import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)


experiments = [
    "feature_lr1e-3",
    "partial_lr1e-4",
    "partial_lr1e-3",
    "scratch_lr1e-3",
]


for experiment in experiments:

    csv_path = os.path.join(
        RESULTS_DIR,
        f"{experiment}_history.csv"
    )

    df = pd.read_csv(csv_path)

    # ==============================
    # Accuracy per Epoch
    # ==============================
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["train_accuracy"] * 100,
        label="Train Accuracy"
    )

    plt.plot(
        df["epoch"],
        df["val_accuracy"] * 100,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy per Epoch - {experiment}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    accuracy_path = os.path.join(
        PLOT_DIR,
        f"{experiment}_accuracy.png"
    )

    plt.savefig(accuracy_path, dpi=150)
    plt.close()


    # ==============================
    # Loss per Epoch
    # ==============================
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        df["epoch"],
        df["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss per Epoch - {experiment}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_path = os.path.join(
        PLOT_DIR,
        f"{experiment}_loss.png"
    )

    plt.savefig(loss_path, dpi=150)
    plt.close()


    print(f"{experiment}: grafik berhasil dibuat")


print()
print("=" * 50)
print("SEMUA GRAFIK SELESAI")
print("=" * 50)
print(f"Folder: {PLOT_DIR}")
print("=" * 50)
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Folder hasil
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

# Data hasil eksperimen
data = [
    {
        "Experiment": "Feature LR 1e-3",
        "Mode": "Feature",
        "Learning Rate": "1e-3",
        "Best Val Accuracy (%)": 68.33,
        "Test Accuracy (%)": 76.67,
        "Training Time (s)": 58.80
    },
    {
        "Experiment": "Partial LR 1e-4",
        "Mode": "Partial",
        "Learning Rate": "1e-4",
        "Best Val Accuracy (%)": 85.00,
        "Test Accuracy (%)": 81.67,
        "Training Time (s)": 74.37
    },
    {
        "Experiment": "Partial LR 1e-3",
        "Mode": "Partial",
        "Learning Rate": "1e-3",
        "Best Val Accuracy (%)": 78.33,
        "Test Accuracy (%)": 86.67,
        "Training Time (s)": 75.24
    },
    {
        "Experiment": "Scratch LR 1e-3",
        "Mode": "Scratch",
        "Learning Rate": "1e-3",
        "Best Val Accuracy (%)": 63.33,
        "Test Accuracy (%)": 48.33,
        "Training Time (s)": 116.47
    }
]

# Buat DataFrame
df = pd.DataFrame(data)

# Pastikan folder tersedia
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Simpan tabel lengkap
summary_path = RESULTS_DIR / "comparison_summary.csv"
df.to_csv(summary_path, index=False)

print("=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(df.to_string(index=False))

# ---------------------------------------------------------
# Grafik 1: Accuracy Comparison
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

x = range(len(df))
width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    df["Best Val Accuracy (%)"],
    width,
    label="Best Validation Accuracy"
)

plt.bar(
    [i + width / 2 for i in x],
    df["Test Accuracy (%)"],
    width,
    label="Test Accuracy"
)

plt.xticks(list(x), df["Experiment"], rotation=15)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison of Transfer Learning Experiments")
plt.ylim(0, 100)
plt.legend()
plt.grid(axis="y", alpha=0.3)

# Tampilkan nilai
for i, value in enumerate(df["Best Val Accuracy (%)"]):
    plt.text(i - width / 2, value + 2, f"{value:.2f}%",
             ha="center", fontsize=9)

for i, value in enumerate(df["Test Accuracy (%)"]):
    plt.text(i + width / 2, value + 2, f"{value:.2f}%",
             ha="center", fontsize=9)

plt.tight_layout()

accuracy_path = PLOTS_DIR / "comparison_accuracy.png"
plt.savefig(accuracy_path, dpi=300)
plt.close()

# ---------------------------------------------------------
# Grafik 2: Training Time
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    df["Experiment"],
    df["Training Time (s)"]
)

plt.ylabel("Training Time (seconds)")
plt.title("Training Time Comparison")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)

for i, value in enumerate(df["Training Time (s)"]):
    plt.text(i, value + 2, f"{value:.2f}s",
             ha="center", fontsize=9)

plt.tight_layout()

time_path = PLOTS_DIR / "comparison_training_time.png"
plt.savefig(time_path, dpi=300)
plt.close()

print("\nFile berhasil dibuat:")
print(f"- {summary_path}")
print(f"- {accuracy_path}")
print(f"- {time_path}")

print("\nComparison selesai.")
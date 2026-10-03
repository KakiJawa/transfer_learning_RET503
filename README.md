# Transfer Learning untuk Klasifikasi Sampah

Project ini merupakan implementasi transfer learning untuk melakukan klasifikasi gambar sampah ke dalam empat kategori, yaitu **plastic, paper, metal, dan glass** menggunakan arsitektur **ResNet-18**.

Project dibuat sebagai bagian dari tugas mata kuliah Computer Vision & Deep Learning.

## Google Colab

Notebook lengkap dapat dibuka langsung di Google Colab melalui tombol berikut:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KakiJawa/transfer_learning_RET503/blob/main/notebooks/transfer_learning_waste_classification.ipynb)

Notebook sudah berisi hasil eksekusi eksperimen, termasuk training, grafik accuracy dan loss, evaluasi test set, classification report, confusion matrix, perbandingan eksperimen, dan inference latency.

## 1. Tujuan

Tujuan project ini adalah membangun dan mengevaluasi model klasifikasi citra sampah dengan pendekatan transfer learning serta membandingkan performa beberapa mode pelatihan.

Mode yang dibandingkan:

1. **Feature** — menggunakan bobot ImageNet dan hanya melatih classifier.
2. **Partial** — menggunakan bobot ImageNet dan melakukan fine-tuning pada `layer4` serta classifier.
3. **Scratch** — melatih seluruh arsitektur dari bobot acak tanpa pretrained weights.

Selain akurasi, project ini juga mengukur waktu training dan latency inference model.

---

## 2. Dataset

Dataset yang digunakan adalah **TrashNet** dengan empat kelas yang digunakan:

* Plastic
* Paper
* Metal
* Glass

Jumlah dataset:

| Class     |  Jumlah |
| --------- | ------: |
| Plastic   |     100 |
| Paper     |     100 |
| Metal     |     100 |
| Glass     |     100 |
| **Total** | **400** |

Dataset dibagi menggunakan rasio:

* Training: 70% = 280 gambar
* Validation: 15% = 60 gambar
* Testing: 15% = 60 gambar

Pembagian dilakukan dengan random seed `42` agar pembagian dataset dapat direproduksi.

---

## 3. Arsitektur Model

Model yang digunakan adalah **ResNet-18**.

Input gambar diubah menjadi ukuran:

```text
224 × 224 × 3
```

Classifier bawaan ResNet-18 diganti menjadi:

```text
Linear → 4 classes
```

Karena terdapat empat kategori sampah.

Jumlah parameter model:

| Mode    | Total Parameters | Trainable Parameters |
| ------- | ---------------: | -------------------: |
| Feature |       11,178,564 |                2,052 |
| Partial |       11,178,564 |            8,395,780 |
| Scratch |       11,178,564 |           11,178,564 |

---

## 4. Konfigurasi Training

Konfigurasi utama:

| Parameter         | Value                |
| ----------------- | -------------------- |
| Model             | ResNet-18            |
| Image Size        | 224 × 224            |
| Batch Size        | 16                   |
| Epoch             | 20                   |
| Optimizer         | Adam                 |
| Loss Function     | CrossEntropyLoss     |
| Dataset           | TrashNet             |
| Number of Classes | 4                    |
| Device            | NVIDIA GeForce MX450 |

Data augmentation pada training menggunakan:

* Random Horizontal Flip
* Random Rotation ±10°

Normalisasi menggunakan mean dan standard deviation dari ImageNet.

---

## 5. Skenario Eksperimen

Empat eksperimen dilakukan berdasarkan mode dan learning rate.

| Experiment      | Mode    | Initial Weights | Trainable Layers | Learning Rate |
| --------------- | ------- | --------------- | ---------------- | ------------: |
| Feature LR 1e-3 | Feature | ImageNet        | `fc`             |          1e-3 |
| Partial LR 1e-4 | Partial | ImageNet        | `layer4 + fc`    |          1e-4 |
| Partial LR 1e-3 | Partial | ImageNet        | `layer4 + fc`    |          1e-3 |
| Scratch LR 1e-3 | Scratch | Random          | All layers       |          1e-3 |

---

## 6. Hasil Eksperimen

Hasil evaluasi pada validation set dan test set:

| Experiment      | Mode    | Learning Rate | Best Val Accuracy | Test Accuracy | Training Time |
| --------------- | ------- | ------------: | ----------------: | ------------: | ------------: |
| Feature LR 1e-3 | Feature |          1e-3 |            68.33% |        76.67% |       58.80 s |
| Partial LR 1e-4 | Partial |          1e-4 |        **85.00%** |        81.67% |       74.37 s |
| Partial LR 1e-3 | Partial |          1e-3 |            78.33% |    **86.67%** |       75.24 s |
| Scratch LR 1e-3 | Scratch |          1e-3 |            63.33% |        48.33% |      116.47 s |

Dari eksperimen yang dilakukan, konfigurasi **Partial LR 1e-3** menghasilkan test accuracy tertinggi, yaitu **86.67%**.

Konfigurasi tersebut menggunakan pretrained ResNet-18 dari ImageNet dengan fine-tuning pada `layer4` dan classifier.

Model ini kemudian digunakan untuk pengukuran latency inference.

---

## 7. Performa Per Kelas

Classification report untuk model Partial LR 1e-3 pada test set:

| Class             | Precision |   Recall | F1-Score |
| ----------------- | --------: | -------: | -------: |
| Glass             |      0.80 |     0.80 |     0.80 |
| Metal             |      0.78 |     0.93 |     0.85 |
| Paper             |      0.93 |     0.93 |     0.93 |
| Plastic           |      1.00 |     0.80 |     0.89 |
| **Macro Average** |  **0.88** | **0.87** | **0.87** |

Hasil menunjukkan bahwa kelas **paper** memiliki precision, recall, dan F1-score yang relatif tinggi. Pada kelas plastic, precision mencapai 1.00, tetapi recall sebesar 0.80 menunjukkan bahwa masih terdapat beberapa gambar plastic yang tidak berhasil dikenali sebagai plastic.

---

## 8. Analisis Training

Pada eksperimen Feature, model hanya melatih classifier sehingga waktu training relatif lebih singkat. Validation accuracy mencapai 68.33%, sedangkan test accuracy mencapai 76.67%.

Pada eksperimen Partial, sebagian layer pretrained ikut dilatih. Hasil menunjukkan bahwa learning rate memberikan pengaruh terhadap performa. Partial dengan learning rate `1e-4` memperoleh validation accuracy tertinggi sebesar 85.00%, sedangkan Partial dengan learning rate `1e-3` memperoleh test accuracy tertinggi sebesar 86.67%.

Pada Partial LR 1e-4, training accuracy meningkat hingga mendekati 100%, sementara validation accuracy cenderung stagnan atau menurun. Hal ini menunjukkan adanya indikasi **overfitting**.

Pada eksperimen Scratch, seluruh parameter model dilatih dari bobot acak. Model memperoleh test accuracy sebesar 48.33% dan membutuhkan waktu training paling lama, yaitu 116.47 detik. Pada dataset yang relatif kecil ini, model dari scratch menunjukkan hasil yang lebih rendah dibandingkan beberapa eksperimen yang menggunakan pretrained weights.

---

## 9. Grafik Hasil

Grafik accuracy dan loss untuk setiap eksperimen tersedia pada folder:

```text
results/plots/
```

Grafik perbandingan accuracy:

```text
results/plots/comparison_accuracy.png
```

Grafik perbandingan waktu training:

```text
results/plots/comparison_training_time.png
```

Grafik training setiap eksperimen:

```text
feature_lr1e-3_accuracy.png
feature_lr1e-3_loss.png

partial_lr1e-4_accuracy.png
partial_lr1e-4_loss.png

partial_lr1e-3_accuracy.png
partial_lr1e-3_loss.png

scratch_lr1e-3_accuracy.png
scratch_lr1e-3_loss.png
```

Confusion matrix dan classification report masing-masing eksperimen juga tersedia pada folder `results/`.

---

## 10. Latency Inference

Model yang digunakan untuk pengukuran latency adalah:

```text
Partial LR 1e-3
```

Model:

```text
models/partial_lr1e-3_best.pth
```

Pengukuran dilakukan dengan:

* Input: 1 gambar
* Image size: 224 × 224
* Warm-up: 20 inference
* Measurement: 100 inference
* GPU synchronization digunakan saat pengukuran

Hasil:

| Metric          |    Result |
| --------------- | --------: |
| Average Latency |  8.551 ms |
| Minimum Latency |  7.933 ms |
| Maximum Latency | 32.139 ms |
| Estimated FPS   |    116.94 |

Nilai FPS tersebut merupakan estimasi throughput inference model dan **bukan pengukuran FPS keseluruhan sistem kamera atau end-to-end pipeline**.

---

## 11. Struktur Project

```text
transfer_learning_sampah/
│
├── dataset_raw/
│   ├── plastic/
│   ├── paper/
│   ├── metal/
│   └── glass/
│
├── dataset_split/
│   ├── train/
│   ├── val/
│   └── test/
│
├── src/
│   ├── check_dataset.py
│   ├── create_comparison.py
│   ├── create_metadata.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── measure_latency.py
│   ├── model.py
│   ├── plot_results.py
│   ├── split_dataset.py
│   └── train.py
│
├── results/
│   ├── plots/
│   ├── evaluation_summary.csv
│   ├── comparison_summary.csv
│   ├── classification reports
│   ├── confusion matrices
│   └── training histories
│
├── models/
│   ├── feature_lr1e-3_best.pth
│   ├── partial_lr1e-4_best.pth
│   ├── partial_lr1e-3_best.pth
│   └── scratch_lr1e-3_best.pth
│
├── metadata.csv
├── README.md
└── requirements.txt
```

---

## 12. Cara Menjalankan

Aktifkan virtual environment:

```cmd
.venv\Scripts\activate
```

### Check dataset

```cmd
python src/check_dataset.py
```

### Membuat metadata

```cmd
python src/create_metadata.py
```

### Membagi dataset

```cmd
python src/split_dataset.py
```

### Mengecek DataLoader

```cmd
python src/dataset.py
```

### Mengecek model

```cmd
python src/model.py
```

### Training

```cmd
python src/train.py
```

### Membuat grafik training

```cmd
python src/plot_results.py
```

### Evaluasi model

```cmd
python src/evaluate.py
```

### Membuat perbandingan hasil

```cmd
python src/create_comparison.py
```

### Mengukur latency

```cmd
python src/measure_latency.py
```

---

## 13. Catatan

Dataset test hanya terdiri dari 60 gambar, sehingga satu gambar merepresentasikan sekitar **1.67 percentage point** pada nilai accuracy. Oleh karena itu, hasil eksperimen perlu dipertimbangkan bersama ukuran dataset dan tidak dapat langsung digeneralisasikan ke dataset yang lebih besar.

Model checkpoint disimpan berdasarkan **validation accuracy terbaik**, kemudian checkpoint tersebut digunakan untuk evaluasi pada test set.

---

## 14. Kesimpulan

Project ini membandingkan tiga pendekatan training ResNet-18, yaitu Feature, Partial, dan Scratch, dengan beberapa konfigurasi learning rate.

Berdasarkan eksperimen yang dilakukan pada dataset ini, konfigurasi **Partial dengan learning rate 1e-3** memperoleh test accuracy sebesar **86.67%**. Model tersebut memiliki average inference latency sebesar **8.551 ms per gambar** pada NVIDIA GeForce MX450.

Hasil eksperimen menunjukkan adanya perbedaan performa antara penggunaan pretrained weights dan training dari scratch pada dataset yang digunakan.

# Initial Design Document

## Waste Classification Using Transfer Learning

### 1. Project Overview

Proyek ini bertujuan untuk membangun sistem klasifikasi gambar sampah menggunakan metode **transfer learning** pada model Convolutional Neural Network (CNN). Sistem akan mengklasifikasikan gambar sampah ke dalam empat kategori, yaitu:

1. Plastic
2. Paper
3. Metal
4. Glass

Model yang digunakan adalah **ResNet-18**. Beberapa skenario training akan dibandingkan untuk mengetahui pengaruh penggunaan pretrained model dan tingkat fine-tuning terhadap performa klasifikasi.

---

### 2. Objectives

Tujuan dari proyek ini adalah:

* Membangun model klasifikasi gambar untuk mengenali jenis sampah.
* Menerapkan metode transfer learning menggunakan ResNet-18 dengan pretrained ImageNet.
* Membandingkan tiga metode training, yaitu Feature, Partial, dan Scratch.
* Membandingkan penggunaan learning rate yang berbeda pada mode Partial.
* Mengevaluasi performa model menggunakan validation accuracy dan test accuracy.
* Mengukur waktu inferensi model untuk mengetahui latency prediksi.

---

### 3. Dataset

Dataset yang digunakan berasal dari **TrashNet**, dengan empat kategori sampah yang digunakan dalam proyek ini.

| Class     | Jumlah Gambar |
| --------- | ------------: |
| Plastic   |           100 |
| Paper     |           100 |
| Metal     |           100 |
| Glass     |           100 |
| **Total** |       **400** |

Setiap kelas memiliki jumlah gambar yang sama sehingga dataset yang digunakan bersifat seimbang berdasarkan jumlah gambar per kelas.

Dataset mentah disimpan pada folder:

```text
dataset_raw/
├── plastic/
├── paper/
├── metal/
└── glass/
```

Metadata dataset disimpan dalam file `metadata.csv`, yang mencatat nama file, label, sumber dataset, serta ukuran gambar.

---

### 4. Dataset Split

Dataset dibagi menjadi tiga bagian:

* Training: 70%
* Validation: 15%
* Testing: 15%

Dengan total 400 gambar, pembagian dataset menjadi:

| Split      |  Jumlah |
| ---------- | ------: |
| Train      |     280 |
| Validation |      60 |
| Test       |      60 |
| **Total**  | **400** |

Pembagian dilakukan menggunakan random seed `42` agar pembagian dataset dapat direproduksi.

Dataset hasil pembagian disimpan pada folder `dataset_split/`. Folder ini tidak dimasukkan ke repository GitHub karena dapat dibuat kembali menggunakan script `src/split_dataset.py`.

---

### 5. Data Preprocessing and Augmentation

Semua gambar akan diproses dengan ukuran input:

```text
224 × 224 pixels
```

Untuk data training digunakan augmentation:

* Resize menjadi 224 × 224
* Random Horizontal Flip
* Random Rotation hingga 10 derajat
* Konversi gambar menjadi tensor
* Normalisasi menggunakan mean dan standard deviation ImageNet

Untuk validation dan testing digunakan:

* Resize menjadi 224 × 224
* Konversi gambar menjadi tensor
* Normalisasi ImageNet

Augmentation hanya diterapkan pada data training agar variasi data training meningkat tanpa mengubah data validation dan test.

---

### 6. Model Architecture

Model yang digunakan adalah **ResNet-18**.

ResNet-18 dipilih karena memiliki arsitektur CNN yang relatif ringan sehingga sesuai untuk eksperimen transfer learning dan dapat digunakan pada perangkat dengan GPU terbatas.

Model menggunakan empat output class:

```text
glass
metal
paper
plastic
```

Layer fully connected (`fc`) pada model asli ImageNet akan diganti menjadi layer baru dengan empat output neuron sesuai jumlah kelas dataset.

---

### 7. Training Modes

Tiga mode training digunakan dalam eksperimen.

#### 7.1 Feature Extraction

Pada mode Feature, ResNet-18 menggunakan bobot pretrained dari ImageNet.

Backbone model dibekukan sehingga parameter pada backbone tidak diperbarui selama training. Hanya layer classifier (`fc`) yang dilatih.

Konfigurasi:

| Parameter        | Configuration       |
| ---------------- | ------------------- |
| Initial weights  | ImageNet pretrained |
| Trainable layers | `fc`                |
| Learning rate    | `1e-3`              |

Tujuan mode ini adalah menguji kemampuan fitur yang telah dipelajari dari ImageNet untuk melakukan klasifikasi pada dataset sampah.

---

#### 7.2 Partial Fine-Tuning

Pada mode Partial, model menggunakan bobot pretrained ImageNet, tetapi sebagian layer backbone ikut dilatih.

Layer yang digunakan untuk fine-tuning adalah:

```text
layer4
fc
```

Dua learning rate akan diuji:

| Experiment      | Initial Weights | Trainable Layers | Learning Rate |
| --------------- | --------------- | ---------------- | ------------- |
| Partial LR 1e-4 | ImageNet        | `layer4 + fc`    | `1e-4`        |
| Partial LR 1e-3 | ImageNet        | `layer4 + fc`    | `1e-3`        |

Eksperimen ini digunakan untuk melihat pengaruh learning rate terhadap proses fine-tuning.

---

#### 7.3 Training from Scratch

Pada mode Scratch, ResNet-18 tidak menggunakan pretrained weights.

Semua parameter model diinisialisasi secara acak dan seluruh layer dilatih.

Konfigurasi:

| Parameter        | Configuration |
| ---------------- | ------------- |
| Initial weights  | Random        |
| Trainable layers | All layers    |
| Learning rate    | `1e-3`        |

Mode ini digunakan sebagai pembanding terhadap pendekatan transfer learning.

---

### 8. Training Configuration

Konfigurasi training yang digunakan:

| Parameter         | Value            |
| ----------------- | ---------------- |
| Architecture      | ResNet-18        |
| Number of classes | 4                |
| Image size        | 224 × 224        |
| Batch size        | 16               |
| Epochs            | 20               |
| Optimizer         | Adam             |
| Loss function     | CrossEntropyLoss |
| Feature LR        | `1e-3`           |
| Partial LR        | `1e-4`, `1e-3`   |
| Scratch LR        | `1e-3`           |
| Random seed       | 42               |

Training dilakukan menggunakan GPU NVIDIA GeForce MX450 apabila CUDA tersedia.

---

### 9. Experimental Design

Total terdapat empat eksperimen:

| Experiment | Mode    | Initial Weights | Trainable Layers | Learning Rate |
| ---------- | ------- | --------------- | ---------------- | ------------: |
| 1          | Feature | ImageNet        | `fc`             |        `1e-3` |
| 2          | Partial | ImageNet        | `layer4 + fc`    |        `1e-4` |
| 3          | Partial | ImageNet        | `layer4 + fc`    |        `1e-3` |
| 4          | Scratch | Random          | All layers       |        `1e-3` |

Semua eksperimen menggunakan dataset split yang sama sehingga perbandingan performa dilakukan pada data yang konsisten.

---

### 10. Evaluation

Performa model akan dievaluasi menggunakan:

* Validation accuracy
* Test accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Model terbaik dalam setiap eksperimen ditentukan berdasarkan validation accuracy selama training.

Checkpoint dengan validation accuracy terbaik kemudian digunakan untuk melakukan evaluasi pada test set.

Selain performa klasifikasi, latency inference juga diukur untuk model yang dipilih.

---

### 11. Expected Output

Output yang direncanakan dari proyek ini meliputi:

1. Model ResNet-18 untuk klasifikasi sampah.
2. Perbandingan hasil tiga mode training.
3. Perbandingan learning rate pada mode Partial.
4. Grafik training dan validation accuracy.
5. Grafik training dan validation loss.
6. Confusion matrix.
7. Classification report.
8. Tabel perbandingan performa.
9. Pengukuran inference latency.
10. Dokumentasi project pada GitHub.

---

### 12. Project Structure

Struktur project yang direncanakan:

```text
transfer_learning_sampah/
├── dataset_raw/
│   ├── plastic/
│   ├── paper/
│   ├── metal/
│   └── glass/
├── dataset_split/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
├── results/
│   └── plots/
├── src/
│   ├── check_dataset.py
│   ├── create_metadata.py
│   ├── split_dataset.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── plot_results.py
│   ├── evaluate.py
│   ├── measure_latency.py
│   └── create_comparison.py
├── docs/
│   └── initial_design.md
├── metadata.csv
├── README.md
└── .gitignore
```

---

### 13. Reproducibility

Untuk menjaga agar eksperimen dapat dilakukan kembali, project menggunakan:

* Random seed `42`
* Dataset split yang tetap
* Konfigurasi training yang terdokumentasi
* Script terpisah untuk preprocessing, training, evaluation, plotting, dan latency measurement

Dengan struktur tersebut, proses eksperimen dapat dijalankan kembali dari dataset mentah menggunakan script yang tersedia.

---

### 14. Summary

Proyek ini menggunakan ResNet-18 untuk melakukan klasifikasi empat jenis sampah, yaitu plastic, paper, metal, dan glass. Tiga pendekatan training akan dibandingkan, yaitu Feature Extraction, Partial Fine-Tuning, dan Training from Scratch. Pada mode Partial juga dilakukan perbandingan dua learning rate untuk melihat pengaruhnya terhadap performa model.

Hasil eksperimen, evaluasi, grafik, model, dan dokumentasi akan disimpan dalam repository GitHub sebagai bagian dari pengumpulan tugas.

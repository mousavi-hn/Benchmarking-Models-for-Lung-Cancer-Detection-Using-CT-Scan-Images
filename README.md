# Lung Cancer Detection Benchmark using CT Scans (LIDC-IDRI)

## Overview

This project presents a comprehensive benchmarking study of deep learning and hybrid quantum-classical models for lung nodule detection using the publicly available LIDC-IDRI dataset.

The goal is to evaluate and compare multiple architectures in detecting lung nodules from CT scan slices, with a focus on medical reliability, model performance, and future integration with Quantum Machine Learning (QML).

---

## Objectives

* Convert raw 3D CT scans (DICOM) into 2D slices
* Generate labeled datasets (nodule vs. non-nodule)
* Benchmark multiple CNN architectures
* Explore hybrid CNN + Quantum Neural Network (QNN) models
* Analyze model performance using medical-relevant metrics

---

## Dataset

* **Dataset:** LIDC-IDRI
* **Modality:** CT (Computed Tomography)
* **Size:** ~133 GB
* **Format:** DICOM + XML annotations
* **Annotations:** Provided by 4 radiologists per scan

Each scan includes:

* Lung nodules with spatial coordinates
* Nodule characteristics (size, texture, malignancy rating)

---

## Data Preprocessing Pipeline

The dataset is processed using a custom pipeline built with pylidc.

### Steps:

1. Load CT scans (DICOM → 3D volume)
2. Extract nodule annotations
3. Convert 3D volumes into 2D axial slices
4. Label slices:

   * **yes** → contains nodule
   * **no** → no nodule present
5. Normalize Hounsfield Units (HU)
6. Resize images to 224×224
7. Perform patient-wise train/validation/test split
8. Balance dataset (equal yes/no samples)

---

## Dataset Structure

```
processed_lidc_2d/
│
├── dataset/
│   ├── train/
│   │   ├── yes/
│   │   └── no/
│   ├── val/
│   │   ├── yes/
│   │   └── no/
│   └── test/
│       ├── yes/
│       └── no/
│
├── metadata_train.csv
├── metadata_val.csv
└── metadata_test.csv
```

---

## Models

### Classical CNNs

* VGG16 / VGG19
* ResNet50V2
* MobileNetV2
* DenseNet121
* DenseNet201
* EfficientNetB0
* InceptionV3
* Xception

### Hybrid Models (Experimental)

* CNN + Quantum Layer (QNN)
* Implemented using quantum frameworks (PennyLane + JAX)

---

## Evaluation Metrics

Given the medical nature of the task, emphasis is placed on:

* **Accuracy**
* **Precision**
* **Recall (Sensitivity)**
* **Specificity**
* **F1 Score**
* **ROC-AUC**

### ⚠️ Important Note

In medical diagnosis, **False Negatives (FN)** are the most critical error:

> A false negative means the model fails to detect an existing tumor.

Therefore, recall (sensitivity) is prioritized when selecting the best model.

---

## Results

All models are evaluated and compared using:

* Confusion Matrix (TP, TN, FP, FN)
* ROC curves
* Training time
* Model complexity

Results are stored in:

```
results/
```

---

## Explainability

To improve interpretability:

* Grad-CAM is used to visualize model attention
* Highlights suspicious regions in CT slices

---

## Future Work

* Malignancy classification:

  * Benign vs Malignant (based on radiologist scores)
* 3D deep learning models (3D CNNs)
* Improved nodule localization (segmentation)
* Advanced QNN architectures
* Clinical validation with real-world datasets

---

## Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy / Pandas
* pylidc
* Quantum frameworks (Qiskit, PennyLane)

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only**.
It is **not a medical diagnostic tool**.

---

## Acknowledgements

* National Cancer Institute
* The Cancer Imaging Archive
* Contributors of the LIDC-IDRI dataset

---

## 📬 Contact

Hossein Mousavi,
BSc Computer Science Engineering — BME
Email: mousavi.hn@gmail.com
GitHub: https://github.com/mousavi-hn

---

## If you find this project useful

Please consider giving it a star on GitHub ⭐

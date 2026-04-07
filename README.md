# advanced-epilepsy-detection
EEG-driven epilepsy detection platform applying ML/DL for focal vs non-focal classification, tackling data imbalance with SMOTE and enabling reliable, faster medical diagnosis.


# Overview : 

Epilepsy is a chronic neurological disorder characterized by recurring seizures, significantly impacting patients' quality of life. Early and accurate detection of epileptic episodes is essential for timely treatment and improved outcomes.
This project leverages EEG (Electroencephalography) signals, combined with Machine Learning (ML) and Deep Learning (DL) techniques, to automatically classify seizures into focal and non-focal categories.

The system implements advanced signal processing, feature extraction, and classification models to deliver a robust, automated solution for epilepsy detection.

# 🧠 Advanced Epilepsy Detection via EEG Signals

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  <b>An EEG-driven epilepsy detection platform leveraging Machine Learning and Deep Learning for automated focal vs. non-focal seizure classification — tackling class imbalance with SMOTE and enabling reliable, faster medical diagnosis.</b>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Notebooks in Detail](#-notebooks-in-detail)
  - [1. Classical ML Baseline (FINAL\_FAI\_EGG.ipynb)](#1-classical-ml-baseline-final_fai_eggipynb)
  - [2. LSTM Deep Learning Model (FAI\_LSTM\_FINAL.ipynb)](#2-lstm-deep-learning-model-fai_lstm_finalipynb)
  - [3. EEG with Signal Filtering (EGG\_WITH\_FILTER.ipynb)](#3-eeg-with-signal-filtering-egg_with_filteripynb)
- [Dataset](#-dataset)
- [Signal Processing Techniques](#-signal-processing-techniques)
- [Feature Engineering](#-feature-engineering)
- [Models & Algorithms](#-models--algorithms)
- [Handling Class Imbalance with SMOTE](#-handling-class-imbalance-with-smote)
- [Results & Performance](#-results--performance)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Dependencies](#-dependencies)
- [Project Report](#-project-report)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

Epilepsy is one of the most prevalent neurological disorders worldwide, affecting over **50 million people** globally. It is characterized by recurring, unprovoked seizures arising from abnormal electrical activity in the brain. Timely and accurate detection of seizure types is critical for selecting appropriate treatment pathways and improving patient quality of life.

This project presents an **automated epilepsy detection system** built on top of EEG (Electroencephalography) signals. By applying a combination of classical Machine Learning and modern Deep Learning techniques — including LSTM-based sequential models — the system classifies EEG recordings as either **focal** (originating from a specific brain region) or **non-focal** (generalized activity), a distinction vital for surgical candidacy evaluation.

The pipeline covers the complete spectrum from raw EEG signal ingestion, through noise filtering and feature extraction, to model training, evaluation, and comparison.

---

## ❗ Problem Statement

Manual interpretation of EEG signals by neurologists is:
- **Time-consuming** — requires hours of expert analysis per patient.
- **Subjective** — inter-rater variability can affect diagnosis.
- **Resource-intensive** — trained specialists are scarce in many regions.
- **Prone to class imbalance** — focal seizure data is significantly underrepresented.

This project automates that process with a reproducible, scalable ML/DL pipeline that can assist clinicians with rapid, consistent screening.

---

## ✨ Key Features

- 📡 **EEG Signal Preprocessing** — Bandpass and notch filtering to remove noise and powerline interference.
- 📊 **Rich Feature Extraction** — Time-domain, frequency-domain, and statistical features computed from EEG segments.
- ⚖️ **SMOTE Oversampling** — Synthetic Minority Over-sampling Technique to address class imbalance between focal and non-focal samples.
- 🤖 **Classical ML Models** — SVM, Random Forest, KNN, Logistic Regression, Decision Tree evaluated and compared.
- 🧬 **LSTM Deep Learning** — Long Short-Term Memory network to capture temporal dependencies in EEG sequences.
- 📈 **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
- 🔎 **Multi-notebook Design** — Incremental experimentation from baseline ML to deep learning with filters.

---

## 📁 Repository Structure

```
advanced-epilepsy-detection/
│
├── FINAL_FAI_EGG.ipynb          # Classical ML baseline — core feature extraction + model comparison
├── FAI_LSTM_FINAL.ipynb         # LSTM-based deep learning model for EEG time-series classification
├── EGG_WITH_FILTER.ipynb        # Full pipeline with advanced bandpass/notch filtering + ML/DL
├── FAI_22AIE204.pdf             # Detailed project report (methodology, results, analysis)
├── LICENSE                      # MIT License
└── README.md                    # Project documentation (this file)
```

---

## 🏗️ System Architecture & Pipeline

The end-to-end pipeline follows these stages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EEG DATA PIPELINE                               │
│                                                                         │
│  Raw EEG Data                                                           │
│      │                                                                  │
│      ▼                                                                  │
│  ┌──────────────────────┐                                               │
│  │   Signal Preprocessing│  ← Bandpass Filter (0.5–40 Hz)              │
│  │                       │  ← Notch Filter (50/60 Hz powerline)         │
│  │                       │  ← Artifact Removal / Normalization          │
│  └──────────┬────────────┘                                              │
│             │                                                           │
│             ▼                                                           │
│  ┌──────────────────────┐                                               │
│  │   Feature Extraction  │  ← Time-Domain: Mean, Variance, Skewness     │
│  │                       │  ← Freq-Domain: PSD, Spectral Entropy        │
│  │                       │  ← Statistical: Kurtosis, ZCR, Hjorth Params │
│  └──────────┬────────────┘                                              │
│             │                                                           │
│             ▼                                                           │
│  ┌──────────────────────┐                                               │
│  │  Class Imbalance      │  ← SMOTE (Synthetic Minority Oversampling)   │
│  │  Correction           │                                              │
│  └──────────┬────────────┘                                              │
│             │                                                           │
│             ▼                                                           │
│  ┌──────────────────────────────────────────────────┐                  │
│  │              Model Training & Evaluation          │                  │
│  │                                                   │                  │
│  │  Classical ML          │  Deep Learning           │                  │
│  │  ─────────────         │  ──────────────          │                  │
│  │  • SVM                 │  • LSTM Network          │                  │
│  │  • Random Forest       │    (Sequential Model)    │                  │
│  │  • K-Nearest Neighbor  │                          │                  │
│  │  • Logistic Regression │                          │                  │
│  │  • Decision Tree       │                          │                  │
│  └──────────┬─────────────┴──────────────────────────┘                 │
│             │                                                           │
│             ▼                                                           │
│  ┌──────────────────────┐                                               │
│  │   Evaluation Metrics  │  ← Accuracy, Precision, Recall, F1          │
│  │                       │  ← ROC-AUC, Confusion Matrix                │
│  └──────────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📓 Notebooks in Detail

### 1. Classical ML Baseline — `FINAL_FAI_EGG.ipynb`

This notebook establishes the foundational ML pipeline:

- **Data Loading** — Reads EEG signal files and organizes them into focal and non-focal categories.
- **Exploratory Data Analysis (EDA)** — Visualizes raw EEG waveforms, signal distributions, and class balance.
- **Feature Extraction** — Computes statistical and time-domain features from segmented EEG windows.
- **Data Splitting** — Stratified train/test split to maintain class proportions.
- **Model Training** — Trains and evaluates multiple classical ML classifiers.
- **Comparison** — Side-by-side performance metrics across all models.

**Focus:** Establishing a reproducible baseline before applying more complex methods.

---

### 2. LSTM Deep Learning Model — `FAI_LSTM_FINAL.ipynb`

This notebook implements a deep learning approach using Long Short-Term Memory (LSTM) networks:

- **Sequence Preparation** — EEG signals are reshaped into time-step sequences suitable for LSTM input.
- **SMOTE Application** — Addresses class imbalance in the training set before model fitting.
- **LSTM Architecture** — Stacked LSTM layers with dropout regularization to prevent overfitting.
- **Training** — Model trained with binary cross-entropy loss, Adam optimizer, and early stopping.
- **Evaluation** — Classification report, confusion matrix, training/validation loss & accuracy curves.

**Why LSTM?** EEG signals are time-series data with long-range temporal dependencies. LSTM's gating mechanisms make it well-suited for capturing seizure-related patterns that evolve over time.

```
Input Layer (EEG Sequence)
        │
   LSTM Layer 1  (units=128, return_sequences=True)
        │
   Dropout (0.3)
        │
   LSTM Layer 2  (units=64)
        │
   Dropout (0.3)
        │
   Dense Layer   (units=32, activation='relu')
        │
   Output Layer  (units=1, activation='sigmoid')
        │
   Binary Classification: Focal / Non-Focal
```

---

### 3. EEG with Signal Filtering — `EGG_WITH_FILTER.ipynb`

This is the most complete and refined notebook in the repository:

- **Advanced Signal Filtering** — Applies Butterworth bandpass (0.5–40 Hz) and notch filters (50 Hz) to remove muscle artifacts, eye blink noise, and powerline interference before any feature computation.
- **Enhanced Feature Set** — Additional frequency-band features (Delta, Theta, Alpha, Beta, Gamma) extracted post-filtering.
- **Full ML + DL Pipeline** — Runs the complete classical ML suite alongside the LSTM model on clean, filtered signals.
- **Performance Comparison** — Demonstrates the improvement in model accuracy gained from proper signal preprocessing.

**Key Insight:** Signal filtering is shown to significantly improve classification performance by eliminating high-frequency noise that would otherwise corrupt feature calculations.

---

## 📂 Dataset

This project uses EEG recordings from the **Bern-Barcelona EEG database**, a widely used benchmark for focal vs. non-focal epilepsy classification research.

| Property | Details |
|---|---|
| **Signal Type** | Intracranial EEG (iEEG) |
| **Classes** | Focal (F) vs. Non-Focal (NF) |
| **Sampling Rate** | 512 Hz |
| **Segment Length** | ~10,240 samples per segment |
| **Channel Pairs** | X (non-focal region) & Y (focal region) |
| **Format** | `.mat` files (MATLAB format) |

> **Note:** The dataset files are not included in this repository due to size and licensing constraints. Download the Bern-Barcelona dataset from the [official source](http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3) and place the `.mat` files in the appropriate working directory before running the notebooks.

**Class Distribution (before SMOTE):**
```
Focal (Positive)     ████████░░░░░░░░░░░░  ~40%
Non-Focal (Negative) ████████████████████  ~60%
```

---

## 🔧 Signal Processing Techniques

### Bandpass Filtering
Removes frequencies outside the clinically relevant EEG band (0.5 Hz – 40 Hz), eliminating DC drift and high-frequency muscle noise using a **Butterworth IIR filter**.

```python
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=512, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)
```

### Notch Filtering
Eliminates powerline interference at 50 Hz (or 60 Hz in North America):

```python
from scipy.signal import iirnotch

def notch_filter(signal, freq=50.0, fs=512, quality=30):
    b, a = iirnotch(freq / (fs / 2), quality)
    return filtfilt(b, a, signal)
```

### Signal Normalization
Z-score normalization is applied per segment to remove amplitude scale differences between channels and patients.

---

## 🧪 Feature Engineering

Features are extracted from fixed-length EEG windows. The feature set includes:

### Time-Domain Features
| Feature | Description |
|---|---|
| Mean | Average amplitude of the EEG segment |
| Variance | Signal variability / spread |
| Standard Deviation | Dispersion from the mean |
| Skewness | Asymmetry of amplitude distribution |
| Kurtosis | Peakedness / heavy-tail behavior |
| Zero Crossing Rate (ZCR) | Number of times the signal crosses zero |
| Root Mean Square (RMS) | Energy measure of the signal |

### Frequency-Domain Features
| Feature | Description |
|---|---|
| Power Spectral Density (PSD) | Energy distribution across frequency bands |
| Spectral Entropy | Randomness of the power spectrum |
| Delta Band Power (0.5–4 Hz) | Deep sleep / high-amplitude oscillations |
| Theta Band Power (4–8 Hz) | Memory and navigation related |
| Alpha Band Power (8–13 Hz) | Relaxed wakefulness |
| Beta Band Power (13–30 Hz) | Active thinking / motor control |
| Gamma Band Power (30–40 Hz) | High-level cognitive processing |

### Nonlinear / Statistical Features
| Feature | Description |
|---|---|
| Hjorth Mobility | Ratio of signal derivative variance to signal variance |
| Hjorth Complexity | Rate of change of the signal's frequency content |
| Sample Entropy | Regularity and unpredictability measure |
| Petrosian Fractal Dimension | Complexity of the EEG signal |

---

## ⚖️ Handling Class Imbalance with SMOTE

Medical datasets — including epilepsy EEG data — often suffer from severe class imbalance. Training a model on imbalanced data leads to biased classifiers that favor the majority class.

**SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic samples in the feature space of the minority class by interpolating between existing minority samples and their k-nearest neighbors.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

| | Before SMOTE | After SMOTE |
|---|---|---|
| Focal samples | ~40% | 50% |
| Non-Focal samples | ~60% | 50% |
| Effect | Biased toward Non-Focal | Balanced training |

---

## 📊 Models & Algorithms

### Classical Machine Learning

| Model | Key Hyperparameters | Notes |
|---|---|---|
| **Support Vector Machine (SVM)** | `kernel='rbf'`, `C=1.0`, `gamma='scale'` | Strong with high-dimensional feature spaces |
| **Random Forest** | `n_estimators=100`, `max_depth=None` | Ensemble — robust to overfitting |
| **K-Nearest Neighbors (KNN)** | `n_neighbors=5`, `metric='euclidean'` | Distance-based, sensitive to scaling |
| **Logistic Regression** | `C=1.0`, `solver='lbfgs'` | Probabilistic, interpretable baseline |
| **Decision Tree** | `max_depth=10`, `criterion='gini'` | Explainable, prone to overfitting alone |

### Deep Learning — LSTM

| Layer | Configuration |
|---|---|
| Input | `(timesteps, features)` |
| LSTM 1 | 128 units, `return_sequences=True`, tanh activation |
| Dropout 1 | Rate = 0.3 |
| LSTM 2 | 64 units, tanh activation |
| Dropout 2 | Rate = 0.3 |
| Dense | 32 units, ReLU activation |
| Output | 1 unit, Sigmoid activation |
| Optimizer | Adam (`lr=0.001`) |
| Loss | Binary Cross-Entropy |

---

## 📈 Results & Performance

### Classical ML (after SMOTE, on filtered EEG)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| SVM | ~85% | ~84% | ~86% | ~85% | ~0.91 |
| Random Forest | ~87% | ~86% | ~88% | ~87% | ~0.93 |
| KNN | ~80% | ~79% | ~81% | ~80% | ~0.87 |
| Logistic Regression | ~76% | ~75% | ~77% | ~76% | ~0.83 |
| Decision Tree | ~78% | ~77% | ~79% | ~78% | ~0.84 |

### LSTM Deep Learning

| Configuration | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| LSTM (no filter) | ~83% | ~82% | ~0.89 |
| LSTM (with filter + SMOTE) | ~90%+ | ~89%+ | ~0.95+ |

> **Note:** Exact results may vary slightly based on random seed and dataset version. Run the notebooks to reproduce the full experiments.

### Key Observations
- Signal filtering consistently improves all model performances by reducing noise contamination in extracted features.
- SMOTE oversampling is essential — without it, recall on the focal (minority) class drops significantly.
- Random Forest achieves the best trade-off among classical models due to its ensemble nature.
- LSTM with filtered input achieves the highest overall performance, leveraging temporal patterns in EEG.

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Step 1: Clone the Repository

```bash
git clone https://github.com/Akhilesh-Mohanasundaram/advanced-epilepsy-detection.git
cd advanced-epilepsy-detection
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On macOS/Linux
venv\Scripts\activate           # On Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not present, install the core dependencies manually (see [Dependencies](#-dependencies) section below).

### Step 4: Download the Dataset

1. Download the **Bern-Barcelona EEG dataset** from [http://epileptologie-bonn.de](http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3)
2. Place all `.mat` files in a folder, e.g., `./data/`
3. Update the data path variable at the top of each notebook accordingly.

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## 🖥️ Usage

Open the notebooks in the following recommended order for a complete understanding:

**1. Start with the ML Baseline:**
```
FINAL_FAI_EGG.ipynb
```
Run all cells to explore EDA, feature extraction, and compare classical ML models.

**2. Move to the Deep Learning Model:**
```
FAI_LSTM_FINAL.ipynb
```
This notebook introduces SMOTE and trains the LSTM network on EEG sequences.

**3. Run the Full Filtered Pipeline:**
```
EGG_WITH_FILTER.ipynb
```
The most complete experiment — applies signal filters first, then runs the full ML + LSTM pipeline.

**Running a specific notebook cell-by-cell:**
```bash
jupyter nbconvert --to notebook --execute FINAL_FAI_EGG.ipynb --output output.ipynb
```

---

## 📦 Dependencies

The project relies on the following Python libraries:

```txt
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
imbalanced-learn>=0.8.0
tensorflow>=2.6.0
keras>=2.6.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

Install all at once:
```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn imbalanced-learn tensorflow jupyter
```

---

## 📄 Project Report

A comprehensive project report (`FAI_22AIE204.pdf`) is included in the repository. It covers:

- Literature review of EEG-based epilepsy detection approaches
- Detailed methodology — signal preprocessing, feature extraction, model design
- Experimental setup and hyperparameter configurations
- Full results, tables, and comparative analysis
- Conclusions and future directions

To read the report:
```bash
open FAI_22AIE204.pdf        # macOS
xdg-open FAI_22AIE204.pdf   # Linux
start FAI_22AIE204.pdf       # Windows
```

---

## 🔭 Future Work

The following enhancements are planned or proposed for future iterations:

- **Real-time EEG inference** — Integrate with BCI hardware (e.g., OpenBCI) for live seizure monitoring.
- **Convolutional Neural Networks (CNN)** — Apply 1D-CNN or CNN-LSTM hybrid architectures for automatic spatial-temporal feature learning.
- **Attention Mechanisms / Transformers** — Explore EEG Transformer architectures to model long-range dependencies better.
- **Multi-channel EEG** — Extend the pipeline from single-channel to full 19-channel clinical EEG montages.
- **Transfer Learning** — Leverage pre-trained EEG foundation models for low-data scenarios.
- **Explainability (XAI)** — Incorporate SHAP / LIME to make model predictions interpretable for clinicians.
- **Mobile / Edge Deployment** — Convert trained models to TensorFlow Lite for wearable seizure detection devices.
- **Patient-independent evaluation** — Leave-one-subject-out cross-validation for generalization assessment.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add: description of your change"
   ```
4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a Pull Request against the `main` branch.

Please ensure your code is well-commented and, if adding a new model or method, include evaluation results in the PR description.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full details.

```
MIT License — Copyright (c) 2024 Akhilesh Mohanasundaram
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software,
subject to the conditions in the LICENSE file.
```

---

## 🙏 Acknowledgements

- **Bern-Barcelona EEG Dataset** — K. Schindler, H. Leung, C. Elger, K. Lehnertz — for the publicly available intracranial EEG recordings.
- **imbalanced-learn** — Developers of the SMOTE implementation used in this project.
- **scikit-learn** — For the comprehensive ML toolkit.
- **TensorFlow / Keras** — For the deep learning framework powering the LSTM model.

---

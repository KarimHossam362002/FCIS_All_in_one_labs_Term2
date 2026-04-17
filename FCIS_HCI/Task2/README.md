# ECG Signal Identification Project

## Overview
This project performs ECG-based person identification using:
- Fiducial feature extraction (QRS detection)
- Non-fiducial features (DWT + AC + DCT)
- Feature selection
- Classification based on Euclidean distance
- Signal visualization

It identifies whether a test ECG signal belongs to **Ali or Mohamed**.

---

## Features Extracted

### 1. Fiducial Features
- QR interval
- RS interval
- QS slope  
(derived from detected R-peaks using NeuroKit2)

### 2. Non-Fiducial Features
- Discrete Wavelet Transform (DWT)
  - Mean
  - Standard deviation
  - Energy
- Autocorrelation + DCT (first 20 coefficients)

---

## Preprocessing
A **Butterworth bandpass filter (1–40 Hz)** is applied to remove:
- Baseline drift
- High-frequency noise

---

## Project Structure

```
ECG_Project/
│
├── TASK2_ECG.py
├── ECG_Ali.txt
├── ECG_Mohamed.txt
├── Test_signal.txt
├── Feature_Map.xlsx
├── requirements.txt
└── README.md
```

---

## Installation

pip install -r requirements.txt

### OR Manual Installation
pip install numpy pandas scipy matplotlib openpyxl PyWavelets neurokit2 scikit-learn


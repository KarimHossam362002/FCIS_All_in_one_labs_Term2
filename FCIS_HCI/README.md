# EOG Signal Processing and Classification

This repository contains a project for **processing and classifying horizontal EOG (Electrooculography) signals**. The project is structured into preprocessing, feature extraction, and classification steps. It was implemented in **Python** for a Human-Computer Interaction (HCI) lab task.

---

## Project Structure
```bash
Task1_EOG/
│
├─ HorizontalSignals.xlsx # Original horizontal EOG signals (20 signals × 250 samples)
├─ TestSignal.txt # Test signal for classification
├─ TASK1_PreProcessing.py # Preprocessing module (bandpass filter, baseline removal)
├─ TASK1_FeatureExtraction.py # Extracts statistical, morphological, and AR features
├─ TASK1_Classification.py # Classifies test signal using Euclidean distance
├─ FeatureMatrix.xlsx # Generated feature matrix from training signals
└─ README.md # Project documentation
```


---

## Requirements

- Python 3.8+
- Libraries:

```bash
pip install numpy pandas scipy statsmodels matplotlib
```

## Preprocessing

- The preprocessing step applies:

- Bandpass filter: 0.1–10 Hz to remove noise.

- Baseline removal: subtracts mean to center the signal around zero.

- All preprocessing functions are in TASK1_PreProcessing.py.

## Feature Extraction

For each signal, the following features are computed:

## Statistical Features

Mean

Standard deviation (Std)

Morphological Features

Maximum peak amplitude

Area under the curve (AUC)

Auto-regression Features

AR(1), AR(2), AR(3) coefficients

The features are stored in a matrix Excel file FeatureMatrix.xlsx with the following structure:

| Signal | Mean | Std | MaxPeak | AUC | AR1 | AR2 | AR3 |

## Classification


The test signal (TestSignal.txt) is classified using Euclidean distance between its feature vector and the feature vectors of the labeled signals.

The signal with the minimum Euclidean distance is assigned as the predicted label.

Example output:

```bash
Test signal classified as: Juri
Minimum Euclidean distance: 0.000000000000005652
```

## Usage

1- Preprocess and extract features:

```bash
python TASK1_FeatureExtraction.py
```
2- Classify a test signal:

```bash
python TASK1_Classification.py
```
Notes

- All numeric features are saved in full decimal format to avoid scientific notation.

- The sampling rate of the signals is 176 Hz, and each signal contains 250 samples.

- The pipeline can be adapted for additional EOG signals or other HCI-related tasks.
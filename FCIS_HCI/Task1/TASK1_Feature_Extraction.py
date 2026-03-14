import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from statsmodels.tsa.ar_model import AutoReg

from TASK1_PreProcessing import preprocess_signal


# -------------------------
# Load signals
# -------------------------
file_path = "HorizontalSignals.xlsx"
df = pd.read_excel(file_path)

feature_matrix = []

# Number of decimal places for small values
DECIMALS = 18

for column in df.columns:

    signal = df[column].values

    # -------------------------
    # Preprocess signal
    # -------------------------
    signal = preprocess_signal(signal)

    # -------------------------
    # Statistical Features
    # -------------------------
    mean_val = np.mean(signal)
    std_val = np.std(signal)

    # -------------------------
    # Morphological Features
    # -------------------------
    max_peak = np.max(signal)
    auc = np.trapezoid(signal)

    # -------------------------
    # Auto-regression Features
    # -------------------------
    model = AutoReg(signal, lags=3).fit()
    ar_coeffs = model.params[1:]  # ignore intercept

    # -------------------------
    # Combine features (formatted)
    # -------------------------
    features = [
        column,
        float(f"{mean_val:.{DECIMALS}f}"),
        float(f"{std_val:.{DECIMALS}f}"),
        float(f"{max_peak:.{DECIMALS}f}"),
        float(f"{auc:.{DECIMALS}f}"),
        float(f"{ar_coeffs[0]:.{DECIMALS}f}"),
        float(f"{ar_coeffs[1]:.{DECIMALS}f}"),
        float(f"{ar_coeffs[2]:.{DECIMALS}f}")
    ]

    feature_matrix.append(features)

# -------------------------
# Create Feature Matrix
# -------------------------
columns = [
    "Signal",
    "Mean",
    "Std",
    "MaxPeak",
    "AUC",
    "AR1",
    "AR2",
    "AR3"
]

features_df = pd.DataFrame(feature_matrix, columns=columns)

# -------------------------
# Save to Excel
# -------------------------
features_df.to_excel("FeatureMatrix.xlsx", index=False, float_format="%.18f")

print("Feature matrix saved successfully.")

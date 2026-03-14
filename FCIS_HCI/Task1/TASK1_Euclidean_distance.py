import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from statsmodels.tsa.ar_model import AutoReg

from TASK1_PreProcessing import preprocess_signal


# -------------------------
# Load Feature Matrix
# -------------------------
features_df = pd.read_excel("FeatureMatrix.xlsx")

signal_names = features_df["Signal"].values
train_features = features_df.drop(columns=["Signal"]).values


# -------------------------
# Load Test Signal
# -------------------------
test_signal = np.loadtxt("TestSignal.txt")


# -------------------------
# Preprocess Test Signal
# -------------------------
test_signal = preprocess_signal(test_signal)


# -------------------------
# Extract Features from Test Signal
# -------------------------

# Statistical
mean_val = np.mean(test_signal)
std_val = np.std(test_signal)

# Morphological
max_peak = np.max(test_signal)
auc = np.trapezoid(test_signal)

# Auto-regression
model = AutoReg(test_signal, lags=3).fit()
ar_coeffs = model.params[1:]

test_features = np.array([
    mean_val,
    std_val,
    max_peak,
    auc,
    ar_coeffs[0],
    ar_coeffs[1],
    ar_coeffs[2]
])


# -------------------------
# Euclidean Distance
# -------------------------
distances = []

for train_vec in train_features:
    d = euclidean(test_features, train_vec)
    distances.append(d)

distances = np.array(distances)

# -------------------------
# Find Closest Signal
# -------------------------
min_index = np.argmin(distances)
predicted_signal = signal_names[min_index]

print("Test signal classified as:", predicted_signal)
print("Minimum Euclidean distance:", format(distances[min_index], ".18f"))

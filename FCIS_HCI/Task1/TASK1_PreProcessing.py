import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt


def butter_bandpass_filter(Input_Signal,low_cutoff, high_cutoff, Sampling_Rate, order):
    nyq = 0.5 * Sampling_Rate # Nyquist Sampling
    low = low_cutoff / nyq
    high = high_cutoff /nyq

    Numerator,denominator = butter(order,[low,high],'band',output='ba',analog=False,fs=None)

    # PASSING THE 1ST COLUMN OF DATA SHAPE (251,) INSTEAD OF (251,1)
    filtered = filtfilt(Numerator,denominator,Input_Signal)

    return filtered

def preprocess_signal(signal, fs=176):

    # 1 Bandpass filter
    filtered = butter_bandpass_filter(signal, 0.1, 10, fs, 4)

    # 2 Baseline removal
    filtered = filtered - np.mean(filtered)

    return filtered

# -----------------------------
# Parameters
# -----------------------------
fs = 176
low_cut = 0.1
high_cut = 10
order = 4

# -----------------------------
# Read Excel File
# -----------------------------
file_path = "HorizontalSignals.xlsx"

df = pd.read_excel(file_path)

# Remove any non-numeric rows if they exist
# df = df.apply(pd.to_numeric, errors='coerce')
# df = df.dropna() # DROP STRING COLUMNS

# -----------------------------
# Filter Each Signal
# -----------------------------
filtered_df = pd.DataFrame()

for column in df.columns:
    signal = df[column].values
    filtered_signal = butter_bandpass_filter(signal, low_cut, high_cut, fs, order)
    filtered_df[column] = filtered_signal

# -----------------------------
# Plot Signals in Grid
# -----------------------------
rows = 5
cols = 4

fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
axes = axes.flatten()

for i, column in enumerate(filtered_df.columns):
    axes[i].plot(filtered_df[column])
    axes[i].set_title(column)
    axes[i].grid(True)

# plt.tight_layout()#
plt.show()
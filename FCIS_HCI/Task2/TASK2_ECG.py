import numpy as np
import pandas as pd
import scipy.fftpack as fft
import pywt
import neurokit2 as nk
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics.pairwise import euclidean_distances

# =====================================================
# SETTINGS
# =====================================================
fs = 250   # Sampling Frequency

# =====================================================
# BANDPASS FILTER (1 Hz - 40 Hz)
# =====================================================
def bandpass_filter(ecg, fs, low=1, high=40, order=4):
    nyquist = 0.5 * fs

    low_cut = low / nyquist
    high_cut = high / nyquist

    b, a = butter(order, [low_cut, high_cut], btype='band')
    filtered = filtfilt(b, a, ecg)

    return filtered

# =====================================================
# LOAD ECG FILES
# =====================================================
def load_ecg(filename):
    data = np.loadtxt(filename)

    # If file has 2 columns -> use second column
    if data.ndim > 1:
        data = data[:, 1]

    data = data.flatten()

    # Apply bandpass filter
    data = bandpass_filter(data, fs)

    return data

ali = load_ecg("ECG_Ali.txt")
mohamed = load_ecg("ECG_Mohamed.txt")
test = load_ecg("Test_signal.txt")

print("Ali Shape:", ali.shape)
print("Mohamed Shape:", mohamed.shape)
print("Test Shape:", test.shape)

# =====================================================
# FIDUCIAL FEATURES
# QRS Detection + QR interval + RS interval + QS slope
# =====================================================
def fiducial_features(ecg, fs):
    try:
        signals, info = nk.ecg_process(ecg, sampling_rate=fs)
        r_peaks = info["ECG_R_Peaks"]

        features = []

        for r in r_peaks:
            q = max(0, r - 10)
            s = min(len(ecg) - 1, r + 10)

            qr_interval = (r - q) / fs
            rs_interval = (s - r) / fs
            qs_slope = (ecg[s] - ecg[q]) / ((s - q) / fs)

            features.append([qr_interval, rs_interval, qs_slope])

        if len(features) == 0:
            return np.zeros(3)

        return np.mean(features, axis=0)

    except:
        return np.zeros(3)

# =====================================================
# NON-FIDUCIAL FEATURES
# DWT
# =====================================================
def dwt_features(ecg):
    coeffs = pywt.wavedec(ecg, 'db4', level=4)

    feats = []

    for c in coeffs:
        feats.append(np.mean(c))
        feats.append(np.std(c))
        feats.append(np.sum(c ** 2))   # Energy

    return np.array(feats)

# =====================================================
# NON-FIDUCIAL FEATURES
# AC + DCT
# =====================================================
def ac_dct_features(ecg):
    ac = np.correlate(ecg, ecg, mode='full')
    ac = ac[len(ac)//2:]

    dct = fft.dct(ac, norm='ortho')

    return dct[:20]

# =====================================================
# COMBINE ALL FEATURES
# =====================================================
def extract_all(ecg):
    f1 = fiducial_features(ecg, fs)
    f2 = dwt_features(ecg)
    f3 = ac_dct_features(ecg)

    return np.concatenate([f1, f2, f3])

# =====================================================
# FEATURE EXTRACTION
# =====================================================
ali_feat = extract_all(ali)
mohamed_feat = extract_all(mohamed)
test_feat = extract_all(test)

X = np.array([
    ali_feat,
    mohamed_feat
])

y = np.array([0, 1])   # 0 = Ali, 1 = Mohamed

# =====================================================
# FEATURE SELECTION
# =====================================================
k = min(10, X.shape[1])

selector = SelectKBest(score_func=f_classif, k=k)
X_new = selector.fit_transform(X, y)

test_new = selector.transform(test_feat.reshape(1, -1))

# =====================================================
# IDENTIFICATION
# =====================================================
dist = euclidean_distances(test_new, X_new)

idx = np.argmin(dist)

person = "Ali" if idx == 0 else "Mohamed"

print("\nTest Signal belongs to:", person)

# =====================================================
# SAVE FEATURE MAP TO EXCEL
# =====================================================
df = pd.DataFrame(X)
df["Person"] = ["Ali", "Mohamed"]

selected_indices = selector.get_support(indices=True)
selected_df = pd.DataFrame({
    "Selected Feature Index": selected_indices
})

with pd.ExcelWriter("Feature_Map.xlsx") as writer:
    df.to_excel(writer, sheet_name="All Features", index=False)
    selected_df.to_excel(writer, sheet_name="Best Features", index=False)

print("Feature_Map.xlsx saved successfully.")

# =====================================================
# PLOTS (ALL IN ONE WINDOW)
# =====================================================
fig, axs = plt.subplots(4, 1, figsize=(14, 12))

# -------- Ali Signal --------
axs[0].plot(ali)
axs[0].set_title("Filtered ECG Signal - Ali")
axs[0].set_xlabel("Samples")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

# -------- Mohamed Signal --------
axs[1].plot(mohamed)
axs[1].set_title("Filtered ECG Signal - Mohamed")
axs[1].set_xlabel("Samples")
axs[1].set_ylabel("Amplitude")
axs[1].grid(True)

# -------- Test Signal --------
axs[2].plot(test)
axs[2].set_title("Filtered ECG Signal - Test")
axs[2].set_xlabel("Samples")
axs[2].set_ylabel("Amplitude")
axs[2].grid(True)

# -------- Feature Map --------
axs[3].plot(ali_feat, marker='o', label="Ali")
axs[3].plot(mohamed_feat, marker='s', label="Mohamed")
axs[3].plot(test_feat, marker='x', label="Test")

axs[3].set_title("Feature Map Comparison")
axs[3].set_xlabel("Feature Index")
axs[3].set_ylabel("Feature Value")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()
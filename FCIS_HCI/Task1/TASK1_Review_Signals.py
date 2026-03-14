import pandas as pd
import matplotlib.pyplot as plt

file_path = "HorizontalSignals.xlsx"

df = pd.read_excel(file_path)

# Convert values to numeric (strings become NaN)
# df = df.apply(pd.to_numeric, errors='coerce')

# Remove rows that contain NaN (like the names row)
# df = df.dropna()

signals = df.columns

rows = 5
cols = 4

fig, axes = plt.subplots(rows, cols, figsize=(15,12))
axes = axes.flatten()

for i, signal in enumerate(signals):
    axes[i].plot(df[signal])
    axes[i].set_title(signal)
    axes[i].grid(True)

plt.tight_layout()
plt.show()
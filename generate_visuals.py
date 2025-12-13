import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical, plot_model

# --- Configuration ---
WINDOW_SIZE = 75  # 75 samples = 1 second at 75 Hz
DATA_FILES = {
    '0_blink_data.csv': 0,
    '1_blink_data.csv': 1,
    '2_blink_data.csv': 2
}
CLASS_NAMES = ['No Blink', '1 Blink', '2 Blinks']

# --- Check for necessary files ---
print("Checking for required files...")
required_files = list(DATA_FILES.keys()) + ['blink_model.h5', 'scaler.pkl']
for f in required_files:
    if not os.path.exists(f):
        print(f"Error: Required file '{f}' not found. Please place it in the same folder as this script.")
        exit()
print("All required files found.")

# --- 1. Load Data and Create Windows (Same as training) ---
all_X, all_y = [], []
for filename, label in DATA_FILES.items():
    df = pd.read_csv(filename)
    df['signal'] = pd.to_numeric(df['signal'], errors='coerce')
    df.dropna(subset=['signal'], inplace=True)
    signal_values = df['signal'].values
    for i in range(0, len(signal_values) - WINDOW_SIZE, WINDOW_SIZE):
        all_X.append(signal_values[i : i + WINDOW_SIZE])
        all_y.append(label)

X = np.array(all_X)
y = np.array(all_y)

# --- 2. Load the Scaler and Model ---
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
model = load_model('blink_model.h5')

# --- 3. Prepare Data for Evaluation ---
X_reshaped = X.reshape(-1, 1)
X_scaled = scaler.transform(X_reshaped)
X_final = X_scaled.reshape(X.shape[0], X.shape[1], 1)

_, X_val, _, y_val = train_test_split(X_final, y, test_size=0.2, random_state=42)

# --- 4. Generate and Save Confusion Matrix (Counts) ---
print("\nGenerating Confusion Matrix (Counts)...")
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Model Performance: Confusion Matrix (Counts)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_counts.png')
print("Saved 'confusion_matrix_counts.png'")

# --- 5. NEW: Generate and Save Confusion Matrix (Percentages) ---
print("\nGenerating Confusion Matrix (Percentages)...")
# Calculate the percentage matrix by dividing each row by its sum
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
# Use fmt='.2%' to format the numbers as percentages
sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Model Performance: Confusion Matrix (Percentages)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_percentage.png')
print("Saved 'confusion_matrix_percentage.png'")

# Print a classification report for more details
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=CLASS_NAMES))


# --- 6. Generate and Save Sample EOG Signal Plots ---
print("\nGenerating sample signal plots...")
fig, axs = plt.subplots(len(CLASS_NAMES), 3, figsize=(15, 10), sharex=True, sharey=True)
fig.suptitle('Sample EOG Signal Windows for Each Class', fontsize=16)

for label_index, label_name in enumerate(CLASS_NAMES):
    class_indices = np.where(y == label_index)[0]
    random_indices = np.random.choice(class_indices, 3, replace=False)
    
    for i, data_index in enumerate(random_indices):
        axs[label_index, i].plot(X[data_index])
        axs[label_index, i].set_title(f'{label_name} - Sample {i+1}')
        axs[label_index, i].grid(True, linestyle='--', alpha=0.6)

plt.ylim(-2.5, 2.5)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('eog_sample_windows.png')
print("Saved 'eog_sample_windows.png'")


# --- 7. Generate and Save Model Architecture Diagram ---
print("\nGenerating model architecture diagram...")
try:
    plot_model(
        model,
        to_file='model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True
    )
    print("Saved 'model_architecture.png'")
except ImportError:
    print("\nCould not generate model architecture diagram.")
    print("Please install pydot and graphviz. In your terminal, run:")
    print("pip install pydot")
    print("You may also need to install Graphviz system-wide: https://graphviz.org/download/")

print("\n--- All visuals have been generated successfully! ---")
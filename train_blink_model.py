import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration ---
WINDOW_SIZE = 75  # 75 samples = 1 second at 75 Hz
DATA_FILES = {
    '0_blink_data.csv': 0,
    '1_blink_data.csv': 1,
    '2_blink_data.csv': 2
}

# --- 1. Load Data and Create Sequential Windows ---
all_X = []
all_y = []

print("Loading data and creating windows...")
for filename, label in DATA_FILES.items():
    print(f"Processing {filename}...")
    df = pd.read_csv(filename)
    
    # Clean the data just in case
    df['signal'] = pd.to_numeric(df['signal'], errors='coerce')
    df.dropna(subset=['signal'], inplace=True)
    
    signal_values = df['signal'].values
    
    # Create windows from this single, sequential file
    for i in range(0, len(signal_values) - WINDOW_SIZE, WINDOW_SIZE):
        window = signal_values[i : i + WINDOW_SIZE]
        all_X.append(window)
        all_y.append(label)

# --- 2. Prepare Data for Training ---
# Convert to NumPy arrays
X = np.array(all_X)
y = np.array(all_y)

# Shuffle the windows and labels together
X, y = shuffle(X, y, random_state=42)

# Scale the data
scaler = StandardScaler()
X_reshaped = X.reshape(-1, 1)
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(X.shape[0], X.shape[1], 1)

# Convert labels to one-hot encoding
y_categorical = to_categorical(y)

print(f"\nData preparation complete. Total windows: {len(X)}")

# --- 3. Build and Train the CNN Model ---
print("Building and training the model...")
model = Sequential([
    Conv1D(8, 3, activation='relu', input_shape=(WINDOW_SIZE, 1)),
    MaxPooling1D(2),
    Conv1D(16, 3, activation='relu'),
    Flatten(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax') # Use shape[1] for number of classes
])

early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    X, y_categorical,
    epochs=200,
    batch_size=16,
    validation_split=0.2, # Use 20% of data for validation
    callbacks=[early_stopping]
)

# --- 4. Save the Model and Scaler ---
print("\nTraining complete. Saving model and scaler.")
model.save('blink_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Process finished successfully!")


# trained the model for 50 epochs using the Adam optimizer, and the accuracy improved steadily from 35% to around 78%.
# The validation accuracy also remained consistent, indicating good generalization.




# Data preparation complete. Total windows: 344
# Building and training the model...
# 2025-10-17 01:53:11.732173: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Epoch 1/50
# 18/18 [==============================] - 1s 13ms/step - loss: 1.1059 - accuracy: 0.3527 - val_loss: 1.0366 - val_accuracy: 0.4203
# Epoch 2/50
# 18/18 [==============================] - 0s 4ms/step - loss: 1.0524 - accuracy: 0.4182 - val_loss: 0.9915 - val_accuracy: 0.4348
# Epoch 3/50
# 18/18 [==============================] - 0s 4ms/step - loss: 0.9483 - accuracy: 0.5491 - val_loss: 0.9149 - val_accuracy: 0.5072
# Epoch 4/50
# 18/18 [==============================] - 0s 4ms/step - loss: 0.8806 - accuracy: 0.5964 - val_loss: 0.8142 - val_accuracy: 0.5652
# Epoch 5/50
# 18/18 [==============================] - 0s 4ms/step - loss: 0.7832 - accuracy: 0.6764 - val_loss: 0.7694 - val_accuracy: 0.6232
# Epoch 6/50
# 18/18 [==============================] - 0s 4ms/step - loss: 0.7051 - accuracy: 0.6582 - val_loss: 0.6934 - val_accuracy: 0.6667
# Epoch 7/50
# 18/18 [==============================] - 0s 4ms/step - loss: 0.6359 - accuracy: 0.7164 - val_loss: 0.6572 - val_accuracy: 0.6522
# Epoch 8/50
# 18/18 [==============================] - 0s 4ms/step - loss: 0.5769 - accuracy: 0.7273 - val_loss: 0.6130 - val_accuracy: 0.6812
# Epoch 9/50
# 18/18 [==============================] - 0s 4ms/step - loss: 0.5147 - accuracy: 0.7745 - val_loss: 0.6018 - val_accuracy: 0.6957
# Epoch 10/50
# 18/18 [==============================] - 0s 4ms/step - loss: 0.5006 - accuracy: 0.7782 - val_loss: 0.5738 - val_accuracy: 0.6957
# Epoch 11/50
# 18/18 [==============================] - 0s 4ms/step - loss: 0.4960 - accuracy: 0.7673 - val_loss: 0.5504 - val_accuracy: 0.7246
# Epoch 12/50
# 18/18 [==============================] - 0s 4ms/step - loss: 0.4548 - accuracy: 0.7891 - val_loss: 0.5635 - val_accuracy: 0.7101
# Epoch 13/50
# 18/18 [==============================] - 0s 4ms/step - loss: 0.4402 - accuracy: 0.7745 - val_loss: 0.5686 - val_accuracy: 0.7681
# Epoch 14/50
# 18/18 [==============================] - 0s 4ms/step - loss: 0.4393 - accuracy: 0.7818 - val_loss: 0.5580 - val_accuracy: 0.7101
# Epoch 15/50
# 18/18 [==============================] - 0s 3ms/step - loss: 0.4321 - accuracy: 0.7673 - val_loss: 0.5582 - val_accuracy: 0.7681
# Epoch 16/50
# 18/18 [==============================] - 0s 4ms/step - loss: 0.4364 - accuracy: 0.8036 - val_loss: 0.5602 - val_accuracy: 0.7391

# Training complete. Saving model and scaler.
# D:\Sara_folder\finalyear\deeplearning\venv\Lib\site-packages\keras\src\engine\training.py:3000: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
#   saving_api.save_model(
# Process finished successfully!
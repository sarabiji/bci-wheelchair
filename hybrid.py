import serial
import time
import numpy as np
import pickle
import pandas as pd
from scipy import signal
from collections import deque
from tensorflow.keras.models import load_model


EEG_PORT = '/dev/ttyUSB0'
EOG_PORT = '/dev/ttyUSB1'
ARDUINO_PORT = '/dev/ttyACM0'
BAUD = 115200

# LOAD MODELS

print("[INFO] Loading models...")
svm = pickle.load(open("model.pkl", "rb"))
scaler_eeg = pickle.load(open("scaler.pkl", "rb"))

blink_model = load_model("blink_model.h5")
scaler_eog = pickle.load(open("scaler.pkl", "rb"))

print("[INFO] Models loaded.")

# =========================
# EEG FILTER SETUP
# =========================
FS = 512
b_notch, a_notch = signal.iirnotch(50/(0.5*FS), 30)
b_band, a_band = signal.butter(4, [0.5/(0.5*FS), 30/(0.5*FS)], 'band')

def process_eeg(x):
    x = signal.filtfilt(b_notch, a_notch, x)
    x = signal.filtfilt(b_band, a_band, x)
    return x

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(x):
    f, psd = signal.welch(x, FS, nperseg=len(x))
    bands = {'alpha':(8,13),'beta':(14,30)}
    features = {}
    for b,(l,h) in bands.items():
        idx = np.where((f>=l)&(f<=h))
        features[f'E_{b}'] = np.sum(psd[idx])
    features['alpha_beta_ratio'] = features['E_alpha']/features['E_beta'] if features['E_beta']>0 else 0
    return features

# =========================
# SERIAL SETUP
# =========================
eeg_ser = serial.Serial(EEG_PORT, BAUD, timeout=1)
eog_ser = serial.Serial(EOG_PORT, BAUD, timeout=1)
arduino_ser = serial.Serial(ARDUINO_PORT, BAUD, timeout=1)

# =========================
# BUFFERS
# =========================
eeg_buffer = deque(maxlen=512)
eog_buffer = deque(maxlen=75)

eeg_move_allowed = False
eog_direction = None  # 'L', 'R', None

print("[INFO] System running...")


while True:
    try:
        # -------- EEG PROCESSING --------
        eeg_line = eeg_ser.readline().decode(errors='ignore').strip()
        if eeg_line:
            eeg_buffer.append(float(eeg_line))

        if len(eeg_buffer) == 512:
            eeg_data = process_eeg(np.array(eeg_buffer))
            feats = extract_features(eeg_data)
            X = scaler_eeg.transform(pd.DataFrame([feats]))
            eeg_pred = svm.predict(X)[0]
            eeg_move_allowed = (eeg_pred == 0)  # 0 = MOVE, 1 = STOP
            eeg_buffer.clear()

        # -------- EOG PROCESSING --------
        eog_line = eog_ser.readline().decode(errors='ignore').strip()
        if ',' in eog_line:
            signal_val, _ = map(float, eog_line.split(','))
            eog_buffer.append(signal_val)

        if len(eog_buffer) == 75:
            window = np.array(eog_buffer).reshape(-1,1)
            window_scaled = scaler_eog.transform(window)
            window_final = window_scaled.reshape(1,75,1)

            probs = blink_model.predict(window_final, verbose=0)[0]
            pred = np.argmax(probs)
            conf = np.max(probs)

            if conf > 0.9:
                if pred == 1:
                    eog_direction = 'L'
                elif pred == 2:
                    eog_direction = 'R'
                else:
                    eog_direction = None

            eog_buffer.clear()

        # -------- DECISION FUSION --------
        if not eeg_move_allowed:
            cmd = 'S'
            print("[INFO] Command: STOP")
        else:
            if eog_direction == 'L':
                cmd = 'L'
                print("[INFO] Command: LEFT")
            elif eog_direction == 'R':
                cmd = 'R'
                print("[INFO] Command: RIGHT")
            else:
                cmd = 'F'
                print("[INFO] Command: FORWARD")

        arduino_ser.write(cmd.encode())
        time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping system.")
        break

    except Exception as e:
        print("[ERROR]", e)

arduino_ser.close()

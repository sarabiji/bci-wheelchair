import serial
import time
import numpy as np
import pickle
from collections import deque
from scipy import signal
from tensorflow.keras.models import load_model

# ===== SERIAL PORTS =====
SENSOR_PORT = '/dev/ttyUSB0'   # Arduino 1 (EEG + EOG)
MOTOR_PORT  = '/dev/ttyUSB1'   # Arduino 2 (Motors)

sensor_ser = serial.Serial(SENSOR_PORT, 115200, timeout=0)
motor_ser  = serial.Serial(MOTOR_PORT, 9600, timeout=0)
time.sleep(2)

# ===== LOAD MODELS =====
with open('eegmodel.pkl', 'rb') as f:
    eeg_model = pickle.load(f)
with open('eegscaler.pkl', 'rb') as f:
    eeg_scaler = pickle.load(f)

eog_model = load_model('eogmodel.h5')
with open('eogscaler.pkl', 'rb') as f:
    eog_scaler = pickle.load(f)

# ===== BUFFERS =====
EEG_FS = 512
EOG_FS = 75

eeg_buffer = deque(maxlen=EEG_FS)
eog_buffer = deque(maxlen=EOG_FS)

motion_state = 'S'   # EEG → F / S
turn_state = None    # EOG → L / R
last_sent = None

# ===== FILTERS =====
b_notch, a_notch = signal.iirnotch(50 / (0.5 * EEG_FS), 30)
b_band, a_band = signal.butter(4, [0.5 / (0.5 * EEG_FS), 30 / (0.5 * EEG_FS)], 'band')

# ===== MOTOR SEND =====
def send(cmd):
    global last_sent
    if cmd != last_sent:
        motor_ser.write(cmd.encode())
        last_sent = cmd
        print("Sent →", cmd)

# ===== EEG PROCESS =====
def process_eeg():
    global motion_state

    if len(eeg_buffer) < EEG_FS:
        return

    data = np.array(eeg_buffer)
    data = signal.filtfilt(b_notch, a_notch, data)
    data = signal.filtfilt(b_band, a_band, data)

    f, psd = signal.welch(data, fs=EEG_FS)
    alpha = np.sum(psd[(f >= 8) & (f <= 13)])
    beta  = np.sum(psd[(f >= 14) & (f <= 30)])
    ratio = alpha / beta if beta > 0 else 0

    X = eeg_scaler.transform([[alpha, beta, ratio]])
    pred = eeg_model.predict(X)[0]

    motion_state = 'F' if pred == 1 else 'S'
    eeg_buffer.clear()

# ===== EOG PROCESS =====
def process_eog():
    global turn_state

    if len(eog_buffer) < EOG_FS:
        return

    window = np.array(eog_buffer).reshape(-1, 1)
    window = eog_scaler.transform(window)
    window = window.reshape(1, EOG_FS, 1)

    probs = eog_model.predict(window, verbose=0)[0]
    pred = np.argmax(probs)
    conf = np.max(probs)

    if conf > 0.9:
        if pred == 1:
            turn_state = 'L'
        elif pred == 2:
            turn_state = 'R'

        eog_buffer.clear()

# ===== MAIN LOOP =====
while True:
    if sensor_ser.in_waiting:
        line = sensor_ser.readline().decode(errors='ignore').strip()

        if line.startswith("EEG"):
            value = int(line.split(',')[1])
            eeg_buffer.append(value)
            process_eeg()

        elif line.startswith("EOG"):
            value = int(line.split(',')[1])
            eog_buffer.append(value)
            process_eog()

    # ===== COMMAND PRIORITY =====
    if turn_state:
        send(turn_state)
        turn_state = None
    else:
        send(motion_state)
        print("Current Motion State:", motion_state)

    time.sleep(0.01)


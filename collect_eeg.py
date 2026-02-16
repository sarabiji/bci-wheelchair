import serial
import csv
import time
import os

# ===== CONFIG =====
PORT = '/dev/ttyACM0'
BAUD = 115200
OUTPUT_FILE = 'eeg_data.csv'
MAX_DURATION = 60  # seconds (change per experiment)

# ==================

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

file_exists = os.path.isfile(OUTPUT_FILE)

with open(OUTPUT_FILE, 'a', newline='') as f:
    writer = csv.writer(f)

    # Write header only if new file
    if not file_exists:
        writer.writerow(['time_sec', 'value'])

    print("Recording EEG... Press CTRL+C to stop")

    start_time = time.time()

    try:
        while time.time() - start_time < MAX_DURATION:

            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if line.startswith("EEG") and ',' in line:
                try:
                    _, value = line.split(',')
                    value = float(value)

                    elapsed = time.time() - start_time

                    writer.writerow([elapsed, value])
                    print(f"{elapsed:.3f} | EEG | {value}")

                except:
                    continue

    except KeyboardInterrupt:
        print("\nStopped by user.")

ser.close()
print("EEG recording finished.")

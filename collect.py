import serial
import csv
import time
import os

COM_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
OUTPUT_FILE = 'client1_signal.csv'
MAX_DURATION = 300  # seconds

ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

file_exists = os.path.isfile(OUTPUT_FILE)

with open(OUTPUT_FILE, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write header only if file is new
    if not file_exists:
        writer.writerow(["elapsed_time", "signal_type", "value"])

    print("Collecting data... Press CTRL+C to stop")

    start_time = time.time()

    try:
        while time.time() - start_time < MAX_DURATION:

            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                continue

            if ',' not in line:
                continue

            try:
                signal_type, value = line.split(',')
                value = float(value)

                elapsed_time = time.time() - start_time

                writer.writerow([elapsed_time, signal_type, value])
                print(f"{elapsed_time:.3f} | {signal_type} | {value}")

            except:
                continue

    except KeyboardInterrupt:
        print("\nStopped by user.")

ser.close()
print("Finished recording.")

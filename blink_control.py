import serial
import time
import numpy as np
import pickle
from tensorflow.keras.models import load_model
# import RPi.GPIO as GPIO # Keep commented out for Windows testing

# --- 1. Setup ---
SERIAL_PORT = 'COM8'
BAUD_RATE = 115200
WINDOW_SIZE = 75
# GPIO.setmode(GPIO.BCM)      # For Raspberry Pi
# MOTOR_PIN = 17
# GPIO.setup(MOTOR_PIN, GPIO.OUT) # For Raspberry Pi

# --- 2. Load the trained model and the scaler ---
print("Loading model and scaler...")
model = load_model('blink_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Model and scaler loaded successfully.")

ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
buffer = []
cooldown_until = 0 # Timestamp for non-blocking cooldown

def control_motor(state):
    print(f"--- MOTOR STATE WOULD BE: {'ON' if state else 'OFF'} ---")
    # GPIO.output(MOTOR_PIN, GPIO.HIGH if state else GPIO.LOW) # For Raspberry Pi
    pass

motor_state = False

print("\nStarting real-time blink detection...")

# --- 3. Main Loop ---
while True:
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if ',' in line:
            signal, peak = map(float, line.split(','))
            buffer.append(signal)

            if len(buffer) > WINDOW_SIZE:
                buffer.pop(0)

            # Only predict if we have a full window
            if len(buffer) == WINDOW_SIZE:
                
                # --- This is the new, improved logic ---
                window_for_scaler = np.array(buffer).reshape(-1, 1)
                window_scaled = scaler.transform(window_for_scaler)
                window_final = window_scaled.reshape(1, WINDOW_SIZE, 1)
                
                prediction_probabilities = model.predict(window_final, verbose=0)[0]
                prediction = np.argmax(prediction_probabilities)
                confidence = np.max(prediction_probabilities)
                signal_max_value = np.max(buffer)
                
                # --- Only act if the model is confident AND the prediction is NOT "no blink" ---
                if prediction != 0 and confidence > 0.90 and signal_max_value > 0.3: # Using a 90% confidence threshold
                    if prediction == 1:  # single blink
                        motor_state = not motor_state
                        control_motor(motor_state)
                        print(f"Prediction: SINGLE BLINK (Conf: {confidence:.2f}) -> Toggling motor to {motor_state}")
                        
                    elif prediction == 2:  # double blink
                        motor_state = False
                        control_motor(False)
                        print(f"Prediction: DOUBLE BLINK (Conf: {confidence:.2f}) -> Stopping motor")
                    
                    # --- CRITICAL FIX: Clear the buffer and wait ---
                    print("--- Action taken. Clearing buffer and entering cooldown. ---")
                    buffer.clear() # Empty the buffer to wait for fresh data
                    time.sleep(2.0) # A simple sleep is now effective because the buffer is empty
    
    except KeyboardInterrupt:
        print("\nExiting program.")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# GPIO.cleanup() # For Raspberry Pi
ser.close()
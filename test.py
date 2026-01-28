import serial
import keyboard
import time

SERIAL_PORT = 'COM7'
BAUD_RATE = 9600

arduino = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)  # wait for Arduino to reset

print("Use WASD to move. Press Q to quit.")

def send(cmd):
    arduino.write(cmd.encode())

while True:
    if keyboard.is_pressed('w'):
        send('F')  # forward
    elif keyboard.is_pressed('s'):
        send('B')  # backward
    elif keyboard.is_pressed('a'):
        send('L')  # left
    elif keyboard.is_pressed('d'):
        send('R')  # right
    else:
        send('S')  # stop

    if keyboard.is_pressed('q'):
        send('S')
        break

    time.sleep(0.05)
arduino.close()
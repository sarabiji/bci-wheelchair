import serial, csv, time

PORT = 'COM8'  
BAUD = 115200

ser = serial.Serial(PORT, BAUD)
time.sleep(2)

with open('0_blink_data.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'signal', 'peak'])
    start = time.time()
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if ',' in line:
                signal, peak = line.split(',')
                writer.writerow([time.time()-start, float(signal), int(peak)])
                print(signal, peak)
        except KeyboardInterrupt:
            break

ser.close()

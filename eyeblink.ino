
#include <math.h>

#define SAMPLE_RATE 75
#define BAUD_RATE 115200
#define INPUT_PIN A0
#define OUTPUT_PIN 13
#define DATA_LENGTH 10
#define PEAK_COOLDOWN 22

int data_index = 0;
bool peak = false;


void setup() {
  // Serial connection begin
  Serial.begin(BAUD_RATE);
  // Setup Input & Output pin
  pinMode(INPUT_PIN, INPUT);
  pinMode(OUTPUT_PIN, OUTPUT);
}

void loop() {
  // Calculate elapsed time
  static unsigned long past = 0;
  unsigned long present = micros();
  unsigned long interval = present - past;
  past = present;

  // Run timer
  static long timer = 0;
  timer -= interval;

  // Sample
  if (timer < 0) {
    timer += 1000000 / SAMPLE_RATE;
    float sensor_value = analogRead(INPUT_PIN);
    float signal = EOGFilter(sensor_value) / 512.0; // Use 512.0 for float division

    // Get the current blink state
    peak = updateBlinkState(signal); // <-- Update the function call here

    // Print and set LED
    Serial.print(signal);
    Serial.print(",");
    Serial.println(peak);
    digitalWrite(OUTPUT_PIN, peak);
  }
}

// A simplified function for calibration using absolute signal values.
bool updateBlinkState(float new_sample) {
  static bool is_blinking = false;

  // --- Calibrated Absolute Thresholds ---
  // YOU WILL CHANGE THESE VALUES BASED ON YOUR DATA
  const float BLINK_START_THRESHOLD = 0.2f;  // Threshold to START the blink
  const float BLINK_END_THRESHOLD   = 0.1f;  // Threshold to END the blink

  if (!is_blinking) {
    // STATE: IDLE - We are looking for a blink to start.
    if (new_sample > BLINK_START_THRESHOLD) {
      is_blinking = true; // Blink has started!
    }
  } else { // is_blinking is true
    // STATE: BLINKING - We are waiting for the blink to end.
    if (new_sample < BLINK_END_THRESHOLD) {
      is_blinking = false; // Blink has ended!
    }
  }
  return is_blinking;
}

// Band-Pass Butterworth IIR digital filter, generated using filter_gen.py.
// Sampling rate: 75.0 Hz, frequency: [0.5, 19.5] Hz.
// Filter is order 4, implemented as second-order sections (biquads).
// Reference:
// https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
// https://courses.ideate.cmu.edu/16-223/f2020/Arduino/FilterDemos/filter_gen.py
float EOGFilter(float input)
{
  float output = input;
  {
    static float z1, z2; // filter section state
    float x = output - 0.02977423 * z1 - 0.04296318 * z2;
    output = 0.09797471 * x + 0.19594942 * z1 + 0.09797471 * z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2; // filter section state
    float x = output - 0.08383952 * z1 - 0.46067709 * z2;
    output = 1.00000000 * x + 2.00000000 * z1 + 1.00000000 * z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2; // filter section state
    float x = output - -1.92167271 * z1 - 0.92347975 * z2;
    output = 1.00000000 * x + -2.00000000 * z1 + 1.00000000 * z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2; // filter section state
    float x = output - -1.96758891 * z1 - 0.96933514 * z2;
    output = 1.00000000 * x + -2.00000000 * z1 + 1.00000000 * z2;
    z2 = z1;
    z1 = x;
  }
  return output;
}

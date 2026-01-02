#include <Arduino.h>
#include <math.h>

/* ================= SETTINGS ================= */
#define BAUD_RATE     115200
#define SAMPLE_RATE   250.0
#define INPUT_PIN     A0

#define ALPHA_FREQ    10.0
#define BETA_FREQ     20.0
#define THETA_FREQ    6.0

#define WINDOW_SIZE   128
#define SMOOTHING     0.6

/* ================= GOERTZEL ================= */
float a_coeff, b_coeff, t_coeff;
float a_s1=0, a_s2=0;
float b_s1=0, b_s2=0;
float t_s1=0, t_s2=0;

int sample_count = 0;
float attention_smooth = 1.0;

float baseline_ratio = 0;
int baseline_samples = 0;
bool calibrating = true;

/* ================= TIMER ================= */
volatile bool sampleReady = false;

ISR(TIMER1_COMPA_vect) {
  sampleReady = true;
}

void setupTimer() {
  cli();
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;
  OCR1A  = (16000000 / (8 * SAMPLE_RATE)) - 1;
  TCCR1B |= (1 << WGM12) | (1 << CS11);
  TIMSK1 |= (1 << OCIE1A);
  sei();
}

/* ================= FILTER ================= */
float EEGFilter(float x) {
  static float z1=0, z2=0;
  float y = x + 1.889345*z1 - 0.893322*z2;
  float out = 0.946661*y - 1.893322*z1 + 0.946661*z2;
  z2 = z1;
  z1 = y;
  return out;
}

/* ================= GOERTZEL ================= */
inline void goertzel(float x, float coeff, float &s1, float &s2) {
  float s = x + coeff*s1 - s2;
  s2 = s1;
  s1 = s;
}

inline float power(float coeff, float s1, float s2) {
  return s2*s2 + s1*s1 - coeff*s1*s2;
}

inline void reset(float &s1, float &s2) {
  s1 = 0; s2 = 0;
}

/* ================= SETUP ================= */
void setup() {
  Serial.begin(BAUD_RATE);

  a_coeff = 2*cos(2*PI*ALPHA_FREQ/SAMPLE_RATE);
  b_coeff = 2*cos(2*PI*BETA_FREQ /SAMPLE_RATE);
  t_coeff = 2*cos(2*PI*THETA_FREQ/SAMPLE_RATE);

  setupTimer();
}

/* ================= LOOP ================= */
void loop() {

  if (!sampleReady) return;
  sampleReady = false;

  float raw = analogRead(INPUT_PIN);
  float eeg = EEGFilter(raw) / 500.0;

  goertzel(eeg, a_coeff, a_s1, a_s2);
  goertzel(eeg, b_coeff, b_s1, b_s2);
  goertzel(eeg, t_coeff, t_s1, t_s2);

  sample_count++;

  if (sample_count >= WINDOW_SIZE) {

    float alpha = power(a_coeff, a_s1, a_s2);
    float beta  = power(b_coeff, b_s1, b_s2);
    float theta = power(t_coeff, t_s1, t_s2);

    reset(a_s1, a_s2);
    reset(b_s1, b_s2);
    reset(t_s1, t_s2);
    sample_count = 0;

    float ratio = beta / (alpha + theta + 1.0);

  if (calibrating) {
    baseline_ratio += ratio;
    baseline_samples++;

    if (baseline_samples >= 20) {   // ~10s (20 windows)
      baseline_ratio /= baseline_samples;
      calibrating = false;
      Serial.println("BASELINE SET");
    }
    return;
  }

  // ----- ATTENTION COMPUTATION -----
float attention_raw = beta / (alpha + theta + 1.0);

attention_smooth = 0.65 * attention_smooth +
                   0.35 * attention_raw;

// soft scaling (no saturation)
float attention_level =
    100.0 * (attention_smooth / (attention_smooth + 1.0));

// ----- STATE CLASSIFICATION -----
const char* state;

static unsigned long focus_start = 0;

if (attention_level > 15) //15
    {if (focus_start == 0)
        focus_start = millis();

    if (millis() - focus_start > 1000)
        state = "FOCUSED";
    }
  else if (attention_level > 10)
    {focus_start = 0;
    state = "NORMAL";}
  else
    {state = "RELAXED";}



// ----- OUTPUT -----
Serial.print(attention_level, 1);
Serial.print(" , ");
Serial.println(state);



  }
}

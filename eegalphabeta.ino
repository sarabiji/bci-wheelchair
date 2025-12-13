
#include <math.h>

#define SAMPLE_RATE 256.0
#define BAUD_RATE 115200
#define INPUT_PIN A0

// Frequency Bins (Optimized for Frontal Cortex)
#define ALPHA_FREQ 10.0      // 8-12 Hz
#define BETA_FREQ 20.0       // 18-25 Hz (Focus/Concentration)
#define THETA_FREQ 6.0       // 4-8 Hz (Drowsiness/Relaxation)

// Classification Thresholds (Tuned for Frontal Lobe)
// Focus Index = Beta / (Alpha + Theta)
#define ATTENTION_RATIO_THRESHOLD 1.2     // Increased from 0.8: Frontal Beta should dominate
#define MIN_POWER_THRESHOLD 150.0         // Minimum total signal to avoid noise classification
#define ARTIFACT_STD_DEV_THRESHOLD 80.0  // Standard deviation threshold for eye blinks/motion
#define HYSTERESIS_COUNT 4                // Number of stable readings required before classification change
#define SMOOTHING_FACTOR 0.8              // Increased for smoother, more stable classification

// Goertzel window (Window size N=128 gives 0.5s time window and 2 Hz frequency resolution)
int N = 128;

// Goertzel coefficients and state
float alpha_coeff, beta_coeff, theta_coeff;
float alpha_s_prev = 0, alpha_s_prev2 = 0;
float beta_s_prev = 0, beta_s_prev2 = 0;
float theta_s_prev = 0, theta_s_prev2 = 0;

int sample_count = 0;

// Smoothed classification and Hysteresis
float smoothed_ratio = 1.0;
int prev_class = 0;
int stable_count = 0;

// Artifact detection buffer (stores 64 samples)
float signal_buffer[64];
int buffer_idx = 0;

// Function Prototypes
void alpha_goertzel(float x);
void beta_goertzel(float x);
void theta_goertzel(float x);
void reset_goertzel();
bool detect_artifact();
float EEGFilter(float input);

void setup() {
  Serial.begin(BAUD_RATE);
  
  // Calculate Goertzel coefficients
  alpha_coeff = 2.0 * cos(2.0 * PI * ALPHA_FREQ / SAMPLE_RATE);
  beta_coeff  = 2.0 * cos(2.0 * PI * BETA_FREQ  / SAMPLE_RATE);
  theta_coeff = 2.0 * cos(2.0 * PI * THETA_FREQ / SAMPLE_RATE);
  
  // Print header
  Serial.println("Alpha,Beta,Theta,Ratio,Class,Status");
  
  // Initialize buffer
  for (int i = 0; i < 64; i++) {
    signal_buffer[i] = 0;
  }
}

void loop() {
  // --- Timing Loop for Consistent Sampling Rate (256 Hz) ---
  static unsigned long past = 0;
  unsigned long present = micros();
  unsigned long interval = present - past;
  past = present;
  
  static long timer = 0;
  timer -= interval;
  
  if (timer < 0) {
    timer += 1000000 / (long)SAMPLE_RATE;
    
    // Read and filter
    // NOTE: analogRead() returns an int (0-1023 or 0-4095 depending on MCU)
    float raw = analogRead(INPUT_PIN);
    float eeg = EEGFilter(raw)/500;
    
    // Store for artifact detection
    signal_buffer[buffer_idx] = eeg;
    buffer_idx = (buffer_idx + 1) % 64;
    
    // Goertzel accumulation
    alpha_goertzel(eeg);
    beta_goertzel(eeg);
    theta_goertzel(eeg);
    
    sample_count++;
    
    // Process every N samples (500ms window)
    if (sample_count >= N) {
      
      
      // Calculate power for each band (Goertzel magnitude squared)
      float alpha_power = alpha_s_prev2 * alpha_s_prev2 +
                          alpha_s_prev * alpha_s_prev -
                          alpha_coeff * alpha_s_prev * alpha_s_prev2;
      
      float beta_power = beta_s_prev2 * beta_s_prev2 +
                          beta_s_prev * beta_s_prev -
                          beta_coeff * beta_s_prev * beta_s_prev2;
      
      float theta_power = theta_s_prev2 * theta_s_prev2 +
                          theta_s_prev * theta_s_prev -
                          theta_coeff * theta_s_prev * theta_s_prev2;

       float total_power = alpha_power + beta_power + theta_power;
      
      // Reset Goertzel state for the next window
      reset_goertzel();
      
      // Artifact detection (sudden large changes from eye blinks/movement)
      bool is_artifact = detect_artifact();
      
      if (!is_artifact) {
        // Calculate attention metrics
        float total_power = alpha_power + beta_power + theta_power;
        
        // Frontal Focus Index: Attention = Beta / (Alpha + Theta)
        float ratio = beta_power / (alpha_power + theta_power + 1.0); // +1.0 for stability
        
        // Smooth the ratio for stable classification
        smoothed_ratio = SMOOTHING_FACTOR * smoothed_ratio +
                         (1.0 - SMOOTHING_FACTOR) * ratio;
        
        // Classification logic
        int current_cls;
        const char* status;

       
        float window_std = 0;
        // compute std from signal_buffer
        float mean = 0;
        for (int i=0; i<64; i++) mean += signal_buffer[i];
        mean /= 64.0;
        for (int i=0; i<64; i++) {
          float diff = signal_buffer[i] - mean;
          window_std += diff*diff;
        }
        window_std = sqrt(window_std/64.0);
        
        // --- Signal presence check ---
        if (total_power < MIN_POWER_THRESHOLD || window_std < 0.5) { 
            current_cls = 0;
            status = "NO_SIGNAL";
        } else if (smoothed_ratio > ATTENTION_RATIO_THRESHOLD) {
            current_cls = 1;
            status = "FOCUSED";
        } else if (theta_power > alpha_power*1.8) {
            current_cls = 3;
            status = "DROWSY";
        } else {
            current_cls = 2;
            status = "RELAXED";
        }
        // Hysteresis: prevent rapid switching (requires HYSTERESIS_COUNT stable reads)
        if (current_cls == prev_class) {
          stable_count = 0; // Reset counter if the state changes
        } else {
           stable_count++;
        }
        
        if (stable_count >= HYSTERESIS_COUNT) {
          prev_class = current_cls;
          stable_count = 0;
        } else {
          // Use previous stable class until count is met
          current_cls = prev_class;
        }

        // --- Output ---
        Serial.print(alpha_power, 1);
        Serial.print(",");
        Serial.print(beta_power, 1);
        Serial.print(",");
        Serial.print(theta_power, 1);
        Serial.print(",");
        Serial.print(smoothed_ratio, 3);
        Serial.print(",");
        Serial.print(current_cls);
        Serial.print(",");
        Serial.println(status);
        
      } else {
        // Artifact detected - skip this window
        // Keep the previous stable classification for continuity
        Serial.print("0,0,0,0,");
        Serial.print(prev_class);
        Serial.println(",ARTIFACT");
      }
      
      sample_count = 0;
    }
  }
  
 

}

// ----------------------------------------
// GOERTZEL FILTERS
// ----------------------------------------
void alpha_goertzel(float x) {
  float s = x + alpha_coeff * alpha_s_prev - alpha_s_prev2;
  alpha_s_prev2 = alpha_s_prev;
  alpha_s_prev = s;
}

void beta_goertzel(float x) {
  float s = x + beta_coeff * beta_s_prev - beta_s_prev2;
  beta_s_prev2 = beta_s_prev;
  beta_s_prev = s;
}

void theta_goertzel(float x) {
  float s = x + theta_coeff * theta_s_prev - theta_s_prev2;
  theta_s_prev2 = theta_s_prev;
  theta_s_prev = s;
}

void reset_goertzel() {
  alpha_s_prev = alpha_s_prev2 = 0;
  beta_s_prev = beta_s_prev2 = 0;
  theta_s_prev = theta_s_prev2 = 0;
}

// ----------------------------------------
// ARTIFACT DETECTION
// ----------------------------------------
bool detect_artifact() {
  // Calculate variance of recent samples
  float mean = 0;
  for (int i = 0; i < 64; i++) {
    mean += signal_buffer[i];
  }
  mean /= 64.0;
  
  float variance = 0;
  for (int i = 0; i < 64; i++) {
    float diff = signal_buffer[i] - mean;
    variance += diff * diff;
  }
  variance /= 64.0;
  
  
  float std_dev = sqrt(variance);
  Serial.print("STD: "); Serial.println(std_dev);
  return (std_dev > ARTIFACT_STD_DEV_THRESHOLD);

}

// ----------------------------------------
// EEG FILTER (Adjusted for better stability at 256 Hz)
// ----------------------------------------
float EEGFilter(float input) {
  float output = input;
  
  // Stage 1: High-pass filter (0.5 Hz) - Slightly adjusted coefficients
  // to ensure DC bias is removed effectively.
  {
    static float z1 = 0, z2 = 0;
    float x = output - -1.889345*z1 - 0.893322*z2;
    output = 0.946661*x + 1.893322*z1 + 0.946661*z2;
    z2 = z1;
    z1 = x;
  }
  
  // Stage 2: Low-pass filter (30 Hz) - Slightly adjusted coefficients
  {
    static float z1 = 0, z2 = 0;
    float x = output - -1.20596630*z1 - 0.60558332*z2;
    output = 1.00000000*x + 2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  
  // Stage 3: Notch filter (50 Hz)
  {
    static float z1 = 0, z2 = 0;
    float x = output - -1.97690645*z1 - 0.97706395*z2;
    output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  
  // Stage 4: Notch filter (60 Hz)
  {
    static float z1 = 0, z2 = 0;
    float x = output - -1.99076487*z1 - 0.99086813*z2;
    output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  
  return output;
}


/*
#include <math.h>

#define SAMPLE_RATE 256
#define BAUD_RATE 115200
#define INPUT_PIN A0

// Frequency bins optimized for FRONTAL cortex
#define ALPHA_FREQ 10.0    // 8-12 Hz (weaker on forehead)
#define BETA_FREQ 20.0     // 15-25 Hz (stronger on forehead, focus)
#define THETA_FREQ 6.0     // 4-8 Hz (drowsiness, also frontal)

// Classification thresholds (ADJUSTED for forehead)
#define ATTENTION_RATIO_THRESHOLD 0.8   // LOWERED - frontal alpha is weaker
#define MIN_POWER_THRESHOLD 100.0       // Minimum signal to avoid noise classification
#define SMOOTHING_FACTOR 0.7            // For stable classification

// Goertzel window
int N = 128;  // Increased to 500ms for better frequency resolution

// Goertzel coefficients and state
float alpha_coeff, beta_coeff, theta_coeff;
float alpha_s_prev = 0, alpha_s_prev2 = 0;
float beta_s_prev = 0, beta_s_prev2 = 0;
float theta_s_prev = 0, theta_s_prev2 = 0;

int sample_count = 0;

// Smoothed classification
float smoothed_ratio = 1.0;
int prev_class = 0;

// Artifact detection
float signal_buffer[64];
int buffer_idx = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  
  // Calculate Goertzel coefficients
  alpha_coeff = 2 * cos(2 * PI * ALPHA_FREQ / SAMPLE_RATE);
  beta_coeff  = 2 * cos(2 * PI * BETA_FREQ  / SAMPLE_RATE);
  theta_coeff = 2 * cos(2 * PI * THETA_FREQ / SAMPLE_RATE);
  
  // Print header
  Serial.println("Alpha,Beta,Theta,Ratio,Class,Status");
  
  // Initialize buffer
  for (int i = 0; i < 64; i++) {
    signal_buffer[i] = 0;
  }
}

void loop() {
  static unsigned long past = 0;
  unsigned long present = micros();
  unsigned long interval = present - past;
  past = present;
  
  static long timer = 0;
  timer -= interval;
  
  if (timer < 0) {
    timer += 1000000 / SAMPLE_RATE;
    
    // Read and filter
    float raw = analogRead(INPUT_PIN);
    float eeg = EEGFilter(raw);
    
    // Store for artifact detection
    signal_buffer[buffer_idx] = eeg;
    buffer_idx = (buffer_idx + 1) % 64;
    
    // Goertzel accumulation
    alpha_goertzel(eeg);
    beta_goertzel(eeg);
    theta_goertzel(eeg);
    
    sample_count++;
    
    // Process every N samples (500ms window)
    if (sample_count >= N) {
      // Calculate power for each band
      float alpha_power = alpha_s_prev2 * alpha_s_prev2 +
                          alpha_s_prev * alpha_s_prev -
                          alpha_coeff * alpha_s_prev * alpha_s_prev2;
      
      float beta_power = beta_s_prev2 * beta_s_prev2 +
                         beta_s_prev * beta_s_prev -
                         beta_coeff * beta_s_prev * beta_s_prev2;
      
      float theta_power = theta_s_prev2 * theta_s_prev2 +
                          theta_s_prev * theta_s_prev -
                          theta_coeff * theta_s_prev * theta_s_prev2;
      
      // Reset Goertzel
      reset_goertzel();
      
      // Artifact detection (sudden large changes)
      bool is_artifact = detect_artifact();
      
      if (!is_artifact) {
        // Calculate attention metrics
        float total_power = alpha_power + beta_power + theta_power;
        
        // For FRONTAL placement: attention = high beta, low alpha+theta
        float ratio = beta_power / (alpha_power + theta_power + 1.0);
        
        // Smooth the ratio for stable classification
        smoothed_ratio = SMOOTHING_FACTOR * smoothed_ratio + 
                        (1 - SMOOTHING_FACTOR) * ratio;
        
        // Classification with power threshold
        int cls;
        const char* status;
        
        if (total_power < MIN_POWER_THRESHOLD) {
          cls = 0;  // No signal / poor contact
          status = "NO_SIGNAL";
        } else if (smoothed_ratio > ATTENTION_RATIO_THRESHOLD) {
          cls = 1;  // Focused / Attentive
          status = "FOCUSED";
        } else if (theta_power > alpha_power * 1.5) {
          cls = 3;  // Drowsy (high theta)
          status = "DROWSY";
        } else {
          cls = 2;  // Relaxed
          status = "RELAXED";
        }
        
        // Hysteresis: prevent rapid switching
        if (cls != prev_class) {
          static int stable_count = 0;
          stable_count++;
          if (stable_count < 3) {  // Require 3 consecutive readings
            cls = prev_class;
          } else {
            stable_count = 0;
            prev_class = cls;
          }
        }
        
        // Output: Alpha,Beta,Theta,Ratio,Class,Status
        Serial.print(alpha_power, 1);
        Serial.print(",");
        Serial.print(beta_power, 1);
        Serial.print(",");
        Serial.print(theta_power, 1);
        Serial.print(",");
        Serial.print(smoothed_ratio, 3);
        Serial.print(",");
        Serial.print(cls);
        Serial.print(",");
        Serial.println(status);
        
      } else {
        // Artifact detected - skip this window
        Serial.println("0,0,0,0,0,ARTIFACT");
      }
      
      sample_count = 0;
    }
  }
}

// ----------------------------------------
// GOERTZEL FILTERS
// ----------------------------------------
void alpha_goertzel(float x) {
  float s = x + alpha_coeff * alpha_s_prev - alpha_s_prev2;
  alpha_s_prev2 = alpha_s_prev;
  alpha_s_prev = s;
}

void beta_goertzel(float x) {
  float s = x + beta_coeff * beta_s_prev - beta_s_prev2;
  beta_s_prev2 = beta_s_prev;
  beta_s_prev = s;
}

void theta_goertzel(float x) {
  float s = x + theta_coeff * theta_s_prev - theta_s_prev2;
  theta_s_prev2 = theta_s_prev;
  theta_s_prev = s;
}

void reset_goertzel() {
  alpha_s_prev = alpha_s_prev2 = 0;
  beta_s_prev = beta_s_prev2 = 0;
  theta_s_prev = theta_s_prev2 = 0;
}

// ----------------------------------------
// ARTIFACT DETECTION
// ----------------------------------------
bool detect_artifact() {
  // Calculate variance of recent samples
  float mean = 0;
  for (int i = 0; i < 64; i++) {
    mean += signal_buffer[i];
  }
  mean /= 64;
  
  float variance = 0;
  for (int i = 0; i < 64; i++) {
    float diff = signal_buffer[i] - mean;
    variance += diff * diff;
  }
  variance /= 64;
  
  // Artifact = very high variance (eye blinks, movement)
  float std_dev = sqrt(variance);
  return (std_dev > 500.0);  // Adjust threshold based on your setup
}

// ----------------------------------------
// EEG FILTER (Your existing filter)
// ----------------------------------------
float EEGFilter(float input) {
  float output = input;
  
  // Stage 1: High-pass filter (0.5 Hz)
  {
    static float z1, z2;
    float x = output - -0.95391350*z1 - 0.25311356*z2;
    output = 0.00735282*x + 0.01470564*z1 + 0.00735282*z2;
    z2 = z1;
    z1 = x;
  }
  
  // Stage 2: Low-pass filter (30 Hz)
  {
    static float z1, z2;
    float x = output - -1.20596630*z1 - 0.60558332*z2;
    output = 1.00000000*x + 2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  
  // Stage 3: Notch filter (50 Hz)
  {
    static float z1, z2;
    float x = output - -1.97690645*z1 - 0.97706395*z2;
    output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  
  // Stage 4: Notch filter (60 Hz)
  {
    static float z1, z2;
    float x = output - -1.99076487*z1 - 0.99086813*z2;
    output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  
  return output;
}


FOREHEAD ELECTRODE PLACEMENT GUIDE:
====================================

BEST CONFIGURATION:
- Active electrode 1: Fp1 (left forehead, ~2cm above left eyebrow)
- Active electrode 2: Fp2 (right forehead, ~2cm above right eyebrow)
- Reference: Mastoid (behind ear) OR center forehead
- Ground: Earlobe or other mastoid

EXPECTED BEHAVIOR:
- Class 0: NO_SIGNAL (poor contact)
- Class 1: FOCUSED (high beta, working/concentrating)
- Class 2: RELAXED (low beta, calm but awake)
- Class 3: DROWSY (high theta, getting sleepy)

TESTING PROCEDURE:
1. Apply electrodes with conductive gel
2. Wait 30 seconds for stabilization
3. FOCUSED state: Do mental math (57 x 23 = ?)
4. RELAXED state: Close eyes, breathe deeply
5. Check that classification changes appropriately

TROUBLESHOOTING:
- Always getting "NO_SIGNAL": Check electrode contact
- Stuck on "ARTIFACT": Lower threshold or check grounding
- Not detecting attention: Try different beta frequency (15-25 Hz range)
- Random switching: Increase SMOOTHING_FACTOR

TUNING PARAMETERS:
- Adjust ATTENTION_RATIO_THRESHOLD (0.5-1.2) based on your baseline
- Adjust MIN_POWER_THRESHOLD (50-200) based on signal strength
- Change BETA_FREQ (18-22 Hz) to optimize for your frontal lobe
*/

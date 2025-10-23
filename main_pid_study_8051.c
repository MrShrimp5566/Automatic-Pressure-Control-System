/*
 * main_pid_study_8051.c
 * ------------------------------------------------------------
 * Runs a PID controller and a tiny ML model for diagnostics
 * on a simulated 8051 microcontroller.
 *
 * C89/ANSI C compliant version for older Keil C51 compilers.
 */
#include <reg51.h>      // Keil SFRs for classic 8051
#include "pc_model.h"   // Exported arrays + FEAT_MIN/MAX + MODEL_CRC16

// --- Global variables for the PID Controller ---
int integral = 0;
int prev_error = 0;

// --- Global variables for the Simulation ---
int simulated_pressure = 50;
int valve_control = 0;

// --- Global variables for the ML Model ---
unsigned char performance_warning_flag = 0; // 0 = OK, 1 = Untuned Performance Detected
int last_pressure = 0; // To calculate the change in pressure

// --- Simulation Functions ---
int read_pressure_sensor(void) {
    return simulated_pressure;
}

void drive_valve(int control_signal) {
    valve_control = control_signal;
}

void update_tank_pressure(int value_signal){
    int leak = 5;
    
    simulated_pressure = simulated_pressure + (value_signal / 10) - leak;

    if (simulated_pressure < 0) {
        simulated_pressure = 0;
    }
}

// --- PID Controller Function ---
int PID_update(int error) {
    int Kp = 12, Ki = 2, Kd = 2; // Your tuned gains
    int derivative, control_output;
    
    int INTEGRAL_MAX = 500; // Define a maximum limit for the integral
    int INTEGRAL_MIN = -500; // Define a minimum limit

    integral += error;

    // --- ADD THIS ANTI-WINDUP FIX ---
    // Clamp the integral term to prevent it from growing too large
    if (integral > INTEGRAL_MAX) {
        integral = INTEGRAL_MAX;
    }
    if (integral < INTEGRAL_MIN) {
        integral = INTEGRAL_MIN;
    }
    // ---------------------------------

    derivative = error - prev_error;
    
    control_output = (Kp * error) + (Ki * integral) + (Kd * derivative);

    prev_error = error;

    return control_output;
}

// --- ML Model Functions (CRC and Classifier) ---

static unsigned char classify_pid_state(const unsigned char *features) {
    int node = 0;
    char f;
    unsigned char x;
    unsigned char th;

    for (;;) {
        f = feat[node];
        if (f == -2) return value[node];
        x  = features[(unsigned char)f];
        th = thresh[node];
        node = (x <= th) ? cleft[node] : cright[node];
    }
}

// ---------- Main Control and Diagnostics Loop ----------
void main(void) {
    /* C89 Compliance: Declare all variables at the top */
    int current_pressure;
    int pressure_setpoint = 100; // Our target pressure
    int error;
    int control_signal;
    
    unsigned char feats[FEATURES]; // Feature vector for the ML model
    unsigned char post_ZN;         // Output of the ML model
    int valve_percent;
    int delta_p;

    // Optional: Model integrity check is disabled to simplify simulation
    /*
    if (model_crc16_recompute() != MODEL_CRC16) {
        while (1) { // trap // }
    }
    */

    while(1){
        // ======================================================
        // 1. PID CONTROL SECTION
        // ======================================================
        current_pressure = read_pressure_sensor();
        error = pressure_setpoint - current_pressure;
        control_signal = PID_update(error);
        drive_valve(control_signal);
        update_tank_pressure(control_signal);
        
        // ======================================================
        // 2. ML DIAGNOSTIC SECTION
        // ======================================================
        // a) Create features from the live control data
        delta_p = current_pressure - last_pressure;
        valve_percent = control_signal / 10; // Simple conversion to a percentage
        if (valve_percent > 100) valve_percent = 100;
        if (valve_percent < 0) valve_percent = 0;
        
        // b) Quantize features (simple mapping for this example)
        feats[0] = (unsigned char)(error + 128);       // Map error
        feats[1] = (unsigned char)((valve_percent * 255) / 1000); // Map 0-100 to 0-255
        feats[2] = (unsigned char)(delta_p + 128);     // Map pressure change
        
        // c) Run the classifier
        post_ZN = classify_pid_state(feats);
        
        // d) Act on the result
        if (post_ZN == 0) {
            performance_warning_flag = 1; // Set flag if performance is "Untuned"
        } else {
            performance_warning_flag = 0; // Clear flag if performance is "Tuned"
        }
        
        // e) Update state for the next loop
        last_pressure = current_pressure;
    }
}

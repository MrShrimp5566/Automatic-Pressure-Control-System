import os
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix

# ----------------- Configuration -----------------
CSV_PATH        = "secondary_tank_runs.csv"   # <-- put your experimental CSV here
EXPORT_HEADER   = "pc_model.h"                # C header for Keil/8051
RNG_SEED        = 42
DT_MAX_DEPTH    = 2                           # ultra-tiny tree (<= 2 compares)
DT_MIN_LEAF     = 25                          # avoid overfitting tiny leaves

# Optional: set to True to synthesize placeholder data if CSV is missing
ALLOW_SYNTHETIC_IF_MISSING = True

# ----------------- Optional: synthetic fallback -----------------
def synth_runs(n_runs=6, time_steps=400, seed=0) -> pd.DataFrame:
    
    Generates placeholder data with similar structure to your experiment.
    First half of runs: pre Z–N (zn_tuned=0), second half: post Z–N (zn_tuned=1).
    
    rng = np.random.default_rng(seed)
    rec = []
    for run in range(n_runs):
        zn_tuned = int(run >= n_runs // 2)
        inlet_kpa = float(rng.uniform(60, 120))
        setpoint  = float(rng.uniform(80, 100))
        outflow   = float(rng.uniform(0.2, 0.8))

        # crude gain difference to simulate pre/post tuning
        if zn_tuned:
            Kp, Ki, Kd = 2.1, 1.2, 0.5
        else:
            Kp, Ki, Kd = 0.7, 0.12, 0.04

        pressure    = inlet_kpa * rng.uniform(0.8, 0.9)
        integral    = 0.0
        prev_error  = 0.0
        for t in range(time_steps):
            error = setpoint - pressure
            integral += error
            d_error = error - prev_error

            u = Kp*error + Ki*integral + Kd*d_error
            valve = np.clip(0.01*u + outflow, 0.0, 1.0)

            # simple tank model + noise
            pressure += 0.5 * (inlet_kpa*valve - outflow*pressure)
            pressure += rng.normal(0, 0.7)
            prev_error = error
            rec.append({
                "run": run,
                "time_s": t,
                "setpoint": setpoint,
                "pressure": pressure,
                "valve_pct": 100.0*valve,
                "inlet_kpa": inlet_kpa,
                "zn_tuned": zn_tuned
            })
    return pd.DataFrame(rec)

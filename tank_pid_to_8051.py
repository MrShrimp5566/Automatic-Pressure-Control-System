# [cite: 274-476]
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
    """
    Generates placeholder data with similar structure to your experiment.
    First half of runs: pre Z–N (zn_tuned=0), second half: post Z–N (zn_tuned=1).
    """
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
# ----------------- Load CSV or synthesize -----------------
def load_or_synthesize(csv_path: str) -> pd.DataFrame:
    """
    Loads your experimental CSV with required columns; else synthesizes data
    (for testing the pipeline) if ALLOW_SYNTHETIC_IF_MISSING is True.
    """
    need = ["run", "time_s", "setpoint", "pressure", "valve_pct", "inlet_kpa", "zn_tuned"]
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")
        return df[need].copy()
    if not ALLOW_SYNTHETIC_IF_MISSING:
        raise FileNotFoundError(f"{csv_path} not found and synthetic fallback disabled.")
    print(f"[INFO] {csv_path} not found — generating synthetic placeholder data.")
    return synth_runs(seed=RNG_SEED)
# ----------------- Feature engineering -----------------
def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Create compact features that are easy for 8051:
        error   = setpoint - pressure
        valve_q = valve_pct (already scaled 0..100) -> we will re-quantize later
        DeltaP  = pressure[t] - pressure[t-1] (trend)
    Returns:
        X (float32, shape [N,3]), y (int 0/1), and a feature DataFrame for inspection.
    """
    fe = df[["setpoint", "pressure", "valve_pct", "zn_tuned"]].copy()
    fe["error"]  = fe["setpoint"] - fe["pressure"]
    fe["DeltaP"] = fe["pressure"].diff().fillna(0.0)
    X = fe[["error", "valve_pct", "DeltaP"]].values.astype(np.float32)
    y = fe["zn_tuned"].values.astype(int)
    return X, y, fe
# ----------------- Per-feature quantization to uint8 -----------------
def quantize_uint8_per_feature(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map each feature column to [0..255] independently:
        Xq = round( (X - min) / (max - min) * 255 )
    Returns (Xq, mins, maxs) so the MCU can mirror the mapping.
    """
    mins = X.min(axis=0).astype(np.float32)
    maxs = X.max(axis=0).astype(np.float32)
    dom  = np.where(maxs > mins, (maxs - mins), 1.0).astype(np.float32)
    X01  = (X - mins) / dom
    Xq   = np.clip(np.round(255.0 * X01), 0, 255).astype(np.uint8)
    return Xq, mins, maxs
# ----------------- CRC16-CCITT for trusted inference -----------------
def crc16_ccitt(data: bytes, poly=0x1021, init=0xFFFF) -> int:
    """
    Bitwise CRC16-CCITT (False). Small and MCU-friendly.
    """
    crc = init
    for b in data:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF
# ----------------- Export header for Keil/8051 -----------------
def export_tree_header(tree: DecisionTreeClassifier,
                       mins: np.ndarray, maxs: np.ndarray,
                       out_path: str) -> None:
    """
    Emits:
        #define NODES, #define FEATURES (3)
        static const int8_t   feat[NODES];     // -2 for leaves
        static const uint8_t  thresh[NODES];   // 0..255
        static const int32_t  cleft[NODES];    // left child
        static const int32_t  cright[NODES];   // right child
        static const uint8_t  value[NODES];    // majority class at node
        static const float    FEAT_MIN[3];     // per-feature min
        static const float    FEAT_MAX[3];     // per-feature max
        static const uint16_t MODEL_CRC16;     // integrity tag
    """
    t = tree.tree_
    n = t.node_count
    LEAF = -2
    feat   = []
    thresh = []
    cleft  = []
    cright = []
    major  = np.argmax(t.value.squeeze(axis=1), axis=1).astype(int).tolist()
    # We trained on uint8 features, so thresholds are already ~0..255.
    for i in range(n):
        L, R = t.children_left[i], t.children_right[i]
        if L == -1 and R == -1:
            feat.append(LEAF); thresh.append(0); cleft.append(-1); cright.append(-1)
        else:
            fidx = int(t.feature[i])
            thr  = int(np.clip(np.round(t.threshold[i]), 0, 255))
            feat.append(fidx); thresh.append(thr); cleft.append(int(L)); cright.append(int(R))
    # Build CRC blob in the same order we will verify on the MCU
    blob = bytearray()
    for v in feat:   blob += ((v + 256) & 0xFF).to_bytes(1, "little")  # pack -2 as 0xFE
    for v in thresh: blob += (v & 0xFF).to_bytes(1, "little")
    # Changed to unsigned 4-byte conversion
    for v in cleft:  blob += (v & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
    for v in cright: blob += (v & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
    for v in major:  blob += (v & 0xFF).to_bytes(1, "little")
    crc = crc16_ccitt(bytes(blob))
    with open(out_path, "w") as f:
        f.write("// Auto-generated Decision Tree for 8051 (Keil)\n")
        f.write("// Task: classify pre/post Ziegler–Nichols (0=pre, 1=post)\n")
        f.write("#ifndef PC_MODEL_H\n#define PC_MODEL_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NODES ({n})\n")
        f.write(f"#define FEATURES (3)\n\n")
        def w(ctype, name, arr, per=16):
            f.write(f"static const {ctype} {name}[NODES] = {{\n  ")
            for i, v in enumerate(arr):
                f.write(str(int(v)))
                if i != len(arr) - 1: f.write(", ")
                if (i + 1) % per == 0 and i != len(arr) - 1: f.write("\n  ")
            f.write("\n};\n\n")
        w("int8_t",   "feat",   feat)
        w("uint8_t",  "thresh", thresh)
        w("int32_t",  "cleft",  cleft)
        w("int32_t",  "cright", cright)
        w("uint8_t",  "value",  major)
        f.write(f"static const float FEAT_MIN[FEATURES] = "
                f"{{ {mins[0]:.6f}f, {mins[1]:.6f}f, {mins[2]:.6f}f }};\n")
        f.write(f"static const float FEAT_MAX[FEATURES] = "
                f"{{ {maxs[0]:.6f}f, {maxs[1]:.6f}f, {maxs[2]:.6f}f }};\n")
        f.write(f"static const uint16_t MODEL_CRC16 = 0x{crc:04X};\n\n")
        f.write("#endif // PC_MODEL_H\n")
    print(f"[OK] Exported '{out_path}', CRC16=0x{crc:04X}")
# ----------------- Main pipeline -----------------
def main():
    # 1) Load your experimental CSV (or synthesize if missing)
    df = load_or_synthesize(CSV_PATH)
    # 2) Build lightweight features (easy for MCU to reproduce)
    X, y, _ = build_features(df)
    # 3) Per-feature quantization to byte domain
    Xq, mins, maxs = quantize_uint8_per_feature(X)
    # 4) Train/test split in byte space (parity with MCU)
    Xtr, Xte, ytr, yte = train_test_split(
        Xq, y, test_size=0.2, random_state=RNG_SEED, stratify=y
    )
    # 5) Ultra-tiny Decision Tree (<= 2 comparisons, non-recursive traversal)
    tree = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH,
        min_samples_leaf=DT_MIN_LEAF,
        random_state=RNG_SEED
    )
    tree.fit(Xtr, ytr)
    # 6) Inspect rules (helps explain to your audience)
    print("\n=== Tiny Decision Tree Rules (byte features) ===")
    print(export_text(tree, feature_names=["error_q","valve_q","DeltaP_q"]))
    # 7) Quick evaluation on holdout
    yhat = tree.predict(Xte)
    print("\n=== Test Report ===")
    print(classification_report(yte, yhat, digits=4))
    print("Confusion matrix:\n", confusion_matrix(yte, yhat))
    # 8) Export Keil/8051 header with arrays, quant ranges, CRC
    export_tree_header(tree, mins, maxs, EXPORT_HEADER)
    print(f"\n[DONE] Copy '{EXPORT_HEADER}' into your Keil project.")
if __name__ == "__main__":
    main()

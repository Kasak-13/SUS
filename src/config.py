# Global Configuration & Calibrated Thresholds for SUS
RANDOM_SEED = 42
INITIAL_TRAIN_SIZE = 400
WINDOW_SIZE = 50

# Calibrated Controller Thresholds
ANOMALY_THRESHOLD = 0.58             # Isolation Forest raw score threshold (calibrated 99th percentile)
ANOMALY_FRACTION_THRESHOLD = 0.10    # Windowed extreme-score fraction threshold for dense anomaly evidence
DRIFT_THRESHOLD = 0.75               # Sequential two-sample Kolmogorov-Smirnov statistic threshold

# Controller Safeguards
MIN_CONSECUTIVE_OBSERVATIONS = 3     # Consecutive ADAPT proposals required before retraining baseline
HOLD_DURATION = 10                   # Lockdown hysteresis timesteps to maintain HOLD after alert

# Canonical continuous raw SKAB testbed dataset
DATA_FILE = "data/raw/skab_repo/data/valve1/0.csv"

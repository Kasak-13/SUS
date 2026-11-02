import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.preprocess import load_and_preprocess
from src.static_detector import IsolationForestDetector
from src.synthetic_drift import inject_gradual_drift
from src.drift_monitor import DriftMonitor

def main():
    data = load_and_preprocess(config.DATA_FILE)
    
    # 1. Inject gradual drift into the stream segment 
    # This simulates legitimate change in the environment, e.g. seasonal shift
    drift_rate = 0.015  # Sufficient rate to cause the static detector to fail
    X_stream_drifted = inject_gradual_drift(data['X_stream'], drift_rate=drift_rate)
    y_stream = data['y_stream']
    t_stream = data['t_stream']
    
    # 2. Train Static Baseline on the clean historical data (No Drift)
    detector = IsolationForestDetector()
    detector.fit(data['X_train'])
    
    train_scores = detector.score(data['X_train'])
    threshold = np.percentile(train_scores, 99)
    print(f"Frozen Threshold (from Train Set): {threshold:.4f}")
    
    # 3. Initialize Drift Monitor using historical data as reference
    monitor = DriftMonitor(reference_data=data['X_train'])
    
    preds = []
    scores = []
    drift_flags = []
    
    print("Streaming data and predicting...")
    # 4. Stream simulation
    for i in range(len(X_stream_drifted)):
        x_obs = X_stream_drifted[i:i+1] # 2D array for sklearn
        
        # Anomaly detection (using frozen static detector)
        score = detector.score(x_obs)[0]
        pred = int(score > threshold)
        scores.append(score)
        preds.append(pred)
        
        # Drift detection
        is_drift, p_vals = monitor.update_and_check(X_stream_drifted[i])
        drift_flags.append(int(is_drift))
        
    preds = np.array(preds)
    
    # 5. Calculate Metrics
    prec = precision_score(y_stream, preds, zero_division=0)
    rec = recall_score(y_stream, preds, zero_division=0)
    f1 = f1_score(y_stream, preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_stream, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'Precision': [prec],
        'Recall': [rec],
        'F1_Score': [f1],
        'FPR': [fpr],
        'FNR': [fnr],
        'TP': [tp],
        'FP': [fp],
        'FN': [fn],
        'TN': [tn]
    }
    
    df_metrics = pd.DataFrame(metrics)
    os.makedirs('results/tables', exist_ok=True)
    df_metrics.to_csv('results/tables/scenario_b_metrics.csv', index=False)
    
    print("\n--- Scenario B (Static Detector on Drifted Stream) Metrics ---")
    print(df_metrics.to_string(index=False))
    print(f"\nNotice the explosion of False Positives (FPR: {fpr:.4f}) compared to Scenario A.")
    print(f"Drift was detected in {sum(drift_flags)} out of {len(t_stream)} timesteps.")
    
    # 6. Plotting
    plt.figure(figsize=(15, 6))
    
    feat_idx = 0
    feature_name = data['feature_names'][feat_idx]
    
    plt.plot(t_stream, X_stream_drifted[:, feat_idx], label=f'Drifted {feature_name}', color='gray', alpha=0.7)
    
    # True anomalies
    anom_times = t_stream[y_stream == 1]
    for t in anom_times:
        plt.axvline(t, color='red', alpha=0.1)
        
    # Detected anomalies (Predictions)
    pred_anom_times = t_stream[preds == 1]
    plt.scatter(pred_anom_times, X_stream_drifted[preds == 1, feat_idx], color='blue', marker='x', s=50, label='Detected Anomaly (False Alarms Explosion)')
    
    # Highlight periods where drift is detected
    drift_times = t_stream[np.array(drift_flags) == 1]
    plt.scatter(drift_times, [np.min(X_stream_drifted[:, feat_idx])] * len(drift_times), color='orange', marker='_', label='KS Drift Detected', alpha=0.5)
    
    plt.title('Scenario B: Static Detector Failure on Legitimate Drift')
    plt.xlabel('Time')
    plt.ylabel('Standardized Value')
    plt.legend()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/scenario_b_plot.png')
    print("\nSaved plot to results/figures/scenario_b_plot.png")

if __name__ == '__main__':
    main()

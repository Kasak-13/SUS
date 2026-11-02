import pandas as pd
import numpy as np
import time
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.preprocess import load_and_preprocess
from src.static_detector import IsolationForestDetector
from src.adapt import NaiveAdaptiveDetector
from src.threshold_adapt import ThresholdAdaptiveDetector
from src.sus_detector import SUSDetector
from src.synthetic_drift import inject_gradual_drift

def evaluate_predictions(y_true, y_pred, runtime, detector_name):
    y_pred = np.array(y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (len(y_true) - np.sum(y_pred), np.sum(y_pred), 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'Detector': detector_name,
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1_Score': round(f1, 4),
        'False_Positive_Rate (FPR)': round(fpr, 4),
        'False_Negative_Rate (FNR)': round(fnr, 4),
        'Runtime(s)': round(runtime, 2)
    }

def run_four_systems(X_train, X_stream, y_stream, scenario_name):
    print(f"\n--- Running 4-System Evaluation on Scenario {scenario_name} ---")
    results = []
    
    # 1. Static Isolation Forest
    t0 = time.time()
    static = IsolationForestDetector()
    static.fit(X_train)
    thresh_static = np.percentile(static.score(X_train), 99)
    preds_static = [int(static.score(x.reshape(1, -1))[0] > thresh_static) for x in X_stream]
    results.append({**{'Scenario': scenario_name}, **evaluate_predictions(y_stream, preds_static, time.time() - t0, '1. Static (Frozen)')})
    
    # 2. Naive Adaptive Detector
    t0 = time.time()
    naive = NaiveAdaptiveDetector(update_interval=50)
    naive.fit(X_train)
    preds_naive = [naive.update_and_predict(x)[0] for x in X_stream]
    results.append({**{'Scenario': scenario_name}, **evaluate_predictions(y_stream, preds_naive, time.time() - t0, '2. Naive Adaptive')})
    
    # 3. Threshold-only Adaptive
    t0 = time.time()
    thresh_det = ThresholdAdaptiveDetector()
    thresh_det.fit(X_train)
    preds_thresh = [thresh_det.update_and_predict(x)[0] for x in X_stream]
    results.append({**{'Scenario': scenario_name}, **evaluate_predictions(y_stream, preds_thresh, time.time() - t0, '3. Threshold Adaptive')})
    
    # 4. SUS Controller (Proposed)
    t0 = time.time()
    sus = SUSDetector()
    sus.fit(X_train)
    preds_sus = []
    for x in X_stream:
        score, decision = sus.update_and_predict(x)
        # SUS flags anomaly if score exceeds its protected threshold
        preds_sus.append(int(score > sus.anomaly_threshold))
    results.append({**{'Scenario': scenario_name}, **evaluate_predictions(y_stream, preds_sus, time.time() - t0, '4. SUS Controller (Ours)')})
    
    return results

def main():
    data = load_and_preprocess(config.DATA_FILE)
    X_train = data['X_train']
    
    # Scenario A (Stable clean baseline)
    X_stream_A = data['X_stream'][:173]
    y_stream_A = data['y_stream'][:173]
    
    # Scenario B (Gradual drift rate 0.001)
    X_stream_B = inject_gradual_drift(X_stream_A, drift_rate=0.001)
    y_stream_B = y_stream_A
    
    # Scenario D (Real physical anomaly)
    X_stream_D = data['X_stream']
    y_stream_D = data['y_stream']
    
    all_results = []
    all_results.extend(run_four_systems(X_train, X_stream_A, y_stream_A, 'A (Stable Clean)'))
    all_results.extend(run_four_systems(X_train, X_stream_B, y_stream_B, 'B (Gradual Drift)'))
    all_results.extend(run_four_systems(X_train, X_stream_D, y_stream_D, 'D (Physical Anomaly)'))
    
    df = pd.DataFrame(all_results)
    os.makedirs('results/tables', exist_ok=True)
    df.to_csv('results/tables/stage8_four_systems_comparison.csv', index=False)
    
    print("\n================ FINAL STAGE 8 FOUR-SYSTEM BENCHMARK ================")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()


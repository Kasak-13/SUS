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
from src.synthetic_drift import inject_gradual_drift

def calculate_detection_delay(y_true, y_pred):
    delays = []
    in_anomaly = False
    start_idx = -1
    detected = False
    
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_anomaly:
            in_anomaly = True
            start_idx = i
            detected = False
            
        if y_true[i] == 1 and in_anomaly and not detected:
            if y_pred[i] == 1:
                delays.append(i - start_idx)
                detected = True
                
        if y_true[i] == 0 and in_anomaly:
            in_anomaly = False
            if not detected:
                delays.append(i - start_idx)
                
    if in_anomaly and not detected:
        delays.append(len(y_true) - start_idx)
        
    return np.mean(delays) if delays else 0

def run_evaluation(detector_name, detector_type, X_train, X_stream, y_stream):
    start_time = time.time()
    
    if detector_type == 'static':
        detector = IsolationForestDetector()
        detector.fit(X_train)
        threshold = np.percentile(detector.score(X_train), 99)
        
        preds = []
        for i in range(len(X_stream)):
            score = detector.score(X_stream[i:i+1])[0]
            preds.append(int(score > threshold))
            
    else:
        if detector_type == 'naive':
            detector = NaiveAdaptiveDetector(update_interval=50)
        elif detector_type == 'threshold':
            detector = ThresholdAdaptiveDetector()
            
        detector.fit(X_train)
        preds = []
        for i in range(len(X_stream)):
            pred, _ = detector.update_and_predict(X_stream[i])
            preds.append(pred)
            
    runtime = time.time() - start_time
    preds = np.array(preds)
    
    prec = precision_score(y_stream, preds, zero_division=0)
    rec = recall_score(y_stream, preds, zero_division=0)
    f1 = f1_score(y_stream, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_stream, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    delay = calculate_detection_delay(y_stream, preds)
    
    return {
        'Detector': detector_name,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'FPR': fpr,
        'FNR': fnr,
        'Delay': delay,
        'Runtime(s)': runtime
    }

def main():
    # 1. Base data for all scenarios: valve1/0.csv
    data_real = load_and_preprocess(config.DATA_FILE)
    X_train = data_real['X_train']
    
    # 2. Scenario A_Clean: The truly stable lead-in segment BEFORE the physical anomaly hits.
    # Training was rows 0-399. The anomaly starts at row 573.
    # Therefore, stream rows 0 to 172 (173 steps) are 100% clean.
    X_stream_A = data_real['X_stream'][:173]
    y_stream_A = data_real['y_stream'][:173]
    
    # 3. Scenario D_Real (Persistent Contamination): The full stream including massive anomalies
    X_stream_D = data_real['X_stream']
    y_stream_D = data_real['y_stream']
    
    # 4. Scenario B_Drifted (Controlled Synthetic Drift): Drift injected into the clean lead-in
    X_stream_B = inject_gradual_drift(X_stream_A, drift_rate=0.005)
    y_stream_B = y_stream_A
    
    results = []
    
    print("Running Scenario A (Clean Baseline)...")
    results.append({**{'Scenario': 'A_Clean'}, **run_evaluation('Static', 'static', X_train, X_stream_A, y_stream_A)})
    results.append({**{'Scenario': 'A_Clean'}, **run_evaluation('Naive_Adaptive', 'naive', X_train, X_stream_A, y_stream_A)})
    results.append({**{'Scenario': 'A_Clean'}, **run_evaluation('Threshold_Adaptive', 'threshold', X_train, X_stream_A, y_stream_A)})
    
    print("Running Scenario B (Controlled Synthetic Drift)...")
    results.append({**{'Scenario': 'B_Drifted'}, **run_evaluation('Static', 'static', X_train, X_stream_B, y_stream_B)})
    results.append({**{'Scenario': 'B_Drifted'}, **run_evaluation('Naive_Adaptive', 'naive', X_train, X_stream_B, y_stream_B)})
    results.append({**{'Scenario': 'B_Drifted'}, **run_evaluation('Threshold_Adaptive', 'threshold', X_train, X_stream_B, y_stream_B)})
    
    print("Running Scenario D_Real (Persistent Contamination)...")
    results.append({**{'Scenario': 'D_Real'}, **run_evaluation('Static', 'static', X_train, X_stream_D, y_stream_D)})
    results.append({**{'Scenario': 'D_Real'}, **run_evaluation('Naive_Adaptive', 'naive', X_train, X_stream_D, y_stream_D)})
    results.append({**{'Scenario': 'D_Real'}, **run_evaluation('Threshold_Adaptive', 'threshold', X_train, X_stream_D, y_stream_D)})
    
    df = pd.DataFrame(results)
    os.makedirs('results/tables', exist_ok=True)
    df.to_csv('results/tables/scenario_metrics.csv', index=False)
    
    print("\n--- Summary of Results ---")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()

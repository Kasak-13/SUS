import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.preprocess import load_and_preprocess
from src.static_detector import IsolationForestDetector
from src.adapt import NaiveAdaptiveDetector
from src.threshold_adapt import ThresholdAdaptiveDetector
from src.sus_detector import SUSDetector
from src.synthetic_drift import inject_gradual_drift

def manual_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1_Score': round(f1, 4),
        'FPR': round(fpr, 4),
        'FNR': round(fnr, 4)
    }

def run_pipeline(data_dict, random_seed=42):
    # Enforce random seed
    np.random.seed(random_seed)
    
    X_train = data_dict['X_train']
    
    # Scenarios
    scenarios = {}
    scenarios['A'] = {
        'name': 'A (Stable Clean)',
        'X': data_dict['X_stream'][:173],
        'y': data_dict['y_stream'][:173]
    }
    scenarios['B'] = {
        'name': 'B (Gradual Drift)',
        'X': inject_gradual_drift(scenarios['A']['X'], drift_rate=0.001),
        'y': scenarios['A']['y']
    }
    scenarios['D'] = {
        'name': 'D (Physical Anomaly)',
        'X': data_dict['X_stream'],
        'y': data_dict['y_stream']
    }
    
    run_records = []
    raw_predictions = {}
    
    for s_code, sc in scenarios.items():
        X_s = sc['X']
        y_s = sc['y']
        
        # 1. Static
        static = IsolationForestDetector(random_state=random_seed)
        static.fit(X_train)
        th_static = np.percentile(static.score(X_train), 99)
        preds_static = [int(static.score(x.reshape(1, -1))[0] > th_static) for x in X_s]
        
        # 2. Naive Adaptive
        naive = NaiveAdaptiveDetector(update_interval=50, random_state=random_seed)
        naive.fit(X_train)
        preds_naive = [naive.update_and_predict(x)[0] for x in X_s]
        
        # 3. Threshold Adaptive
        thresh = ThresholdAdaptiveDetector(random_state=random_seed)
        thresh.fit(X_train)
        preds_thresh = [thresh.update_and_predict(x)[0] for x in X_s]
        
        # 4. Corrected SUS
        sus = SUSDetector(random_state=random_seed)
        sus.fit(X_train)
        preds_sus = []
        for x in X_s:
            sus.update_and_predict(x)
            preds_sus.append(int(sus.last_pred))
            
        raw_predictions[s_code] = {
            'y_true': y_s,
            'Static': preds_static,
            'Naive': preds_naive,
            'Threshold': preds_thresh,
            'SUS': preds_sus
        }
        
        for det_name, preds in [('1. Static (Frozen)', preds_static),
                                ('2. Naive Adaptive', preds_naive),
                                ('3. Threshold Adaptive', preds_thresh),
                                ('4. SUS Controller (Ours)', preds_sus)]:
            m = manual_metrics(y_s, preds)
            run_records.append({
                'Scenario': sc['name'],
                'Detector': det_name,
                **m
            })
            
    return pd.DataFrame(run_records), raw_predictions

def run_ablation_investigation(data_dict, random_seed=42):
    """
    Investigates the exact cause of performance improvement in Scenario D:
    Tests 4 combinations:
    1. Original: HOLD observations included in buffer_X + post-retrain threshold
    2. Version Fix Only: HOLD observations included in buffer_X + pre-retrain threshold (last_pred)
    3. HOLD Exclusion Only: HOLD observations excluded from buffer_X + post-retrain threshold
    4. Both (Current Corrected): HOLD observations excluded + pre-retrain threshold (last_pred)
    """
    X_train = data_dict['X_train']
    X_stream_D = data_dict['X_stream']
    y_stream_D = data_dict['y_stream']
    
    configs = [
        ('1. Old Implementation (Leakage + Contaminated Buffer)', False, 'post'),
        ('2. Fix Versioning Only (Pre-Retrain Thresh, Contaminated Buffer)', False, 'pre'),
        ('3. Fix HOLD Exclusion Only (Post-Retrain Thresh, Clean Buffer)', True, 'post'),
        ('4. Both Fixes (Current Corrected SUS)', True, 'pre')
    ]
    
    ablation_results = []
    
    for label, exclude_hold, thresh_mode in configs:
        np.random.seed(random_seed)
        det = SUSDetector(random_state=random_seed)
        det.fit(X_train)
        
        preds = []
        hold_count = 0
        adapt_count = 0
        
        for i, x in enumerate(X_stream_D):
            # Model & Threshold before this step
            thresh_before = det.anomaly_threshold
            x_2d = x.reshape(1, -1)
            score = float(-det.model.score_samples(x_2d)[0])
            drift_score = float(det.drift_monitor.update_and_check(x))
            det.last_drift_score = drift_score
            
            decision = None
            if len(det.drift_monitor.stream_buffer) >= det.drift_monitor.buffer_size:
                decision = det.controller.decide(score, drift_score)
            else:
                det.controller.record_observation(score)
                
            # Buffer insertion policy
            if exclude_hold:
                if decision != 'HOLD':
                    det.buffer_X.append(x)
                    if len(det.buffer_X) > det.window_size:
                        det.buffer_X.pop(0)
                else:
                    hold_count += 1
            else:
                # Old bug: append unconditionally
                det.buffer_X.append(x)
                if len(det.buffer_X) > det.window_size:
                    det.buffer_X.pop(0)
                if decision == 'HOLD':
                    hold_count += 1
                    
            if decision == 'ADAPT':
                adapt_count += 1
                det._retrain()
                
            # Evaluation threshold mode
            if thresh_mode == 'pre':
                pred = int(score > thresh_before)
            else:
                pred = int(score > det.anomaly_threshold)
                
            preds.append(pred)
            
        m = manual_metrics(y_stream_D, preds)
        ablation_results.append({
            'Configuration': label,
            'HOLD Count': hold_count,
            'ADAPT Count': adapt_count,
            **m
        })
        
    return pd.DataFrame(ablation_results)

if __name__ == '__main__':
    data = load_and_preprocess(config.DATA_FILE)
    
    print("=================================================================")
    print("PART 1: INDEPENDENT BENCHMARK & METRIC CALCULATION VERIFICATION")
    print("=================================================================")
    df_metrics, raw_preds = run_pipeline(data, random_seed=config.RANDOM_SEED)
    print(df_metrics.to_string(index=False))
    
    print("\n=================================================================")
    print("PART 2: REPRODUCIBILITY VERIFICATION (3 INDEPENDENT RUNS)")
    print("=================================================================")
    dfs = []
    for r in range(1, 4):
        df_r, _ = run_pipeline(data, random_seed=config.RANDOM_SEED)
        df_r['Run'] = r
        dfs.append(df_r)
        print(f"Run {r} completed.")
        
    # Check consistency across runs
    diff_count = 0
    cols_to_check = ['TP', 'TN', 'FP', 'FN', 'Precision', 'Recall', 'F1_Score', 'FPR', 'FNR']
    for c in cols_to_check:
        if not (dfs[0][c].equals(dfs[1][c]) and dfs[1][c].equals(dfs[2][c])):
            print(f"Discrepancy detected in column {c}!")
            diff_count += 1
            
    if diff_count == 0:
        print("=> REPRODUCIBILITY CONFIRMED: All 3 runs produced 100% BIT-FOR-BIT IDENTICAL results across all metrics and scenarios.")
    else:
        print(f"=> NON-DETERMINISM DETECTED across {diff_count} columns!")

    print("\n=================================================================")
    print("PART 3: PERFORMANCE IMPROVEMENT ABLATION (SCENARIO D)")
    print("=================================================================")
    df_ablation = run_ablation_investigation(data, random_seed=config.RANDOM_SEED)
    print(df_ablation.to_string(index=False))

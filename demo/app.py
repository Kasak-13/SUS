import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.preprocess import load_and_preprocess
from src.static_detector import IsolationForestDetector
from src.adapt import NaiveAdaptiveDetector
from src.threshold_adapt import ThresholdAdaptiveDetector
from src.sus_detector import SUSDetector
from src.synthetic_drift import inject_gradual_drift

app = Flask(__name__, static_folder='static')

print(">> [SUS Backend] Loading & Preprocessing SKAB Sensor Data...")
data_real = load_and_preprocess(config.DATA_FILE)
X_train = data_real['X_train']
X_stream_A = data_real['X_stream'][:173]
y_stream_A = data_real['y_stream'][:173]

X_stream_D = data_real['X_stream']
y_stream_D = data_real['y_stream']

X_stream_B = inject_gradual_drift(X_stream_A, drift_rate=0.001)
y_stream_B = y_stream_A

def run_all_detectors_stream(X_stream, y_stream):
    """
    Executes all 4 systems in parallel step-by-step:
    1. Static Isolation Forest
    2. Naive Adaptive Detector (Blind Retraining)
    3. Threshold Adaptive Detector (Rolling Quantile)
    4. SUS Controller & Detector (Intelligent Adaptation Safeguard)
    """
    # 1. Static
    static_det = IsolationForestDetector()
    static_det.fit(X_train)
    static_thresh = np.percentile(static_det.score(X_train), 99)
    
    # 2. Naive
    naive_det = NaiveAdaptiveDetector(update_interval=50)
    naive_det.fit(X_train)
    
    # 3. Threshold
    thresh_det = ThresholdAdaptiveDetector()
    thresh_det.fit(X_train)
    
    # 4. SUS
    sus_det = SUSDetector()
    sus_det.fit(X_train)
    
    results = []
    
    for i, x in enumerate(X_stream):
        # 1. Static
        score_static = float(static_det.score(x.reshape(1, -1))[0])
        pred_static = int(score_static > static_thresh)
        
        # 2. Naive
        pred_naive, score_naive = naive_det.update_and_predict(x)
        pred_naive = int(pred_naive)
        
        # 3. Threshold
        pred_thresh, score_thresh = thresh_det.update_and_predict(x)
        pred_thresh = int(pred_thresh)
        
        # 4. SUS
        score_sus, decision_sus = sus_det.update_and_predict(x)
        pred_sus = int(score_sus > sus_det.anomaly_threshold)
        drift_score = float(sus_det.last_drift_score)
        
        # Calculate anomaly fraction
        fraction_high = 0.0
        if len(sus_det.controller.anomaly_buffer) > 0:
            fraction_high = sum(1 for s in sus_det.controller.anomaly_buffer if s > sus_det.controller.anomaly_threshold) / len(sus_det.controller.anomaly_buffer)
            
        results.append({
            "step": i,
            "sensor_mean": float(x.mean()),
            "is_true_anomaly": int(y_stream[i]),
            "anomaly_score": float(score_sus),
            "anomaly_fraction": float(fraction_high),
            "drift_score": float(drift_score),
            "decision": decision_sus if decision_sus is not None else "BUFFERING",
            
            # 4-System Outputs
            "static_pred": pred_static,
            "static_score": round(score_static, 4),
            "naive_pred": pred_naive,
            "naive_score": round(float(score_naive), 4),
            "thresh_pred": pred_thresh,
            "thresh_score": round(float(score_thresh), 4),
            "sus_pred": pred_sus,
            "sus_score": round(float(score_sus), 4)
        })
        
    return results

# Precompute cache for instant UI response
SCENARIO_CACHE = {}

def init_cache():
    print(">> [SUS Backend] Precomputing Scenario A (Stable Clean)...")
    SCENARIO_CACHE['A'] = run_all_detectors_stream(X_stream_A, y_stream_A)
    
    print(">> [SUS Backend] Precomputing Scenario B (Gradual Drift)...")
    SCENARIO_CACHE['B'] = run_all_detectors_stream(X_stream_B, y_stream_B)
    
    print(">> [SUS Backend] Precomputing Scenario D (Physical Anomaly)...")
    SCENARIO_CACHE['D'] = run_all_detectors_stream(X_stream_D, y_stream_D)
    print(">> [SUS Backend] Precomputation complete! Server ready.")

init_cache()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/scenario/<scenario_name>')
def get_scenario(scenario_name):
    if scenario_name in SCENARIO_CACHE:
        return jsonify(SCENARIO_CACHE[scenario_name])
    return jsonify({"error": "Unknown scenario"}), 400

@app.route('/api/benchmark')
def get_benchmark():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables', 'stage8_four_systems_comparison.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return jsonify(df.to_dict(orient='records'))
    return jsonify([])

@app.route('/api/custom_simulation', methods=['POST'])
def custom_simulation():
    """Allows live interactive simulation with custom fault/drift injection parameters."""
    req_data = request.get_json() or {}
    drift_rate = float(req_data.get('drift_rate', 0.001))
    inject_anomaly = bool(req_data.get('inject_anomaly', False))
    anomaly_step = int(req_data.get('anomaly_step', 60))
    anomaly_duration = int(req_data.get('anomaly_duration', 10))
    
    # Base is A
    X_custom = inject_gradual_drift(X_stream_A.copy(), drift_rate=drift_rate)
    y_custom = y_stream_A.copy()
    
    if inject_anomaly:
        end_step = min(len(X_custom), anomaly_step + anomaly_duration)
        X_custom[anomaly_step:end_step] *= 4.5
        y_custom[anomaly_step:end_step] = 1
        
    results = run_all_detectors_stream(X_custom, y_custom)
    return jsonify(results)

if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
    app.run(debug=False, port=5000)

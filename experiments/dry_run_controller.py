import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.preprocess import load_and_preprocess
from src.static_detector import IsolationForestDetector
from src.drift_monitor import DriftMonitor
from src.synthetic_drift import inject_gradual_drift
from src.controller import SUSController

def plot_decisions(name, decisions, y_true):
    """Plots the timeline of KEEP/ADAPT/HOLD decisions."""
    plt.figure(figsize=(12, 4))
    
    # Mapping for colors
    colors = {'KEEP': 'green', 'ADAPT': 'blue', 'HOLD': 'red'}
    y_vals = {'KEEP': 0, 'ADAPT': 1, 'HOLD': 2}
    
    # Plot true anomalies as background shading
    in_anomaly = False
    start_idx = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_anomaly:
            in_anomaly = True
            start_idx = i
        elif y_true[i] == 0 and in_anomaly:
            in_anomaly = False
            plt.axvspan(start_idx, i, color='gray', alpha=0.3, label='True Anomaly' if start_idx == 0 else "")
            
    # Plot decisions
    for i, dec in enumerate(decisions):
        if dec is not None:
            plt.scatter(i, y_vals[dec], color=colors[dec], s=10)
            
    plt.yticks([0, 1, 2], ['KEEP', 'ADAPT', 'HOLD'])
    plt.title(f"SUS Controller Dry Run Decisions: {name}")
    plt.xlabel("Timestep")
    
    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys())
        
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig(f'results/figures/dry_run_{name.replace(" ", "_")}.png')
    plt.close()

def dry_run_scenario(name, X_train, X_stream, y_stream):
    print(f"\n--- Running Dry Run for {name} ---")
    
    # Fit Static Detector
    detector = IsolationForestDetector()
    detector.fit(X_train)
    
    # Init Drift Monitor & Controller
    drift_monitor = DriftMonitor(X_train)
    controller = SUSController(anomaly_threshold=0.58, drift_threshold=0.75)
    
    decisions = []
    
    for i, x in enumerate(X_stream):
        # 1. Get Anomaly Score (Static)
        score = detector.score(x.reshape(1, -1))[0]
        
        # 2. Get Drift Score (KS Stat)
        drift_score = drift_monitor.update_and_check(x)
        
        # 3. Controller decides (only after buffer fills)
        if i >= 2 * config.WINDOW_SIZE:
            decision = controller.decide(score, drift_score)
        else:
            decision = None
            
        decisions.append(decision)
        
    # Filter out None for summary
    valid_decisions = [d for d in decisions if d is not None]
    counts = Counter(valid_decisions)
    total = len(valid_decisions)
    
    print(f"Decisions made (excluding first {2 * config.WINDOW_SIZE} buffer steps):")
    for k in ['KEEP', 'ADAPT', 'HOLD']:
        c = counts.get(k, 0)
        print(f"  {k}: {c} ({(c/total)*100:.1f}%)")
        
    plot_decisions(name, decisions, y_stream)

def main():
    data_real = load_and_preprocess(config.DATA_FILE)
    X_train = data_real['X_train']
    
    # Scenarios
    X_stream_A = data_real['X_stream'][:173]
    y_stream_A = data_real['y_stream'][:173]
    
    X_stream_D = data_real['X_stream']
    y_stream_D = data_real['y_stream']
    
    X_stream_B = inject_gradual_drift(X_stream_A, drift_rate=0.001)
    y_stream_B = y_stream_A
    
    dry_run_scenario("Scenario_A", X_train, X_stream_A, y_stream_A)
    dry_run_scenario("Scenario_B", X_train, X_stream_B, y_stream_B)
    dry_run_scenario("Scenario_D", X_train, X_stream_D, y_stream_D)

if __name__ == '__main__':
    main()

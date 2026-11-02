import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.preprocess import load_and_preprocess
from src.adapt import NaiveAdaptiveDetector
from src.threshold_adapt import ThresholdAdaptiveDetector

def main():
    data = load_and_preprocess(config.DATA_FILE)
    X_train = data['X_train']
    X_stream_A = data['X_stream']
    y_stream = data['y_stream']
    t_stream = data['t_stream']
    
    # 1. Naive Adaptive
    naive = NaiveAdaptiveDetector(update_interval=10) # fast updates to see effect
    naive.fit(X_train)
    
    naive_scores = []
    naive_thresholds = []
    
    for i in range(len(X_stream_A)):
        pred, score = naive.update_and_predict(X_stream_A[i])
        naive_scores.append(score)
        naive_thresholds.append(naive.threshold)
        
    # 2. Threshold Adaptive
    thresh_adapt = ThresholdAdaptiveDetector()
    thresh_adapt.fit(X_train)
    
    thresh_scores = []
    thresh_thresholds = []
    
    for i in range(len(X_stream_A)):
        pred, score = thresh_adapt.update_and_predict(X_stream_A[i])
        thresh_scores.append(score)
        thresh_thresholds.append(thresh_adapt.threshold)
        
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Naive Plot
    axes[0].plot(t_stream, naive_scores, label='Anomaly Score', color='blue', alpha=0.6)
    axes[0].plot(t_stream, naive_thresholds, label='Dynamic Threshold', color='orange')
    axes[0].fill_between(t_stream, 0, 1, where=(y_stream == 1), color='red', alpha=0.3, transform=axes[0].get_xaxis_transform(), label='True Anomaly')
    axes[0].set_title('Naive Adaptive on Scenario A')
    axes[0].legend()
    
    # Threshold Plot
    axes[1].plot(t_stream, thresh_scores, label='Anomaly Score', color='blue', alpha=0.6)
    axes[1].plot(t_stream, thresh_thresholds, label='Dynamic Threshold', color='orange')
    axes[1].fill_between(t_stream, 0, 1, where=(y_stream == 1), color='red', alpha=0.3, transform=axes[1].get_xaxis_transform(), label='True Anomaly')
    axes[1].set_title('Threshold Adaptive on Scenario A')
    axes[1].legend()
    
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/scenario_a_debug_collapse.png')
    print("Saved diagnostic plot to results/figures/scenario_a_debug_collapse.png")

if __name__ == '__main__':
    main()

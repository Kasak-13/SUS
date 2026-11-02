import json
import urllib.request
import pandas as pd
import numpy as np
import sys

BASE = 'http://127.0.0.1:5000'

def fetch_json(url):
    with urllib.request.urlopen(f'{BASE}{url}') as res:
        return json.loads(res.read().decode('utf-8'))

print("=== 3. Deep System Logic & Scenario Verification ===")

# Test Scenario A
data_A = fetch_json('/api/scenario/A')
print(f"\n[Scenario A (Stable Clean)]")
print(f"  Length: {len(data_A)} steps")
true_anoms_A = sum(1 for d in data_A if d['is_true_anomaly'] == 1)
hold_A = sum(1 for d in data_A if d['decision'] == 'HOLD')
adapt_A = sum(1 for d in data_A if d['decision'] == 'ADAPT')
print(f"  True Anomalies: {true_anoms_A} (Expected: 0)")
print(f"  HOLD Triggers: {hold_A} (Expected: 0)")
print(f"  ADAPT Triggers: {adapt_A} (Harmless thermal adaptation: {adapt_A} steps)")
assert true_anoms_A == 0, "Scenario A should have 0 true anomalies"
assert hold_A == 0, "Scenario A should have 0 false HOLD triggers"
print("  => PASS: Scenario A is clean with 0 false anomaly emergency locks.")

# Test Scenario B
data_B = fetch_json('/api/scenario/B')
print(f"\n[Scenario B (Gradual Drift)]")
print(f"  Length: {len(data_B)} steps")
adapt_B = sum(1 for d in data_B if d['decision'] == 'ADAPT')
hold_B = sum(1 for d in data_B if d['decision'] == 'HOLD')
max_drift_B = max(d['drift_score'] for d in data_B)
print(f"  Max KS Drift Score: {max_drift_B:.4f} (Expected: >0.75)")
print(f"  ADAPT Triggers: {adapt_B} (Expected: >30 steps)")
print(f"  HOLD Triggers: {hold_B} (Expected: 0 steps)")
assert max_drift_B > 0.75, "Scenario B must register significant KS drift"
assert adapt_B > 30, "Scenario B must actively trigger ADAPT"
assert hold_B == 0, "Scenario B with real adaptation must have 0 false HOLD triggers"
print("  => PASS: Scenario B correctly adapts to drift without false alarms.")

# Test Scenario D
data_D = fetch_json('/api/scenario/D')
print(f"\n[Scenario D (Physical Anomaly)]")
print(f"  Length: {len(data_D)} steps")
anom_steps_D = [d['step'] for d in data_D if d['is_true_anomaly'] == 1]
hold_steps_D = [d['step'] for d in data_D if d['decision'] == 'HOLD']
print(f"  True Anomaly Start: Step {anom_steps_D[0]} (Total anomaly steps: {len(anom_steps_D)})")
print(f"  First HOLD Trigger: Step {hold_steps_D[0]} (Total HOLD steps: {len(hold_steps_D)})")

# Check Anomaly Contamination in Naive Adaptive
# True anomaly points where Naive predicted 0 (Normal)
missed_by_naive = sum(1 for d in data_D if d['is_true_anomaly'] == 1 and d['naive_pred'] == 0)
naive_miss_rate = (missed_by_naive / len(anom_steps_D)) * 100
print(f"  Naive Missed Anomalies: {missed_by_naive} / {len(anom_steps_D)} ({naive_miss_rate:.1f}% Missed due to Contamination)")

# Check SUS Protection
sus_hold_rate = (len(hold_steps_D) / (len(data_D) - 100)) * 100
print(f"  SUS Protection HOLD Rate: {sus_hold_rate:.1f}% of active stream protected")
assert len(hold_steps_D) > 300, "SUS must engage HOLD during contamination"
assert naive_miss_rate > 80.0, "Naive Adaptive must exhibit catastrophic contamination"
print("  => PASS: Scenario D proves SUS locks down in HOLD while Naive Adaptive gets contaminated.")

# Test Benchmark Table
benchmark_data = fetch_json('/api/benchmark')
print(f"\n[Benchmark Table API]")
print(f"  Total Rows: {len(benchmark_data)} records")
scenarios_covered = set(r['Scenario'] for r in benchmark_data)
detectors_covered = set(r['Detector'] for r in benchmark_data)
print(f"  Scenarios Covered: {scenarios_covered}")
print(f"  Detectors Covered: {detectors_covered}")
assert len(scenarios_covered) == 3, "Benchmark must cover all 3 scenarios"
assert len(detectors_covered) == 4, "Benchmark must cover all 4 detectors"
print("  => PASS: Stage 8 Benchmark Table is complete and verified.")

print("\n================ ALL INTEGRITY CHECKS PASSED SUCCESSFULLY! ================\n")

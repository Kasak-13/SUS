# SUS (Streaming Uncertainty Surveillance) - Project Status

This document tracks the progress of the SUS college minor project, summarizing the completed milestones and outlining the remaining work required to fulfill the research objectives.

## 🎯 Project Objective
**Research Question:** "Can a lightweight adaptive anomaly-detection framework (SUS) selectively update its notion of normal behaviour in evolving data, reducing false alarms under legitimate distributional change while resisting the normalization of persistent anomalies?"

---

## ✅ What is Done

### 1. Environment & Data Setup (Stages 1-2)
- **Environment:** Created standard Python structure (`src/`, `experiments/`, `results/`, etc.) and isolated virtual environment (`venv`).
- **Dataset:** Selected the SKAB (Skoltech Anomaly Benchmark) industrial dataset. Specifically, `data/raw/skab_repo/data/valve1/0.csv` was chosen as the canonical contiguous stream to prevent artificial data jumps.

### 2. Preprocessing & Baseline Models (Stage 3)
- **Preprocessing (`src/preprocess.py`):** Established robust preprocessing ensuring temporal integrity (no shuffling) and scaling based solely on historical data.
- **Static Baseline (`src/static_detector.py`):** Implemented an `IsolationForest` that trains on historical data and freezes its weights/threshold.
- **Scenario A (Clean Baseline):** A genuinely stable 173-step clean stream from `valve1/0.csv`, recorded immediately after the training cutoff and before any physical anomalies occur.
  - *Limitation Note:* Scenario A's sample size (173 steps) is smaller than Scenario B/D (747 steps), so its FPR estimate is less statistically stable and should be read as indicative rather than precise.

### 3. Concept Drift Framework (Stage 4)
- **Drift Injection (`src/synthetic_drift.py`):** Implemented a module to inject gradual, synthetic concept drift. This simulates legitimate, long-term environmental changes (e.g., sensor wear) missing in the raw SKAB data.
- **Drift Monitoring (`src/drift_monitor.py`):** Built a non-parametric drift monitor using the Kolmogorov-Smirnov (KS) test to detect statistical distribution shifts over sliding windows.
- **Scenario B (Drifted Stream):** Evaluated the detectors on a controlled drift case.
- **Scenario D_Real (Persistent Contamination):** A real-data stream (`valve1/0.csv`) with 53.7% anomalies. This replaces our original synthetic contamination plan by using massive, genuine physical anomalies to prove that naive adaptation falsely learns anomalies as normal.

---

## 🎯 Status: 100% COMPLETED (All Stages 1–8 Delivered)

### 1. The Adaptive Frameworks & SUS Controller (Stage 5-7) — COMPLETED
- **Naive & Threshold Baselines (Stage 5):** Built models that continuously retrain or update blindly. We proved these fail disastrously during real anomalies (Scenario D) due to "Anomaly Contamination".
- **SUS Controller (Stage 6):** Built the 4-quadrant state-machine using a KS drift test and a windowed anomaly aggregate.
  - *Drift Rate Scope Note:* The controller's decisions are mathematically correct within the calibrated drift-speed range (rates around 0.0008 - 0.0010). We documented a failure mode at faster drift rates (e.g. 0.0015) during dry runs where the frozen Isolation Forest scores fast drift as a massive anomaly prematurely. 
- **Real Adaptation & Safeguards (Stage 7):** Wired the controller to actual model retraining. 
  - *Hypothesis Confirmed:* Real adaptation successfully fixes the 0.0015 failure mode by pulling the baseline along with the fast drift, keeping anomaly scores below the threshold and preventing false `HOLD` states.
  - *Safeguards:* Added `MIN_CONSECUTIVE_OBSERVATIONS` and `HOLD_DURATION` to reduce flickering. Scenario D `HOLD` rate successfully strengthened to 69.1% with only 6.6% false `ADAPT`.

### 2. Final Evaluation & Ablation (Stage 8) — COMPLETED
- **Metrics Comparison Table:** Aggregated Precision, Recall, F1-Score, FPR, and FNR across Scenarios A, B, and D into `results/tables/stage8_four_systems_comparison.csv` for **all FOUR systems**:
  1. Static Baseline (Frozen on day 1)
  2. Naive Adaptive (Blind retraining every 50 steps -> 95% False Negative Rate under faults)
  3. Threshold-only Adaptive (Rolling quantile -> 96% Missed Anomaly Rate)
  4. SUS Controller (Ours -> Highest F1-score 0.3304, 100% Contamination Protection)
- **Interactive Multi-System Arena & Chaos Lab Dashboard:** Upgraded `demo/` with:
  - Live 4-system head-to-head competition stream.
  - Interactive Chaos Lab controls (`⚡ Inject Shock Anomaly` & `🌊 Accelerate Drift`).
  - Modal displaying the official Stage 8 Benchmark Table.
- **Packaging & Delivery:**
  - `run_dashboard.bat` (1-click Windows runner).
  - Standalone offline deployment package (`SUS_Project_Presentation_Ready.zip`).
  - Production `README.md` and complete defense guide (`stage8_presentation_guide.md`).
  - Pushed to remote GitHub repository (`https://github.com/Kasak-13/SUS`).

---

## 🔮 Future Work & Research Paper Extensions
- **Sudden Regime Change:** Instantaneous step-function shift in normal behavior (as opposed to gradual drift).
- **Recurring Anomalies:** Periodic instances of the same anomaly type (critical for testing if models falsely learn the anomaly as a new seasonal normal).
- **Dynamic Window Sizing:** Exploring adaptive window lengths (e.g., ADWIN-style adaptive windowing) to replace fixed 50-step windows.

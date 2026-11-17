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

## 🎯 Status: 100% COMPLETED (All Stages 1–8 Delivered & Stabilized)

### 1. The Adaptive Frameworks & SUS Controller (Stage 5-7) — COMPLETED
- **Naive & Threshold Baselines (Stage 5):** Built models that continuously retrain or update blindly. We proved these fail disastrously during real anomalies (Scenario D) due to "Anomaly Contamination".
- **SUS Controller (Stage 6):** Built the 4-quadrant state-machine using a KS drift test and a windowed anomaly aggregate.
  - *Drift Rate Scope Note:* The controller's decisions are mathematically correct within the calibrated drift-speed range (rates around 0.0008 - 0.0010). We documented a failure mode at faster drift rates (e.g. 0.0015) during dry runs where the frozen Isolation Forest scores fast drift as a massive anomaly prematurely. 
- **Real Adaptation & Safeguards (Stage 7):** Wired the controller to actual model retraining. 
  - *Hypothesis Confirmed:* Real adaptation successfully fixes the 0.0015 failure mode by pulling the baseline along with the fast drift, keeping anomaly scores below the threshold and preventing false `HOLD` states.
  - *Safeguards:* Added `MIN_CONSECUTIVE_OBSERVATIONS` (3) and `HOLD_DURATION` (10) to eliminate flickering. Scenario D `HOLD` rate successfully strengthened with emergency lockdown during physical faults.

### 2. Comprehensive Rigor & Finalization Audit (Stage 8) — COMPLETED
- **Detector Correctness & Threshold Versioning:** Fixed scoring model version consistency ensuring each point is scored and evaluated using the threshold of the model that generated its score (`sus.last_pred`). Excluded `HOLD` observations from polluting the retraining buffer `buffer_X`.
- **Training Buffer Purity & Admission Safeguards:**
  - `KEEP` Anomaly Admission Gate: Only observations with $pred == 0$ enter `buffer_X` during steady-state or warmup. Detected anomaly spikes are rejected.
  - `ADAPT` Buffer Purity Gate: Before retraining is executed, the entire candidate `buffer_X` is re-evaluated under the active model. If contamination exceeds `ANOMALY_FRACTION_THRESHOLD` (10%), adaptation is aborted and the controller engages `HOLD` with full hysteresis countdown.
- **Independent Benchmark Audit & Bit-for-Bit Reproducibility:** Verified across 3 independent benchmark runs that SUS produces 100% bit-for-bit identical confusion matrices and metrics.
- **Final Stage 8 Comparative Benchmark:**
  - **Scenario A (Clean Baseline):** SUS FPR = **2.31%** (Static: 5.20%, Naive: 0.58%, Threshold: 3.47%).
  - **Scenario B (Gradual Drift):** SUS FPR = **3.47%** (Static: 6.36%, Naive: 1.73%, Threshold: 4.62%).
  - **Scenario D (Physical Anomaly):**
    - SUS Controller: **F1 = 0.4848**, **Recall = 41.65%**, **Precision = 57.99%**, **FNR = 58.35%**, **FPR = 34.97%**.
    - Naive Adaptive: **F1 = 0.0939**, **Recall = 4.99%**, **FNR = 95.01%** (Catastrophic Anomaly Contamination).
    - Threshold Adaptive: **F1 = 0.0749**, **Recall = 3.99%**, **FNR = 96.01%** (Threshold Collapse).
    - Static (Frozen): **F1 = 0.3204**, **Recall = 21.70%**, **FNR = 78.30%** (Static Model Blindness).
- **Presentation Dashboard & APIs:** Flask application precomputes Scenarios A, B, and D, providing 60 FPS timeline scrubbing, live chaos injection, and a parameter-driven simulation endpoint (`POST /api/custom_simulation`).

---

## 🔬 Academic Limitations & Future Research Directions

1. **Stationarity Recovery After Physical Disturbances (Future Work):**
   - In Scenario D, 100% of remaining false positives (121 steps) occur after the physical valve fault ends (step 573 onwards).
   - Empirical analysis across SKAB records revealed that physical valve cycling induces permanent hydraulic equilibrium shifts (two-sample KS distance = 1.0000) while variance collapses back to baseline ($0.77 \sim 1.12$).
   - The SUS controller prioritizes safety by remaining locked in `HOLD`, refusing to normalize the shifted regime without operator confirmation.
   - *Future Work:* Developing an automated, variance-stabilized stationarity-recovery mechanism that safely resets the baseline only after a prolonged quiescent post-fault period.

2. **Underlying Detector Sensitivity:**
   - The baseline Isolation Forest achieves ~21.7% base recall on raw SKAB physical faults. SUS prevents contamination from collapsing this recall to 4.99%, effectively doubling fault detection to 41.65%. Future work could integrate deep generative or semi-supervised base models into the SUS gating architecture.



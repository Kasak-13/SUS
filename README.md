# SUS: Streaming Uncertainty Surveillance

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: Stage 8 Complete](https://img.shields.io/badge/Status-Stage%208%20Evaluated-brightgreen.svg)](PROJECT_STATUS.md)
[![Benchmark: SKAB Testbed](https://img.shields.io/badge/Benchmark-SKAB%20Water%20Testbed-orange.svg)](https://github.com/waikato-datamining/skab)

> **A Dual-Gated, Contamination-Resilient Anomaly Detection Framework for Non-Stationary Industrial Cyber-Physical Streams.**

---

## 📌 Executive Summary

Modern industrial Cyber-Physical Systems (CPS)—such as water distribution testbeds, manufacturing lines, and power grids—operate under continuous, non-stationary environmental changes (e.g., thermal warmup, sensor oxidation, mechanical wear-and-tear). Standard streaming anomaly detection approaches face a fatal dilemma:

1. **Static Baseline Models (Frozen):** Weights remain frozen on day one. As sensors naturally drift, static models trigger endless **False Positives (False Alarms)**, inducing catastrophic operator alarm fatigue.
2. **Naive Adaptive Models (Blind Continuous Retraining):** Models retrain blindly on recent observations. When real, persistent physical damage occurs, the model absorbs the dirty observations into its training baseline—a disaster known as **Anomaly Contamination**. The model normalizes the damaged state and suffers devastating **False Negatives (Missed Failures)**.

**Streaming Uncertainty Surveillance (SUS)** resolves this dilemma through a lightweight, mathematically interpretable **Dual-Gated State Machine Controller**. By simultaneously decoupling **Distributional Drift Evidence** from **Anomaly Density Evidence**, SUS safely permits baseline model adaptation during legitimate operational drift, while immediately engaging a lockdown **HOLD Emergency Brake** the millisecond physical damage is detected.

---

## 🔬 Core Theoretical Framework

```
                          Streaming Sensor Point x_t
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   Isolation Forest Scoring                 Sequential 100-Step Window
                 │                                         │
        Windowed Extreme Density             Two-Sample Kolmogorov-Smirnov
       (Fraction > Anom Threshold)                   (KS Statistic)
                 │                                         │
                 ▼                                         ▼
          [ Anomaly Evidence ]                      [ Drift Evidence ]
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │   4-Quadrant Controller   │
                        │     State-Machine Gating  │
                        └─────────────┬─────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
      [ KEEP ]                    [ ADAPT ]                   [ HOLD ]
   Normal Operation;           Safe Distributional         Dense Physical Fault;
   Preserve Baseline           Drift; Gated Retrain        Emergency Lockdown;
                                                           Block Contamination!
```

### 1. Drift Evidence: Sequential Kolmogorov-Smirnov (KS) Test
Unlike historical-distance drift monitors that compare streaming data against an outdated initial training set, SUS tracks **distributional momentum**. It continuously evaluates adjacent sliding windows ($W_{t-1}$ vs. $W_t$) using the non-parametric two-sample Kolmogorov-Smirnov test:
$$D_{KS} = \sup_x |F_{W_{t-1}}(x) - F_{W_t}(x)|$$
A drift score crossing $D_{KS} > 0.75$ indicates significant distributional change across features.

### 2. Anomaly Evidence: Windowed Quantile Fraction
Rather than relying on noisy, single-point blips, SUS computes the **rolling fraction** of recent observations (over a 50-step window) that violate the strict 99th-percentile anomaly threshold ($\tau = 0.58$) established by the underlying Isolation Forest:
$$\text{AnomalyFraction}_t = \frac{1}{W} \sum_{i=0}^{W-1} \mathbb{I}(s_{t-i} > 0.58)$$
An anomaly fraction $> 0.10$ signifies dense, persistent anomaly activity.

### 3. The 4-Quadrant Decision Policy
* **Quadrant 1 (Low Drift, Low Anomaly) $\rightarrow$ `KEEP`:** Normal operation. No update needed.
* **Quadrant 2 (Low Drift, High Anomaly) $\rightarrow$ `KEEP`:** Isolated sensor spike or transient electrical glitch. Alert is logged, but model retraining is blocked.
* **Quadrant 3 (High Drift, Low Anomaly) $\rightarrow$ `ADAPT`:** Legitimate operational change (e.g. ambient temperature rise or sensor aging). The controller gates and permits the Isolation Forest to update its baseline.
* **Quadrant 4 (High Drift, High Anomaly) $\rightarrow$ `HOLD`:** Persistent physical equipment fault (e.g., valve failure). The controller slams on the emergency brake, locking model weights to prevent **Anomaly Contamination**.

### 4. Stateful Hysteresis Safeguards
To prevent decision flickering at threshold boundaries, the controller enforces:
* **`MIN_CONSECUTIVE_OBSERVATIONS = 3`:** Requires 3 consecutive `ADAPT` proposals before retraining is executed.
* **`HOLD_DURATION = 10`:** Enforces an active 10-step lockdown window once `HOLD` is engaged, preventing contaminated trailing data from poisoning the baseline.

---

## 📊 Empirical Evaluation: Stage 8 Benchmark Results

The framework was benchmarked across all 4 architectural paradigms on the Skoltech Anomaly Benchmark (SKAB) hydraulic water testbed:

| Scenario | Architecture | Precision | Recall | F1-Score | False Alarm Rate (FPR) | Missed Fault Rate (FNR) | Contamination Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A (Stable Clean)** | 1. Static (Frozen) | 0.00% | 0.00% | 0.0000 | 5.20% | 0.00% | Clean |
| **A (Stable Clean)** | 2. Naive Adaptive | 0.00% | 0.00% | 0.0000 | 0.58% | 0.00% | Clean |
| **A (Stable Clean)** | 3. Threshold Adaptive | 0.00% | 0.00% | 0.0000 | 3.47% | 0.00% | Clean |
| **A (Stable Clean)** | **4. SUS Controller (Ours)** | **0.00%** | **0.00%** | **0.0000** | **2.89%** | **0.00%** | **Clean Baseline** |
| **B (Gradual Drift)** | 1. Static (Frozen) | 0.00% | 0.00% | 0.0000 | 6.36% ⚠️ | 0.00% | High False Alarms |
| **B (Gradual Drift)** | 2. Naive Adaptive | 0.00% | 0.00% | 0.0000 | 1.73% | 0.00% | Blind Adaptation |
| **B (Gradual Drift)** | 3. Threshold Adaptive | 0.00% | 0.00% | 0.0000 | 4.62% | 0.00% | Threshold Drift |
| **B (Gradual Drift)** | **4. SUS Controller (Ours)** | **0.00%** | **0.00%** | **0.0000** | **2.89%** | **0.00%** | **Safe Gated ADAPT** |
| **D (Physical Fault)** | 1. Static (Frozen) | 61.27% | 21.70% | 0.3204 | 15.90% | 78.30% | Low Recall |
| **D (Physical Fault)** | 2. Naive Adaptive | 80.00% | 4.99% | 0.0939 | 1.45% | **95.01% ❌** | **CATASTROPHIC CONTAMINATION** |
| **D (Physical Fault)** | 3. Threshold Adaptive | 61.54% | 3.99% | 0.0749 | 2.89% | **96.01% ❌** | **Threshold Collapse** |
| **D (Physical Fault)** | **4. SUS Controller (Ours)** | **55.95%** | **23.44%** | **0.3304** | **21.39%** | **76.56%** | **PROTECTED (HOLD Shield)** |

### Key Empirical Findings:
1. **The Contamination Catastrophe:** When exposed to real physical valve faults (Scenario D), Naive Adaptive retrains on contaminated data, suffering a **95.01% False Negative Rate** (misses 95% of real physical faults!).
2. **SUS Resilience:** SUS engages `HOLD` on 69.2% of the active fault stream, completely preserving baseline purity and achieving the **highest overall F1-Score (0.3304)** and highest recall.

---

## 🖥️ Interactive Presentation Dashboard & Chaos Lab

The project features a full-stack, glassmorphism dark-mode web application for live demonstrations:

* **Live 4-System Arena:** Streams sensor telemetry and updates all four competing models simultaneously in real time.
* **Interactive Chaos Lab:**
  * **`⚡ Inject Shock Anomaly`:** Injects an instant 5-step valve/pump failure live on stream. Watch Naive Adaptive fail while SUS engages `HOLD` immediately.
  * **`🌊 Accelerate Sensor Drift`:** Ramps continuous distributional drift to demonstrate live `ADAPT` tracking.
  * **`↺ Reset Injections`:** Restores clean baseline data instantly.
* **Timeline Scrubber & Quick Jumps:**
  * Scrub to any timestep (0 to 747).
  * `⚡ Jump to Fault (Step 175)` button skips the clean warmup and jumps directly into the physical fault region.
* **Stage 8 Benchmark Modal:** 1-click modal rendering the official empirical comparison table.

---

## 🚀 Setup & Execution Guide

### Option 1: 1-Click Launch (Windows)
1. Ensure **Python 3.10+** is installed on your system.
2. Simply double-click **`run_dashboard.bat`** in the project root.
3. The script will automatically:
   * Verify your Python installation.
   * Create a virtual environment (`venv`).
   * Install all required dependencies (`requirements.txt`).
   * Launch the dashboard and open your browser to **`http://localhost:5000`**.

---

### Option 2: Manual Terminal Setup (Windows, macOS, Linux)

#### 1. Clone the Repository
```bash
git clone https://github.com/Kasak-13/SUS.git
cd SUS
```

#### 2. Create and Activate Virtual Environment
* **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Launch the Interactive Dashboard
```bash
python demo/app.py
```
Open your web browser and navigate to:
👉 **`http://localhost:5000`**

---

## 🧪 Reproducing Experiments & Benchmark Evaluations

All experimental scripts are self-contained in the `experiments/` directory:

* **Run Full Stage 8 Four-System Benchmark:**
  ```bash
  python experiments/stage8_final_evaluation.py
  ```
  *(Outputs results to `results/tables/stage8_four_systems_comparison.csv`)*

* **Run End-to-End System Integrity & Contamination Test:**
  ```bash
  python experiments/test_system_integrity.py
  ```

* **Run Sequential Drift Calibration:**
  ```bash
  python experiments/calibrate.py
  ```

* **Verify UI & DOM Consistency:**
  ```bash
  python experiments/test_ui_consistency.py
  ```

---

## 📂 Repository Structure

```
SUS/
├── demo/                           # Web presentation dashboard
│   ├── app.py                      # Flask backend & scenario streaming API
│   └── static/
│       ├── index.html              # Modern glassmorphism UI & Arena layout
│       ├── script.js               # Real-time streaming charts & live chaos controls
│       └── style.css               # Dark-mode styling, animations, and modal CSS
├── src/                            # Core research framework modules
│   ├── config.py                   # Global constants and threshold definitions
│   ├── controller.py               # SUS 4-Quadrant state machine & safeguards
│   ├── drift_monitor.py            # Sequential Kolmogorov-Smirnov drift test
│   ├── sus_detector.py             # Integrated detector (Model + Controller + Drift)
│   ├── static_detector.py          # Baseline frozen Isolation Forest
│   ├── adapt.py                    # Naive adaptive detector (blind retraining)
│   ├── threshold_adapt.py          # Rolling quantile threshold baseline
│   ├── preprocess.py               # SKAB data preprocessing & train/stream splits
│   └── synthetic_drift.py          # Controlled gradual drift injection utilities
├── experiments/                    # Evaluation & reproduction scripts
│   ├── stage8_final_evaluation.py  # Stage 8 official 4-system comparative benchmark
│   ├── test_system_integrity.py    # Deep system logic assertion test
│   ├── test_ui_consistency.py     # HTML/JS DOM ID consistency validator
│   ├── calibrate.py                # Threshold calibration script
│   └── create_project_zip.py       # Deployment packaging script
├── data/                           # Skoltech Anomaly Benchmark (SKAB) records
├── results/                        # Generated tables and evaluation artifacts
│   └── tables/
│       └── stage8_four_systems_comparison.csv
├── requirements.txt                # Version-locked Python package dependencies
├── run_dashboard.bat               # 1-click Windows automated runner
├── README_SETUP.md                 # Standalone offline deployment instructions
├── PROJECT_STATUS.md               # Engineering roadmap and stage milestones
└── README.md                       # Master project documentation
```

---

## 📖 Citation & Academic Context

If you build upon or reference the SUS framework in your research, please cite:

```bibtex
@article{sus2026stream,
  title={SUS: Streaming Uncertainty Surveillance for Contamination-Resilient Anomaly Detection in Non-Stationary Industrial Cyber-Physical Systems},
  author={Kasak Tolani and Contributors},
  journal={GitHub Repository},
  year={2026},
  url={https://github.com/Kasak-13/SUS}
}
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


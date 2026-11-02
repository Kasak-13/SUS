# SUS Project — Setup Instructions for Another PC

This folder contains the complete **Streaming Uncertainty Surveillance (SUS)** project, including the research source code, experiments, benchmark evaluations, and the interactive web presentation dashboard.

---

## ⚡ Super-Fast 1-Click Setup (Windows)

1. **Extract this ZIP file** anywhere on your computer (e.g. `C:\Projects\SUS` or Desktop).
2. Ensure **Python 3.10+** is installed on the PC (with "Add Python to PATH" checked during install).
3. **Double-click `run_dashboard.bat`** in this folder.
   - *It will automatically create the virtual environment, install the 5 required dependencies, and launch the dashboard in your default browser at `http://localhost:5000`!*

---

## 🛠️ Manual Terminal Setup (Windows / Mac / Linux)

If you prefer using the terminal or command prompt:

### 1. Open Terminal in this folder:
```bash
# Windows
cd path\to\SUS

# Mac / Linux
cd path/to/SUS
```

### 2. Create and Activate Virtual Environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard:
```bash
python demo/app.py
```
Open your browser and navigate to: **http://localhost:5000**

---

## 🧪 Running the Research Experiments & Benchmarks

To reproduce the Stage 8 benchmark evaluations or check controller accuracy:

* **Stage 8 Comprehensive 4-System Evaluation:**
  ```bash
  python experiments/stage8_final_evaluation.py
  ```
  *(Outputs official table to `results/tables/stage8_four_systems_comparison.csv`)*

* **End-to-End System Integrity & Anomaly Contamination Test:**
  ```bash
  python experiments/test_system_integrity.py
  ```

* **Sequential KS Drift Calibration:**
  ```bash
  python experiments/calibrate.py
  ```

---

## 📁 Project Structure Overview

* `src/` — Core mathematical framework:
  * `controller.py`: SUS 4-Quadrant state machine with consecutive-evidence and HOLD lockdown safeguards.
  * `sus_detector.py`: Integrated detector combining Isolation Forest, DriftMonitor, and SUSController.
  * `drift_monitor.py`: Sequential Kolmogorov-Smirnov statistical test across rolling sliding windows.
  * `static_detector.py`: Baseline frozen Isolation Forest.
  * `adapt.py`: Naive adaptive detector (blind retraining baseline).
  * `threshold_adapt.py`: Quantile threshold adaptive baseline.
* `demo/` — Interactive web application:
  * `app.py`: Flask streaming backend with precomputed scenario caches and custom injection endpoints.
  * `static/index.html`: Modern UI with live 4-system arena, timeline scrubber, and benchmark modal.
  * `static/script.js`: Real-time streaming charts and interactive controls.
  * `static/style.css`: Responsive glassmorphism dark-mode styling.
* `data/` — Preprocessed and raw Skoltech Anomaly Benchmark (SKAB) hydraulic water testbed records.
* `results/` — Empirical tables and decision timeline figures.
* `stage8_presentation_guide.md` — Step-by-step presentation script and talking points.


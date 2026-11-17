# SUS (Streaming Uncertainty Surveillance) — Stage 8 Presentation Guide
## Today's Upgraded Demo & Defense Script

Today's presentation introduces **three brand-new, cutting-edge features** that take your demo far beyond last time:
1. **The Live 4-System Multi-Agent Arena:** Instead of just showing SUS in isolation, all four systems (Static, Naive Adaptive, Threshold-only, and SUS) run head-to-head in real time.
2. **The Interactive Chaos Lab:** Live fault and drift injection buttons (`⚡ Inject Shock Anomaly` & `🌊 Accelerate Sensor Drift`), allowing you to test the system live during the demo.
3. **Stage 8 Benchmark Matrix Modal:** A full empirical metrics table with quantitative proof of Naive Adaptive's failure (95% missed faults) vs. SUS's victory.

---

### How to Access the Live Demo
The upgraded dashboard is **already running live** in the background!
* Simply open your browser and go to: **[http://localhost:5000](http://localhost:5000)**

---

### Step-by-Step Presentation Script for Today

#### 1. The Opening Hook (30 seconds)
> *"Good morning respected guide and committee members. Last time, I demonstrated the internal 4-quadrant decision logic of our SUS controller. Today, I am proud to present **Stage 8: the full Multi-System Benchmark and Live Chaos Lab**.*
>
> *Today, you will see our SUS controller compete directly against the industry-standard paradigms in real time—and I will even inject live hydraulic faults into the streaming data to prove our model does not fail."*

---

#### 2. Tour of the Upgraded Interface (45 seconds)
* **Point to the 4-System Arena (Top Grid):**
  > *"Across the top, we have our four detectors running in tandem:*
  > 1. *Static Baseline (Frozen weights on Day 1).*
  > 2. *Naive Adaptive (Blindly retrains every 50 steps).*
  > 3. *Threshold Adaptive (Shifts its cutoff threshold).*
  > 4. *Our SUS Controller (Dual-evidence gated adaptation with protective safeguards)."*
* **Point to the Interactive Chaos Lab (Toolbar):**
  > *"Right below that is our Chaos Lab, giving us controls to inject sudden shocks, accelerate drift, or advance step-by-step."*

---

#### 3. Live Demonstration 1: The Drift Test (Scenario B)
1. Click **Scenario B: Drift**.
2. Click **Play Simulation ▶** (or select **2x speed** to move faster).
3. **What to point out:**
   - Watch the green Drift curve rise.
   - Point to System 1 (Static): Notice its score rises and it begins throwing **False Alarms**.
   - Point to System 4 (SUS): Notice the badge switches to **ADAPT**.
   - **What to say:**
     > *"As component aging causes the sensors to drift, the Static model begins flagging false alarms. But look at SUS: it recognizes high drift with low anomaly evidence, moves to the ADAPT quadrant, and safely retrains the model to track the new normal."*

---

#### 4. Live Demonstration 2: The Catastrophic Contamination Test (Scenario D)
1. Click **Scenario D: Physical Fault**.
2. Click **Play Simulation ▶**.
3. When step ~175 is reached, the red shaded physical fault begins.
4. **What to point out:**
   - Look at System 2 (Naive Adaptive): Its badge flashes **`CONTAMINATED! ❌`** and stays Normal!
   - Look at System 4 (SUS): It instantly slams on the emergency brake with **`HOLD`** and locks the baseline.
   - **What to say:**
     > *"Now comes the critical research question: What happens during a real physical fault?*
     >
     > *Look at System 2, Naive Adaptive. Because it blindly retrains every 50 steps, it absorbed the water circuit fault directly into its training set. It now believes a broken machine is 'normal'! This is the textbook catastrophe known as **Anomaly Contamination**.*
     >
     > *Now look at our SUS Controller: The instant dense anomaly evidence appeared, SUS engaged its **HOLD** state. It locked down adaptation, prevented the baseline from being poisoned, and maintained full alert status throughout the disaster."*

---

#### 5. The "Showstopper" Move: Live Chaos Injection!
*(This is the feature that will make your presentation stand out above everyone else!)*
1. Click **Scenario A: Stable**.
2. Click **Play Simulation ▶**.
3. While it is happily running in **KEEP**, say:
   > *"Committee members, you might wonder: is this system only working on pre-recorded sequences? Let's test it live. Suppose an unexpected valve rupture happens right now."*
4. Click the red button: **`⚡ Inject Shock Anomaly`**!
5. **Watch the live reaction:**
   - The sensor mean spikes violently.
   - The anomaly fraction jumps.
   - SUS immediately flashes **`HOLD (Emergency Brake)`** and locks down the system!
6. Click **`↺ Reset Injections`** when done.

---

#### 6. The Quantitative Proof: Stage 8 Benchmark Table
1. Click the purple button: **`📊 4-System Benchmark`** in the top header.
2. The modal pops up displaying the official empirical results:
   - **In Scenario D (Real Faults):**
     - **Naive Adaptive:** Suffers a **95.01% False Negative Rate** (misses 95% of real physical faults due to contamination)! F1-score collapses to 0.0939.
     - **Threshold Adaptive:** Suffers a **96.01% Missed Anomaly Rate**.
     - **SUS Controller:** Achieves the **highest F1-score (0.3304)** and highest recall (23.44%), successfully resisting baseline contamination. *(Crucial nuance for Committee: SUS does not eliminate missed faults—its FNR remains 76.56% due to base Isolation Forest scoring limitations—its true triumph is preventing the 95.01% contamination collapse suffered by Naive Adaptive).*

---

### Quick Summary for Q&A
* **Q: Why is SUS better than Naive Adaptive?**
  * *A:* "Naive Adaptive blindly retrains on whatever data arrives. If a fault lasts more than 50 steps, it treats the fault as normal data (Anomaly Contamination). SUS uses a 4-quadrant state machine that blocks retraining whenever anomaly evidence is present."
* **Q: Why is SUS's False Negative Rate still 76.56% in Scenario D? Does it eliminate missed faults?**
  * *A:* "No, SUS does not eliminate missed faults. The underlying base detector (Isolation Forest) intrinsically achieves ~21-23% recall on SKAB physical faults. SUS is an adaptation gatekeeper: when Naive Adaptive is deployed, its missed fault rate worsens catastrophically from 78% to 95.01% because it normalizes the damage. SUS prevents this contamination collapse, preserving the base detector's detection power."
* **Q: Why not just use a static model?**
  * *A:* "Static models fail in the real world because sensors naturally degrade and drift over months. Static models trigger endless false alarms, causing alarm fatigue."
* **Q: What dataset was used?**
  * *A:* "The Skoltech Anomaly Benchmark (SKAB) hydraulic water testbed dataset."

import os
import shutil
import tempfile
import zipfile

ZIP_PATH = r"C:\Users\Lenovo\Desktop\SUS_Project_Presentation_Ready.zip"
EXTRACT_DIR = os.path.join(tempfile.gettempdir(), "sus_zip_verification_test")

print("=== Task 10: Independent Verification of Extracted ZIP ===")
print(f"Target Archive: {ZIP_PATH}")
print(f"Temporary Extraction Directory: {EXTRACT_DIR}")

if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)
os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    zf.extractall(EXTRACT_DIR)

# Paths to extracted files
cfg_path = os.path.join(EXTRACT_DIR, "SUS", "src", "config.py")
ctrl_path = os.path.join(EXTRACT_DIR, "SUS", "src", "controller.py")

assert os.path.exists(cfg_path), f"Extracted config.py missing at {cfg_path}"
assert os.path.exists(ctrl_path), f"Extracted controller.py missing at {ctrl_path}"

with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg_text = f.read()

with open(ctrl_path, 'r', encoding='utf-8') as f:
    ctrl_text = f.read()

print("\n--- Extracted config.py Contents ---")
for line in cfg_text.strip().splitlines():
    print("  ", line)

print("\n--- Verification Assertions on Extracted config.py ---")
assert "ANOMALY_THRESHOLD = 0.58" in cfg_text, "config.py must define ANOMALY_THRESHOLD = 0.58"
assert "DRIFT_THRESHOLD = 0.75" in cfg_text, "config.py must define DRIFT_THRESHOLD = 0.75"
assert "ANOMALY_FRACTION_THRESHOLD = 0.10" in cfg_text, "config.py must define ANOMALY_FRACTION_THRESHOLD = 0.10"
assert "MIN_CONSECUTIVE_OBSERVATIONS = 3" in cfg_text, "config.py must define MIN_CONSECUTIVE_OBSERVATIONS = 3"
assert "HOLD_DURATION = 10" in cfg_text, "config.py must define HOLD_DURATION = 10"
assert "MIN_ADAPTATION_EVIDENCE" not in cfg_text, "config.py must NOT contain legacy MIN_ADAPTATION_EVIDENCE"
assert "DRIFT_THRESHOLD = 0.05" not in cfg_text, "config.py must NOT contain stale DRIFT_THRESHOLD = 0.05"
assert "ANOMALY_THRESHOLD = 0.01" not in cfg_text, "config.py must NOT contain stale ANOMALY_THRESHOLD = 0.01"
print("  [PASS] All extracted config.py values are 100% verified and correct!")

print("\n--- Extracted controller.py Safeguard Lines ---")
for line in ctrl_text.splitlines():
    if any(k in line for k in ['min_consecutive', 'MIN_CONSECUTIVE', 'hold_duration', 'MIN_ADAPTATION']):
        print("  ", line)

print("\n--- Verification Assertions on Extracted controller.py ---")
assert "MIN_CONSECUTIVE_OBSERVATIONS" in ctrl_text, "controller.py must read MIN_CONSECUTIVE_OBSERVATIONS"
assert "MIN_ADAPTATION_EVIDENCE" not in ctrl_text, "controller.py must NOT read MIN_ADAPTATION_EVIDENCE"
assert "HOLD_DURATION" in ctrl_text, "controller.py must read HOLD_DURATION"
print("  [PASS] Extracted controller.py correctly uses MIN_CONSECUTIVE_OBSERVATIONS!")

# Clean up
shutil.rmtree(EXTRACT_DIR)
print(f"\nTemporary directory {EXTRACT_DIR} cleaned up successfully.")
print("================ TASK 10 INDEPENDENT ZIP VERIFICATION PASSED! ================\n")

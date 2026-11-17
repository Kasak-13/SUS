import os
import zipfile
import tempfile
import shutil

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
zip_path = os.path.join(repo_root, 'SUS_Project_Presentation_Ready.zip')

EXCLUDE_DIRS = {'.git', 'venv', '__pycache__', '.pytest_cache'}
EXCLUDE_EXTS = {'.pyc', '.pyo'}

print(f"Building updated ZIP: {zip_path}")
if os.path.exists(zip_path):
    os.remove(zip_path)

file_count = 0
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(repo_root):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith('.zip') or any(f.endswith(ext) for ext in EXCLUDE_EXTS):
                continue
            abs_path = os.path.join(root, f)
            rel_path = os.path.relpath(abs_path, repo_root)
            zipf.write(abs_path, rel_path)
            file_count += 1

print(f"Archive created with {file_count} files. Size: {os.path.getsize(zip_path)} bytes.")

# Independent Verification: Extract to temporary directory
temp_extract_dir = tempfile.mkdtemp(prefix='sus_zip_verification_')
print(f"Extracting to temporary directory: {temp_extract_dir}")
with zipfile.ZipFile(zip_path, 'r') as zipf:
    zipf.extractall(temp_extract_dir)

# Inspect extracted src/sus_detector.py
extracted_sus_detector = os.path.join(temp_extract_dir, 'src', 'sus_detector.py')
with open(extracted_sus_detector, 'r', encoding='utf-8') as f:
    detector_content = f.read()

assert 'class SUSResult(tuple):' in detector_content, "SUSResult class missing from extracted zip!"
assert 'current_threshold = float(self.anomaly_threshold)' in detector_content, "Versioned threshold capture missing!"
assert 'if decision == \'HOLD\':' in detector_content, "HOLD exclusion safeguard missing!"
assert 'candidate_anom_frac <= self.controller.anomaly_fraction_threshold' in detector_content, "ADAPT buffer purity gate missing!"

# Inspect extracted src/controller.py
extracted_controller = os.path.join(temp_extract_dir, 'src', 'controller.py')
with open(extracted_controller, 'r', encoding='utf-8') as f:
    controller_content = f.read()

assert 'def record_observation(self, anomaly_score):' in controller_content, "record_observation missing from extracted controller!"

# Inspect extracted experiments/test_detector_correctness.py
extracted_test = os.path.join(temp_extract_dir, 'experiments', 'test_detector_correctness.py')
assert os.path.exists(extracted_test), "test_detector_correctness.py missing from extracted zip!"

shutil.rmtree(temp_extract_dir)
print("=> INDEPENDENT ZIP VERIFICATION PASSED: All updated modules present and verified!")

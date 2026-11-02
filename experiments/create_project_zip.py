import os
import zipfile

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ZIP_OUTPUT = os.path.abspath(os.path.join(PROJECT_DIR, '..', 'SUS_Project_Presentation_Ready.zip'))

EXCLUDE_DIRS = {'venv', '__pycache__', '.git', '.idea', '.vscode'}
EXCLUDE_EXTS = {'.pyc', '.zip', '.log'}

print(f">> Packaging SUS project from: {PROJECT_DIR}")
print(f">> Target ZIP file: {ZIP_OUTPUT}")

total_files = 0
total_size = 0

with zipfile.ZipFile(ZIP_OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(PROJECT_DIR):
        # Exclude directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in EXCLUDE_EXTS:
                continue
            if file == 'SUS_Project_Presentation_Ready.zip':
                continue
                
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, PROJECT_DIR)
            arcname = os.path.join('SUS', arcname) # Place inside clean 'SUS' root directory
            
            zipf.write(file_path, arcname)
            total_files += 1
            total_size += os.path.getsize(file_path)

zip_size_mb = os.path.getsize(ZIP_OUTPUT) / (1024 * 1024)
print(f">> Successfully packaged {total_files} files!")
print(f">> Raw content size: {total_size / (1024 * 1024):.2f} MB")
print(f">> Compressed ZIP size: {zip_size_mb:.2f} MB")
print(f">> ZIP location: {ZIP_OUTPUT}")


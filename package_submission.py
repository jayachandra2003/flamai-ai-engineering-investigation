#!/usr/bin/env python3
"""
package_submission.py -- Creates the final clean submission ZIP.
"""

import os
import zipfile

ZIP_NAME = "AI_Team_Intern_Assignment_Submission_FINAL.zip"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ZIP_PATH = os.path.join(BASE_DIR, ZIP_NAME)

# Files & directories to include under 'your-submission/' prefix
INCLUDED_PATHS = [
    "README.md",
    "NOTEBOOK.md",
    "AI_USAGE.md",
    "DEFENSE_PREP.md",
    "requirements.txt",
    "partA",
    "partB",
    "partC"
]

EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd", ".zip", ".DS_Store"}
EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", ".idea", ".vscode"}


def should_include(rel_path):
    parts = rel_path.split(os.sep)
    for p in parts:
        if p in EXCLUDE_DIRS:
            return False
    _, ext = os.path.splitext(rel_path)
    if ext in EXCLUDE_EXTS:
        return False
    return True


def create_zip():
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in INCLUDED_PATHS:
            full_item_path = os.path.join(BASE_DIR, item)
            if os.path.isfile(full_item_path):
                arcname = os.path.join("your-submission", item)
                zipf.write(full_item_path, arcname)
                print(f"Added file: {arcname}")
            elif os.path.isdir(full_item_path):
                for root, dirs, files in os.walk(full_item_path):
                    # Filter out excluded dirs
                    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                    for f in files:
                        file_path = os.path.join(root, f)
                        rel_path = os.path.relpath(file_path, BASE_DIR)
                        if should_include(rel_path):
                            arcname = os.path.join("your-submission", rel_path)
                            zipf.write(file_path, arcname)
                            print(f"Added file: {arcname}")

    print(f"\nSuccessfully created: {ZIP_PATH} ({os.path.getsize(ZIP_PATH)} bytes)")


if __name__ == "__main__":
    create_zip()

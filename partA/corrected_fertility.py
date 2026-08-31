#!/usr/bin/env python3
"""
corrected_fertility.py -- Entrypoint wrapper for Part A3 multi-tokenizer analysis.
Executes partA/corrected_analysis/analyze_metrics.py.
"""
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(SCRIPT_DIR, "corrected_analysis", "analyze_metrics.py")

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, TARGET] + sys.argv[1:]))

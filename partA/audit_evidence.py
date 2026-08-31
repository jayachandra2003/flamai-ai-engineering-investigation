#!/usr/bin/env python3
"""
audit_evidence.py -- Entrypoint wrapper for Part A2 forensic audit.
Executes partA/audit/run_audit.py.
"""
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(SCRIPT_DIR, "audit", "run_audit.py")

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, TARGET] + sys.argv[1:]))

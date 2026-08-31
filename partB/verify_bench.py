#!/usr/bin/env python3
"""
verify_bench.py -- Entrypoint wrapper for Part B capacity and benchmark forensics.
Executes partB/analysis/capacity_analysis.py and partB/analysis/benchmark_analysis.py.
"""
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPACITY = os.path.join(SCRIPT_DIR, "analysis", "capacity_analysis.py")
BENCH = os.path.join(SCRIPT_DIR, "analysis", "benchmark_analysis.py")

if __name__ == "__main__":
    ret1 = subprocess.call([sys.executable, CAPACITY])
    if ret1 != 0:
        sys.exit(ret1)
    ret2 = subprocess.call([sys.executable, BENCH])
    sys.exit(ret2)

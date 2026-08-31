#!/usr/bin/env python3
"""
analysis_scratch.py -- Arithmetic verification for Part C Decision Memo.
"""

# Assignment Facts
weeks_before_launch = 3
gpu_weeks_available = 2
reviewer_hours_per_week = 10
target_languages = ["Hindi", "Kannada", "Tamil", "Telugu", "Bengali", "Marathi"]
covered_languages = ["Hindi", "Kannada"]

# Derived Values
total_reviewer_hours = reviewer_hours_per_week * weeks_before_launch # 30 hours
total_gpu_hours_available = gpu_weeks_available * 7 * 24 # 336 GPU-hours
covered_count = len(covered_languages)
total_lang_count = len(target_languages)
lang_ratio_covered = covered_count / total_lang_count # 2/6 = 0.3333

# Illustrative Scenario Assumptions
minutes_per_pair = 2.0
illustrative_sft_pairs = 1000
illustrative_sft_review_hours = (illustrative_sft_pairs * minutes_per_pair) / 60.0 # 33.33 hours

day1_eval_prompts = 60 # 30 Hindi + 30 Kannada
rounds_tested = 3
day1_total_evals = day1_eval_prompts * rounds_tested # 180 evaluations
day1_review_hours = (day1_total_evals * minutes_per_pair) / 60.0 # 6.0 hours

print(f"Total Reviewer Hours Available: {total_reviewer_hours} hours")
print(f"Total A100 GPU Hours Available: {total_gpu_hours_available} GPU-hours")
print(f"Native Review Coverage: {covered_count} of {total_lang_count} languages ({lang_ratio_covered*100:.1f}%)")
print(f"Illustrative SFT Review Hours (1000 pairs @ 2 min/pair): {illustrative_sft_review_hours:.1f} hours")
print(f"Prompt Engineering Review Hours (60 prompts x 3 rounds @ 2 min/pair): {day1_review_hours:.1f} hours")

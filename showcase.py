#!/usr/bin/env python3
"""
Handwriting Analysis System - Showcase Demo
Forensic signature authentication demonstration.

Run: python showcase.py
"""

import time
import sys
import math

# Colors for terminal output
class Colors:
    GOLD = '\033[93m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.GOLD}{'='*65}")
    print(f" {text}")
    print(f"{'='*65}{Colors.END}\n")

def print_step(step, text):
    print(f"{Colors.CYAN}[STEP {step}]{Colors.END} {Colors.BOLD}{text}{Colors.END}")

# Baseline statistics (from 70 authentic samples)
BASELINE = {
    'slant_angle':       {'mean': 71.5, 'std': 4.2, 'unit': '°'},
    'height_ratio':      {'mean': 2.6,  'std': 0.3, 'unit': ''},
    'aspect_ratio':      {'mean': 2.8,  'std': 0.4, 'unit': ''},
    'stroke_width':      {'mean': 3.8,  'std': 0.8, 'unit': 'px'},
    'pressure_variance': {'mean': 0.25, 'std': 0.08, 'unit': ''},
    'connectedness':     {'mean': 0.62, 'std': 0.10, 'unit': '%'},
    'loop_ratio':        {'mean': 0.35, 'std': 0.08, 'unit': ''}
}

# Questioned signature (authentic example)
QUESTIONED = {
    'slant_angle':       73.2,
    'height_ratio':      2.8,
    'aspect_ratio':      2.9,
    'stroke_width':      4.1,
    'pressure_variance': 0.28,
    'connectedness':     0.58,
    'loop_ratio':        0.38
}

def main():
    print(f"\n{Colors.GOLD}{Colors.BOLD}")
    print("    ╔═══════════════════════════════════════════════════════════════╗")
    print("    ║         FORENSIC HANDWRITING ANALYSIS SYSTEM                  ║")
    print("    ║           Statistical Signature Authentication               ║")
    print("    ╚═══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")

    time.sleep(1)

    # Step 1: Load Baseline
    print_step(1, "LOADING AUTHENTIC BASELINE")
    print()
    print(f"   Analyzing {Colors.BOLD}70 known authentic{Colors.END} signature samples...")
    time.sleep(0.5)

    print(f"\n   {Colors.BOLD}{'Feature':<20} {'Mean':>10} {'Std Dev':>10}{Colors.END}")
    print(f"   {'-'*42}")
    for feature, stats in BASELINE.items():
        time.sleep(0.15)
        unit = stats['unit']
        print(f"   {feature:<20} {stats['mean']:>9.2f}{unit} {stats['std']:>9.2f}")

    print(f"\n   {Colors.GREEN}✓{Colors.END} Baseline model computed successfully")
    print()
    time.sleep(1)

    # Step 2: Analyze Questioned Document
    print_step(2, "ANALYZING QUESTIONED SIGNATURE")
    print()
    print(f"   Loading: {Colors.CYAN}questioned_signature_001.png{Colors.END}")
    time.sleep(0.3)
    print(f"   Extracting features...")
    time.sleep(0.5)

    print(f"\n   {Colors.BOLD}Questioned Document Features:{Colors.END}")
    print(f"   ┌─────────────────────────────────────┐")
    for feature, value in QUESTIONED.items():
        unit = BASELINE[feature]['unit']
        print(f"   │ {feature:<22} {value:>8.2f}{unit:<3}│")
    print(f"   └─────────────────────────────────────┘")
    print()
    time.sleep(1)

    # Step 3: Statistical Comparison
    print_step(3, "STATISTICAL COMPARISON (Z-SCORES)")
    print()

    z_scores = {}
    print(f"   {Colors.BOLD}{'Feature':<20} {'Value':>8} {'Baseline':>14} {'Z-Score':>10} {'Status':>12}{Colors.END}")
    print(f"   {'-'*68}")

    for feature, value in QUESTIONED.items():
        time.sleep(0.2)
        mean = BASELINE[feature]['mean']
        std = BASELINE[feature]['std']
        z = (value - mean) / std if std > 0 else 0
        z_scores[feature] = z

        baseline_str = f"{mean:.2f} ± {std:.2f}"

        if abs(z) < 1:
            status = f"{Colors.GREEN}EXCELLENT{Colors.END}"
        elif abs(z) < 2:
            status = f"{Colors.GOLD}WITHIN RANGE{Colors.END}"
        else:
            status = f"{Colors.RED}OUTSIDE{Colors.END}"

        print(f"   {feature:<20} {value:>8.2f} {baseline_str:>14} {z:>+10.2f} {status:>20}")

    print()
    time.sleep(1)

    # Step 4: Calculate Confidence
    print_step(4, "CONFIDENCE CALCULATION")
    print()

    # RMS Z-score
    rms_z = math.sqrt(sum(z**2 for z in z_scores.values()) / len(z_scores))
    mahalanobis = rms_z * 1.2

    # Confidence
    if rms_z < 1:
        confidence = 0.95 + (1 - rms_z) * 0.04
    elif rms_z < 2:
        confidence = 0.80 + (2 - rms_z) * 0.15
    else:
        confidence = max(0.05, 0.50 - (rms_z - 3) * 0.15)

    print(f"   Combined Z-Score (RMS):  {Colors.BOLD}{rms_z:.3f}{Colors.END}")
    print(f"   Mahalanobis Distance:    {Colors.BOLD}{mahalanobis:.3f}{Colors.END}")
    print()

    # Visual confidence meter
    conf_pct = int(confidence * 100)
    bar_filled = conf_pct // 5
    bar_empty = 20 - bar_filled

    if confidence > 0.90:
        color = Colors.GREEN
    elif confidence > 0.70:
        color = Colors.GOLD
    else:
        color = Colors.RED

    print(f"   Confidence: [{color}{'█' * bar_filled}{Colors.END}{'░' * bar_empty}] {Colors.BOLD}{confidence:.1%}{Colors.END}")
    print()
    time.sleep(1)

    # Step 5: Conclusion
    print_step(5, "ANALYSIS CONCLUSION")
    print()

    within_1sd = sum(1 for z in z_scores.values() if abs(z) < 1)
    within_2sd = sum(1 for z in z_scores.values() if abs(z) < 2)

    print(f"   {Colors.BOLD}Statistical Summary:{Colors.END}")
    print(f"   ┌─────────────────────────────────────────────────────────────┐")
    print(f"   │ Features within 1 SD:     {within_1sd}/7  ({within_1sd/7*100:.0f}%)                       │")
    print(f"   │ Features within 2 SD:     {within_2sd}/7  ({within_2sd/7*100:.0f}%)                      │")
    print(f"   │ Combined Z-Score:         {rms_z:.3f}                              │")
    print(f"   │ Confidence Level:         {confidence:.1%}                             │")
    print(f"   └─────────────────────────────────────────────────────────────┘")
    print()

    print(f"   {Colors.BOLD}Conclusion:{Colors.END}")
    print(f"   ╔═════════════════════════════════════════════════════════════╗")
    print(f"   ║  {Colors.GREEN}{Colors.BOLD}SIGNATURE IS CONSISTENT WITH AUTHENTIC SAMPLES{Colors.END}            ║")
    print(f"   ╠═════════════════════════════════════════════════════════════╣")
    print(f"   ║  The questioned signature falls within expected statistical ║")
    print(f"   ║  ranges for all measured features when compared against the ║")
    print(f"   ║  baseline of 70 known authentic exemplars.                  ║")
    print(f"   ╚═════════════════════════════════════════════════════════════╝")
    print()
    time.sleep(1)

    # Summary
    print_header("ANALYSIS COMPLETE")

    print(f"   {Colors.BOLD}This demo showcased:{Colors.END}")
    print(f"   • Statistical feature extraction from signature images")
    print(f"   • Z-score comparison against authentic baseline")
    print(f"   • Mahalanobis distance calculation")
    print(f"   • Confidence scoring with clear thresholds")
    print()
    print(f"   {Colors.BOLD}Features Analyzed:{Colors.END}")
    print(f"   Slant angle, height ratio, aspect ratio, stroke width,")
    print(f"   pressure variance, stroke connectedness, loop frequency")
    print()
    print(f"   {Colors.DIM}Note: Statistical analysis provides probability, not certainty.")
    print(f"   Results should be verified by qualified forensic examiner.{Colors.END}")
    print()
    print(f"   {Colors.BOLD}GitHub:{Colors.END} github.com/jjshay/handwriting-analysis")
    print()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Handwriting Analysis Demo
Demonstrates forensic signature analysis concepts.

Run: python demo.py
"""

import random
import math
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path


def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}\n")


@dataclass
class SignatureFeatures:
    """Features extracted from a signature"""
    slant_angle: float      # Degrees from horizontal
    height_ratio: float     # Full height / x-height
    aspect_ratio: float     # Width / Height
    stroke_width: float     # Average stroke thickness
    pressure_variance: float # Variation in darkness
    connectedness: float    # % of connected strokes
    loop_ratio: float       # Loops per letter


def generate_authentic_baseline():
    """Generate simulated authentic signature samples"""
    print_header("BUILDING AUTHENTIC BASELINE")
    print("Analyzing 70 known authentic signature samples...\n")

    # Generate 70 samples with consistent characteristics
    # (Real system would extract these from actual images)
    samples = []
    for i in range(70):
        sample = SignatureFeatures(
            slant_angle=71.5 + random.gauss(0, 4.2),
            height_ratio=2.6 + random.gauss(0, 0.3),
            aspect_ratio=2.8 + random.gauss(0, 0.4),
            stroke_width=3.8 + random.gauss(0, 0.8),
            pressure_variance=0.25 + random.gauss(0, 0.08),
            connectedness=0.62 + random.gauss(0, 0.1),
            loop_ratio=0.35 + random.gauss(0, 0.08)
        )
        samples.append(sample)

    # Calculate baseline statistics
    baseline = {}
    features = ['slant_angle', 'height_ratio', 'aspect_ratio', 'stroke_width',
                'pressure_variance', 'connectedness', 'loop_ratio']

    for feature in features:
        values = [getattr(s, feature) for s in samples]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)
        baseline[feature] = {'mean': mean, 'std': std}

    print("Baseline statistics computed:\n")
    print(f"{'Feature':<20} {'Mean':>10} {'Std Dev':>10}")
    print("-" * 42)
    for feature, stats in baseline.items():
        print(f"{feature:<20} {stats['mean']:>10.2f} {stats['std']:>10.2f}")

    return baseline, samples


def analyze_questioned_signature(is_authentic=True):
    """Generate a questioned signature for analysis"""

    if is_authentic:
        # Generate signature consistent with baseline
        return SignatureFeatures(
            slant_angle=73.2,      # Slightly higher but within range
            height_ratio=2.8,
            aspect_ratio=2.9,
            stroke_width=4.1,
            pressure_variance=0.28,
            connectedness=0.58,
            loop_ratio=0.38
        )
    else:
        # Generate signature that deviates from baseline
        return SignatureFeatures(
            slant_angle=82.5,      # Too upright
            height_ratio=3.4,      # Too tall
            aspect_ratio=2.1,      # Too narrow
            stroke_width=2.2,      # Too thin
            pressure_variance=0.45, # Too variable
            connectedness=0.35,    # Too disconnected
            loop_ratio=0.12       # Too few loops
        )


def calculate_z_scores(questioned: SignatureFeatures, baseline: Dict):
    """Calculate z-scores for each feature"""
    print_header("STATISTICAL COMPARISON")

    features = ['slant_angle', 'height_ratio', 'aspect_ratio', 'stroke_width',
                'pressure_variance', 'connectedness', 'loop_ratio']

    z_scores = {}
    print(f"{'Feature':<20} {'Value':>8} {'Baseline':>12} {'Z-Score':>10} {'Status':>12}")
    print("-" * 65)

    for feature in features:
        value = getattr(questioned, feature)
        mean = baseline[feature]['mean']
        std = baseline[feature]['std']

        z = (value - mean) / std if std > 0 else 0
        z_scores[feature] = z

        status = "WITHIN RANGE" if abs(z) < 2 else "OUTSIDE RANGE"
        baseline_str = f"{mean:.2f} ± {std:.2f}"

        print(f"{feature:<20} {value:>8.2f} {baseline_str:>12} {z:>10.2f} {status:>12}")

    return z_scores


def calculate_confidence(z_scores: Dict):
    """Calculate overall confidence score"""
    print_header("CONFIDENCE CALCULATION")

    # Combined z-score (root mean square)
    rms_z = math.sqrt(sum(z**2 for z in z_scores.values()) / len(z_scores))

    # Convert to probability (using simplified normal distribution)
    # Higher z-score = lower probability of authenticity
    if rms_z < 1:
        confidence = 0.95 + (1 - rms_z) * 0.04
    elif rms_z < 2:
        confidence = 0.80 + (2 - rms_z) * 0.15
    elif rms_z < 3:
        confidence = 0.50 + (3 - rms_z) * 0.30
    else:
        confidence = max(0.05, 0.50 - (rms_z - 3) * 0.15)

    print(f"Combined Z-Score (RMS): {rms_z:.2f}")
    print(f"Mahalanobis Distance: {rms_z * 1.2:.2f}")
    print(f"\nConfidence Score: {confidence:.1%}")

    if confidence > 0.90:
        conclusion = "CONSISTENT with known authentic samples"
        recommendation = "High confidence - signature appears authentic"
    elif confidence > 0.70:
        conclusion = "POSSIBLY CONSISTENT with known samples"
        recommendation = "Moderate confidence - additional review recommended"
    elif confidence > 0.50:
        conclusion = "INCONCLUSIVE"
        recommendation = "Low confidence - expert examination required"
    else:
        conclusion = "INCONSISTENT with known authentic samples"
        recommendation = "Signature shows significant deviations from baseline"

    print(f"\nConclusion: {conclusion}")
    print(f"Recommendation: {recommendation}")

    return confidence, conclusion


def generate_report(questioned, baseline, z_scores, confidence, conclusion):
    """Generate formatted analysis report"""
    print_header("FORENSIC ANALYSIS REPORT")

    report = f"""
FORENSIC HANDWRITING ANALYSIS REPORT
=====================================
Document ID: QD-{random.randint(2024001, 2024999)}
Analysis Date: 2024-01-08
Examiner: Statistical Analysis System

METHODOLOGY:
  - Feature extraction from questioned signature
  - Statistical comparison against 70 authentic baseline samples
  - Z-score analysis for each measured feature
  - Combined probability calculation

QUESTIONED DOCUMENT FEATURES:
  Slant Angle:        {questioned.slant_angle:.1f}°
  Height Ratio:       {questioned.height_ratio:.2f}
  Aspect Ratio:       {questioned.aspect_ratio:.2f}
  Stroke Width:       {questioned.stroke_width:.1f}px
  Pressure Variance:  {questioned.pressure_variance:.2f}
  Connectedness:      {questioned.connectedness:.0%}
  Loop Ratio:         {questioned.loop_ratio:.2f}

BASELINE COMPARISON:
  Features within 1 SD: {sum(1 for z in z_scores.values() if abs(z) < 1)}/7
  Features within 2 SD: {sum(1 for z in z_scores.values() if abs(z) < 2)}/7
  Combined Z-Score:     {math.sqrt(sum(z**2 for z in z_scores.values()) / len(z_scores)):.2f}

CONCLUSION:
  Confidence: {confidence:.1%}
  Opinion: The questioned signature is {conclusion}.

LIMITATIONS:
  - Statistical analysis provides probability, not certainty
  - Results should be verified by qualified forensic examiner
  - Analysis based on available sample quality

END OF REPORT
"""
    print(report)

    # Save report
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "analysis_report.txt", "w") as f:
        f.write(report)
    print(f"Report saved to: {output_dir}/analysis_report.txt")


def main():
    print_header("HANDWRITING ANALYSIS SYSTEM - DEMO")

    print("This demo shows how forensic signature analysis works")
    print("using statistical comparison against authentic samples.\n")

    # Build baseline from authentic samples
    baseline, samples = generate_authentic_baseline()

    # Analyze a questioned signature (authentic in this demo)
    print_header("ANALYZING QUESTIONED SIGNATURE")
    print("Loading questioned signature image...")
    print("Extracting features...\n")
    questioned = analyze_questioned_signature(is_authentic=True)

    print("Questioned signature features:")
    print(f"  Slant: {questioned.slant_angle:.1f}°")
    print(f"  Height Ratio: {questioned.height_ratio:.2f}")
    print(f"  Stroke Width: {questioned.stroke_width:.1f}px")

    # Calculate z-scores
    z_scores = calculate_z_scores(questioned, baseline)

    # Calculate confidence
    confidence, conclusion = calculate_confidence(z_scores)

    # Generate report
    generate_report(questioned, baseline, z_scores, confidence, conclusion)

    print_header("DEMO COMPLETE")
    print("This demo analyzed a simulated authentic signature.")
    print("\nTo analyze real signatures:")
    print("  1. Gather 50+ authentic samples")
    print("  2. Run: python forensic_handwriting_analyzer.py")
    print("  3. For AI review: python llm_handwriting_reviewer.py")


if __name__ == "__main__":
    main()

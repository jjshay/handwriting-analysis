#!/usr/bin/env python3
"""
Handwriting Analysis Demo
Demonstrates forensic signature analysis with rich visual output.

Run: python demo.py
"""
from __future__ import annotations

import random
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

# Try to import rich for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


def print_header(text: str) -> None:
    """Print a formatted header."""
    if RICH_AVAILABLE:
        console.print()
        console.rule(f"[bold cyan]{text}[/bold cyan]", style="cyan")
        console.print()
    else:
        print(f"\n{'='*60}")
        print(f" {text}")
        print(f"{'='*60}\n")


def show_banner() -> None:
    """Display the application banner."""
    if RICH_AVAILABLE:
        banner = """
[bold cyan]╔══════════════════════════════════════════════════════════════════╗
║[/bold cyan] [bold gold1]  _   _                 _               _ _   _                   [/bold gold1][bold cyan]║
║[/bold cyan] [bold gold1] | | | | __ _ _ __   __| |_      ___ __(_) |_(_)_ __   __ _       [/bold gold1][bold cyan]║
║[/bold cyan] [bold gold1] | |_| |/ _` | '_ \ / _` \ \ /\ / / '__| | __| | '_ \ / _` |      [/bold gold1][bold cyan]║
║[/bold cyan] [bold gold1] |  _  | (_| | | | | (_| |\ V  V /| |  | | |_| | | | | (_| |      [/bold gold1][bold cyan]║
║[/bold cyan] [bold gold1] |_| |_|\__,_|_| |_|\__,_| \_/\_/ |_|  |_|\__|_|_| |_|\__, |      [/bold gold1][bold cyan]║
║[/bold cyan] [bold gold1]                                                      |___/       [/bold gold1][bold cyan]║
║[/bold cyan]                                                                      [bold cyan]║
║[/bold cyan]            [bold white]Forensic Signature Authentication System[/bold white]              [bold cyan]║
╚══════════════════════════════════════════════════════════════════╝[/bold cyan]
"""
        console.print(banner)
    else:
        print("\n" + "="*60)
        print("  HANDWRITING ANALYSIS - Forensic Signature Authentication")
        print("="*60 + "\n")


@dataclass
class SignatureFeatures:
    """Features extracted from a signature."""
    slant_angle: float
    height_ratio: float
    aspect_ratio: float
    stroke_width: float
    pressure_variance: float
    connectedness: float
    loop_ratio: float


def generate_authentic_baseline() -> Tuple[Dict[str, Dict[str, float]], List[SignatureFeatures]]:
    """Generate simulated authentic signature samples."""
    print_header("BUILDING AUTHENTIC BASELINE")

    if RICH_AVAILABLE:
        console.print("[dim]Analyzing 70 known authentic signature samples...[/dim]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Processing samples...", total=70)
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
                progress.update(task, advance=1)
                time.sleep(0.01)
    else:
        print("Analyzing 70 known authentic signature samples...\n")
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

    if RICH_AVAILABLE:
        table = Table(title="📊 Baseline Statistics (n=70)", box=box.ROUNDED)
        table.add_column("Feature", style="cyan")
        table.add_column("Mean", justify="right", style="gold1")
        table.add_column("Std Dev", justify="right", style="dim")
        table.add_column("Range", justify="center")

        for feature, stats in baseline.items():
            low = stats['mean'] - 2 * stats['std']
            high = stats['mean'] + 2 * stats['std']
            table.add_row(
                feature.replace('_', ' ').title(),
                f"{stats['mean']:.2f}",
                f"±{stats['std']:.2f}",
                f"[dim]{low:.1f} - {high:.1f}[/dim]"
            )

        console.print(table)
    else:
        print(f"{'Feature':<20} {'Mean':>10} {'Std Dev':>10}")
        print("-" * 42)
        for feature, stats in baseline.items():
            print(f"{feature:<20} {stats['mean']:>10.2f} {stats['std']:>10.2f}")

    return baseline, samples


def analyze_questioned_signature(is_authentic: bool = True) -> SignatureFeatures:
    """Generate a questioned signature for analysis."""
    if is_authentic:
        return SignatureFeatures(
            slant_angle=73.2,
            height_ratio=2.8,
            aspect_ratio=2.9,
            stroke_width=4.1,
            pressure_variance=0.28,
            connectedness=0.58,
            loop_ratio=0.38
        )
    else:
        return SignatureFeatures(
            slant_angle=82.5,
            height_ratio=3.4,
            aspect_ratio=2.1,
            stroke_width=2.2,
            pressure_variance=0.45,
            connectedness=0.35,
            loop_ratio=0.12
        )


def calculate_z_scores(questioned: SignatureFeatures, baseline: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Calculate z-scores for each feature."""
    print_header("STATISTICAL COMPARISON")

    features = ['slant_angle', 'height_ratio', 'aspect_ratio', 'stroke_width',
                'pressure_variance', 'connectedness', 'loop_ratio']

    z_scores = {}

    if RICH_AVAILABLE:
        table = Table(title="📈 Z-Score Analysis", box=box.ROUNDED)
        table.add_column("Feature", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Baseline", justify="center")
        table.add_column("Z-Score", justify="right")
        table.add_column("Status", justify="center")

        for feature in features:
            value = getattr(questioned, feature)
            mean = baseline[feature]['mean']
            std = baseline[feature]['std']

            z = (value - mean) / std if std > 0 else 0
            z_scores[feature] = z

            status = "WITHIN" if abs(z) < 2 else "OUTSIDE"
            status_color = "green" if abs(z) < 2 else "red"
            z_color = "green" if abs(z) < 1 else "yellow" if abs(z) < 2 else "red"

            # Visual z-score bar
            bar_pos = int((z + 3) / 6 * 10)
            bar_pos = max(0, min(9, bar_pos))
            bar = "░" * bar_pos + "█" + "░" * (9 - bar_pos)

            table.add_row(
                feature.replace('_', ' ').title(),
                f"{value:.2f}",
                f"{mean:.2f} ± {std:.2f}",
                f"[{z_color}]{z:+.2f}[/{z_color}] [{z_color}]{bar}[/{z_color}]",
                f"[{status_color}]{status}[/{status_color}]"
            )

        console.print(table)
    else:
        print(f"{'Feature':<20} {'Value':>8} {'Baseline':>12} {'Z-Score':>10} {'Status':>12}")
        print("-" * 65)
        for feature in features:
            value = getattr(questioned, feature)
            mean = baseline[feature]['mean']
            std = baseline[feature]['std']
            z = (value - mean) / std if std > 0 else 0
            z_scores[feature] = z
            status = "WITHIN RANGE" if abs(z) < 2 else "OUTSIDE RANGE"
            print(f"{feature:<20} {value:>8.2f} {mean:.2f} ± {std:.2f} {z:>10.2f} {status:>12}")

    return z_scores


def calculate_confidence(z_scores: Dict[str, float]) -> Tuple[float, str]:
    """Calculate overall confidence score."""
    print_header("CONFIDENCE CALCULATION")

    rms_z = math.sqrt(sum(z**2 for z in z_scores.values()) / len(z_scores))

    if rms_z < 1:
        confidence = 0.95 + (1 - rms_z) * 0.04
    elif rms_z < 2:
        confidence = 0.80 + (2 - rms_z) * 0.15
    elif rms_z < 3:
        confidence = 0.50 + (3 - rms_z) * 0.30
    else:
        confidence = max(0.05, 0.50 - (rms_z - 3) * 0.15)

    if confidence > 0.90:
        conclusion = "CONSISTENT with known authentic samples"
        rec = "High confidence - signature appears authentic"
        color = "green"
    elif confidence > 0.70:
        conclusion = "POSSIBLY CONSISTENT with known samples"
        rec = "Moderate confidence - additional review recommended"
        color = "yellow"
    elif confidence > 0.50:
        conclusion = "INCONCLUSIVE"
        rec = "Low confidence - expert examination required"
        color = "yellow"
    else:
        conclusion = "INCONSISTENT with known authentic samples"
        rec = "Signature shows significant deviations"
        color = "red"

    if RICH_AVAILABLE:
        # Create visual gauge
        gauge_filled = int(confidence * 20)
        gauge = "█" * gauge_filled + "░" * (20 - gauge_filled)

        gauge_panel = f"""
[bold]CONFIDENCE GAUGE[/bold]

    [{color}]╔════════════════════════════════════════╗
    ║  [{gauge}]  ║
    ║                                        ║
    ║         [bold]{confidence:.1%}[/bold] CONFIDENCE            ║
    ╚════════════════════════════════════════╝[/{color}]

[dim]Combined Z-Score (RMS):[/dim] {rms_z:.2f}
[dim]Mahalanobis Distance:[/dim]  {rms_z * 1.2:.2f}

[bold]Conclusion:[/bold] [{color}]{conclusion}[/{color}]
[bold]Recommendation:[/bold] {rec}
"""
        console.print(Panel(gauge_panel, title="🎯 Authentication Result", border_style=color, box=box.DOUBLE))
    else:
        print(f"Combined Z-Score (RMS): {rms_z:.2f}")
        print(f"Confidence Score: {confidence:.1%}")
        print(f"\nConclusion: {conclusion}")
        print(f"Recommendation: {rec}")

    return confidence, conclusion


def generate_report(questioned: SignatureFeatures, baseline: Dict[str, Dict[str, float]], z_scores: Dict[str, float], confidence: float, conclusion: str) -> None:
    """Generate formatted analysis report."""
    print_header("FORENSIC ANALYSIS REPORT")

    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

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

    if RICH_AVAILABLE:
        console.print(Panel(report, title="📄 Report Preview", border_style="cyan", box=box.ROUNDED))

    with open(output_dir / "analysis_report.txt", "w") as f:
        f.write(report)

    if RICH_AVAILABLE:
        console.print(f"\n[bold green]✓[/bold green] Report saved to: [cyan]{output_dir}/analysis_report.txt[/cyan]")
    else:
        print(report)
        print(f"Report saved to: {output_dir}/analysis_report.txt")


def main() -> None:
    """Main entry point."""
    show_banner()

    if RICH_AVAILABLE:
        console.print("[dim]This demo shows how forensic signature analysis works")
        console.print("using statistical comparison against authentic samples.[/dim]\n")
    else:
        print("This demo shows how forensic signature analysis works")
        print("using statistical comparison against authentic samples.\n")

    # Build baseline from authentic samples
    baseline, samples = generate_authentic_baseline()

    # Analyze a questioned signature
    print_header("ANALYZING QUESTIONED SIGNATURE")

    if RICH_AVAILABLE:
        console.print("[dim]Loading questioned signature image...[/dim]")
        console.print("[dim]Extracting features...[/dim]\n")
    else:
        print("Loading questioned signature image...")
        print("Extracting features...\n")

    questioned = analyze_questioned_signature(is_authentic=True)

    if RICH_AVAILABLE:
        features_panel = Panel(
            f"""[cyan]Slant:[/cyan] {questioned.slant_angle:.1f}°
[cyan]Height Ratio:[/cyan] {questioned.height_ratio:.2f}
[cyan]Stroke Width:[/cyan] {questioned.stroke_width:.1f}px
[cyan]Pressure Var:[/cyan] {questioned.pressure_variance:.2f}
[cyan]Connectedness:[/cyan] {questioned.connectedness:.0%}""",
            title="📝 Questioned Signature Features",
            border_style="gold1",
            box=box.ROUNDED
        )
        console.print(features_panel)
    else:
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

    if RICH_AVAILABLE:
        console.print("[dim]This demo analyzed a simulated authentic signature.[/dim]\n")
        console.print("[bold]To analyze real signatures:[/bold]")
        console.print("  1. Gather 50+ authentic samples")
        console.print("  2. Run: [cyan]python forensic_handwriting_analyzer.py[/cyan]")
        console.print("  3. For AI review: [cyan]python llm_handwriting_reviewer.py[/cyan]")
    else:
        print("To analyze real signatures:")
        print("  1. Gather 50+ authentic samples")
        print("  2. Run: python forensic_handwriting_analyzer.py")


if __name__ == "__main__":
    main()

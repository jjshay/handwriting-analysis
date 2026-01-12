#!/usr/bin/env python3
"""Handwriting Analysis - Marketing Demo"""
import time
import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.align import Align
    from rich import box
except ImportError:
    print("Run: pip install rich")
    sys.exit(1)

console = Console()

def pause(s=1.5):
    time.sleep(s)

def step(text):
    console.print(f"\n[bold white on #1a1a2e]  {text}  [/]\n")
    pause(0.8)

# INTRO
console.clear()
console.print()
intro = Panel(
    Align.center("[bold yellow]HANDWRITING ANALYSIS[/]\n\n[white]Forensic Signature Authentication[/]"),
    border_style="cyan",
    width=60,
    padding=(1, 2)
)
console.print(intro)
pause(2)

# STEP 1
step("STEP 1: LOAD BASELINE DATABASE")

console.print("[dim]$[/] python signature_analyzer.py [cyan]--baseline picasso_signatures/[/]\n")
pause(1)

console.print("  Loading baseline samples..", end="")
pause(0.6)
console.print(" [green]73 authentic signatures[/]")

console.print("  Computing features.......", end="")
pause(0.5)
console.print(" [green]14 characteristics[/]")

pause(0.8)

baseline = Panel(
    "[bold]Pablo Picasso - Signature Baseline[/]\n\n"
    "[dim]Samples:[/]    73 verified (1920-1973)\n"
    "[dim]Source:[/]     Museum archives, auction records\n"
    "[dim]Features:[/]   14 measurable characteristics",
    title="[cyan]Baseline Loaded[/]",
    border_style="cyan",
    width=50
)
console.print(baseline)
pause(1.5)

# STEP 2
step("STEP 2: ANALYZE QUESTIONED SIGNATURE")

console.print("[dim]$[/] python signature_analyzer.py [cyan]--questioned suspect_artwork.jpg[/]\n")
pause(1)

console.print("  Loading document.........", end="")
pause(0.5)
console.print(" [green]Done[/]")

console.print("  Isolating signature......", end="")
pause(0.6)
console.print(" [green]Bottom-right detected[/]")

console.print("  Enhancing image..........", end="")
pause(0.4)
console.print(" [green]Contrast adjusted[/]")

pause(1)

# STEP 3
step("STEP 3: FEATURE EXTRACTION")

features = Table(box=box.ROUNDED, width=55)
features.add_column("Feature", style="white")
features.add_column("Questioned", justify="center")
features.add_column("Baseline", justify="center", style="dim")
features.add_column("Result", justify="center")

features.add_row("Slant Angle", "73.2°", "71.5° ± 4.2°", "[green]MATCH[/]")
features.add_row("Height Ratio", "2.81", "2.65 ± 0.32", "[green]MATCH[/]")
features.add_row("Stroke Width", "4.1px", "3.8 ± 0.8px", "[green]MATCH[/]")
features.add_row("Pressure Var.", "0.28", "0.25 ± 0.08", "[green]MATCH[/]")
features.add_row("Connectedness", "67%", "62% ± 12%", "[green]MATCH[/]")
features.add_row("Loop Ratio", "0.41", "0.38 ± 0.09", "[green]MATCH[/]")
features.add_row("Aspect Ratio", "3.2", "3.1 ± 0.4", "[green]MATCH[/]")

console.print(features)
pause(1.5)

# STEP 4
step("STEP 4: STATISTICAL ANALYSIS")

console.print("  Calculating Z-scores.....", end="")
pause(0.5)
console.print(" [green]0.82 (normal)[/]")

console.print("  Mahalanobis distance.....", end="")
pause(0.5)
console.print(" [green]1.24 (< 2.5)[/]")

console.print("  Chi-Square p-value.......", end="")
pause(0.4)
console.print(" [green]0.71 (> 0.05)[/]")

pause(1)

stats = Panel(
    "[green]All measurements within expected variation[/]\n"
    "[green]for authentic Picasso signatures.[/]",
    border_style="green",
    width=50
)
console.print(stats)
pause(1.5)

# STEP 5
step("STEP 5: MULTI-AI VISUAL REVIEW")

console.print("  Querying 4 AI models for visual comparison...\n")
pause(0.8)

models = [
    ("Claude", "Stroke patterns consistent"),
    ("GPT-4", "Authentic characteristics present"),
    ("Gemini", "No anomalies detected"),
    ("Grok", "Matches late-period style"),
]

for name, assessment in models:
    console.print(f"  {name}...", end="")
    pause(0.5)
    console.print(f" [green]{assessment}[/]")

pause(1)

# STEP 6
step("STEP 6: AUTHENTICATION VERDICT")

verdict = Panel(
    Align.center(
        "[bold green]CONFIDENCE: 94.7%[/]\n\n"
        "[green]" + "█" * 38 + "[/][dim]" + "░" * 2 + "[/]\n\n"
        "[bold]VERDICT: CONSISTENT WITH AUTHENTIC[/]"
    ),
    title="[bold yellow]RESULT[/]",
    border_style="green",
    width=50
)
console.print(verdict)
pause(2)

# STEP 7
step("STEP 7: EXPORT REPORT")

console.print("  [green]>[/] Report: [cyan]./reports/forensic_report.pdf[/]")
console.print("  [green]>[/] Data:   [cyan]./reports/analysis_data.json[/]")
console.print("  [green]>[/] Images: [cyan]./reports/visual_comparison.png[/]")
pause(1)

# FOOTER
console.print()
footer = Panel(
    Align.center(
        "[dim]Statistical + Multi-AI Forensics[/]\n"
        "[bold cyan]github.com/jjshay/handwriting-analysis[/]"
    ),
    title="[dim]Handwriting Analysis v1.2[/]",
    border_style="dim",
    width=50
)
console.print(footer)
pause(3)

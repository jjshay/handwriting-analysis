#!/usr/bin/env python3
"""Marketing Demo - Handwriting Analysis"""
import time
import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    console = Console()
except ImportError:
    print("Run: pip install rich")
    sys.exit(1)

def pause(seconds=2):
    time.sleep(seconds)

def clear():
    console.clear()

# SCENE 1: Hook
clear()
console.print("\n" * 5)
console.print("[bold yellow]              IS THAT SIGNATURE REAL?[/bold yellow]", justify="center")
pause(2)

# SCENE 2: Problem
clear()
console.print("\n" * 3)
console.print(Panel("""
[bold red]FAKES COST YOU MONEY:[/bold red]

   • Art forgeries worth billions
   • Document fraud everywhere
   • Human experts expensive
   • Subjective opinions vary

[dim]You need SCIENTIFIC analysis.[/dim]
""", title="❌ Can You Spot a Forgery?", border_style="red", width=60), justify="center")
pause(3)

# SCENE 3: Solution
clear()
console.print("\n" * 3)
console.print(Panel("""
[bold green]FORENSIC-GRADE ANALYSIS:[/bold green]

   ✓ Measures 7 signature features
   ✓ Compares against 70+ samples
   ✓ Statistical Z-score analysis
   ✓ Confidence percentage

[bold]Science, not guesswork.[/bold]
""", title="✅ Handwriting Analysis System", border_style="green", width=60), justify="center")
pause(3)

# SCENE 4: Analysis
clear()
console.print("\n\n")
console.print("[bold cyan]              🔍 ANALYZING SIGNATURE...[/bold cyan]", justify="center")
console.print()
pause(1)

features = [
    ("Slant Angle", "73.2°", "+0.4", "green"),
    ("Height Ratio", "2.8", "+0.7", "green"),
    ("Stroke Width", "4.1px", "+0.4", "green"),
    ("Pressure", "0.28", "+0.4", "green"),
    ("Connectedness", "58%", "-0.4", "green"),
    ("Loop Ratio", "0.38", "+0.4", "green"),
]

table = Table(box=box.ROUNDED, width=55)
table.add_column("Feature", style="cyan")
table.add_column("Measured", justify="center")
table.add_column("Z-Score", justify="center")

for feat, val, z, color in features:
    z_display = f"[{color}]{z}[/{color}]"
    table.add_row(feat, val, z_display)
    console.clear()
    console.print("\n\n")
    console.print("[bold cyan]              🔍 ANALYZING SIGNATURE...[/bold cyan]", justify="center")
    console.print()
    console.print(table, justify="center")
    pause(0.4)

pause(2)

# SCENE 5: Result
clear()
console.print("\n" * 2)
console.print(Panel("""
[bold green]
    ╔═══════════════════════════════════════════╗
    ║                                           ║
    ║         CONFIDENCE: [bold]94.2%[/bold]              ║
    ║                                           ║
    ║    ████████████████████░░░░░░░░░░░░░░░    ║
    ║                                           ║
    ║         [bold]✅ AUTHENTIC[/bold]                    ║
    ║                                           ║
    ╚═══════════════════════════════════════════╝
[/bold green]

[bold]VERDICT:[/bold] Signature is CONSISTENT with authentic samples.

All 7 features within expected range (Z < 2.0)
""", title="📊 ANALYSIS COMPLETE", border_style="green", width=55), justify="center")
pause(3)

# SCENE 6: CTA
clear()
console.print("\n" * 4)
console.print("[bold yellow]           ⭐ VERIFY BEFORE YOU BUY ⭐[/bold yellow]", justify="center")
console.print()
console.print("[bold white]          github.com/jjshay/handwriting-analysis[/bold white]", justify="center")
console.print()
console.print("[dim]                       python demo.py[/dim]", justify="center")
pause(3)

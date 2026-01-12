#!/usr/bin/env python3
"""
Generate animated demo GIF for Handwriting Analysis

This script creates an animated GIF showing the signature analysis process.
Requires: Pillow

Usage: python generate_demo_gif.py
Output: demo.gif
"""

import os
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Installing Pillow...")
    os.system("pip install Pillow")
    from PIL import Image, ImageDraw, ImageFont


def create_frame(size, step, total_steps):
    """Create a single animation frame"""
    img = Image.new('RGB', size, '#0d1117')
    draw = ImageDraw.Draw(img)

    # Colors
    gold = '#ffd700'
    white = '#ffffff'
    green = '#00ff88'
    cyan = '#00d4ff'

    # Title
    draw.text((size[0]//2 - 180, 30), "Handwriting Analysis", fill=gold)
    draw.text((size[0]//2 - 160, 60), "Forensic Signature Verification", fill=white)

    # Animation steps
    steps_text = [
        "Loading baseline samples...",
        "Extracting slant angle features...",
        "Analyzing height ratios...",
        "Measuring stroke characteristics...",
        "Computing pressure patterns...",
        "Calculating Z-scores...",
        "Computing Mahalanobis distance...",
        "Generating confidence score...",
        "Analysis complete!"
    ]

    # Progress bar
    progress = (step + 1) / total_steps
    bar_width = 400
    bar_x = (size[0] - bar_width) // 2
    bar_y = size[1] - 80

    draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + 20],
                   outline=white, width=2)

    fill_width = int(bar_width * progress)
    if fill_width > 0:
        draw.rectangle([bar_x + 2, bar_y + 2, bar_x + fill_width - 2, bar_y + 18],
                       fill=cyan)

    # Draw current step text
    current_text = steps_text[min(step, len(steps_text) - 1)]
    draw.text((size[0]//2 - 140, bar_y - 40), current_text, fill=white)

    # Draw step indicators
    y_pos = 120
    for i, text in enumerate(steps_text):
        if i < step:
            color = green
            prefix = "✓ "
        elif i == step:
            color = gold
            prefix = "→ "
        else:
            color = '#666666'
            prefix = "  "
        draw.text((80, y_pos + i * 25), prefix + text, fill=color)

    # Show confidence result on final frame
    if step == total_steps - 1:
        draw.rectangle([size[0]//2 - 100, size[1] - 130, size[0]//2 + 100, size[1] - 100],
                       fill='#1a3a1a', outline=green)
        draw.text((size[0]//2 - 80, size[1] - 125), "Confidence: 94.7%", fill=green)

    return img


def generate_demo_gif():
    """Generate the demo GIF"""
    size = (600, 450)
    total_steps = 9
    frames = []

    for step in range(total_steps):
        frame = create_frame(size, step, total_steps)
        for _ in range(3):
            frames.append(frame)

    for _ in range(5):
        frames.append(frames[-1])

    output_path = Path(__file__).parent / "demo.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )

    print(f"✓ Demo GIF created: {output_path}")
    print(f"  Size: {size[0]}x{size[1]}")
    print(f"  Frames: {len(frames)}")


if __name__ == "__main__":
    generate_demo_gif()

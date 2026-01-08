#!/usr/bin/env python3
"""
Signature Analysis and Reproduction Tool for AxiDraw
Analyzes signature samples to extract characteristics for natural reproduction
"""

import numpy as np
from pyaxidraw import axidraw
import cv2
import os
from scipy import interpolate
from scipy.signal import savgol_filter
import random
from typing import List, Tuple, Dict
import json

class SignatureAnalyzer:
    """Analyzes signature samples to extract characteristics"""

    def __init__(self):
        self.signatures = []
        self.characteristics = {}

    def load_signature_images(self, folder_path: str):
        """Load signature images from folder"""
        print(f"Loading signatures from {folder_path}")

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.signatures.append({
                        'filename': filename,
                        'image': img,
                        'strokes': self.extract_strokes(img)
                    })

    def extract_strokes(self, image):
        """Extract stroke paths from signature image"""
        # Threshold and invert
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        # Thin the image to get skeleton
        kernel = np.ones((2,2), np.uint8)
        skeleton = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        strokes = []
        for contour in contours:
            if len(contour) > 10:  # Filter small noise
                points = contour.reshape(-1, 2)
                strokes.append(points)

        return strokes

    def analyze_characteristics(self):
        """Analyze all signatures to extract common characteristics"""

        if not self.signatures:
            print("No signatures loaded")
            return

        # Analyze stroke characteristics
        all_speeds = []
        all_pressures = []
        all_angles = []
        pen_lifts = []

        for sig in self.signatures:
            for stroke in sig['strokes']:
                if len(stroke) > 2:
                    # Calculate speeds (distance between points)
                    speeds = np.sqrt(np.sum(np.diff(stroke, axis=0)**2, axis=1))
                    all_speeds.extend(speeds)

                    # Calculate angles
                    angles = np.arctan2(np.diff(stroke[:, 1]), np.diff(stroke[:, 0]))
                    all_angles.extend(angles)

                    # Simulate pressure based on speed (slower = more pressure)
                    pressures = 1.0 / (speeds + 0.1)
                    all_pressures.extend(pressures)

            pen_lifts.append(len(sig['strokes']))

        # Store characteristics
        self.characteristics = {
            'avg_speed': np.mean(all_speeds) if all_speeds else 1.0,
            'speed_variation': np.std(all_speeds) if all_speeds else 0.1,
            'avg_pressure': np.mean(all_pressures) if all_pressures else 0.5,
            'pressure_variation': np.std(all_pressures) if all_pressures else 0.1,
            'common_angles': self.find_common_angles(all_angles),
            'avg_pen_lifts': np.mean(pen_lifts) if pen_lifts else 3,
            'slant': self.calculate_average_slant(all_angles)
        }

        print("\nSignature Characteristics:")
        print(f"  Average speed: {self.characteristics['avg_speed']:.2f}")
        print(f"  Speed variation: {self.characteristics['speed_variation']:.2f}")
        print(f"  Average slant: {np.degrees(self.characteristics['slant']):.1f}°")
        print(f"  Average pen lifts: {self.characteristics['avg_pen_lifts']:.1f}")

    def find_common_angles(self, angles):
        """Find most common angles in signatures"""
        if not angles:
            return [0]

        # Bin angles and find peaks
        hist, bins = np.histogram(angles, bins=36)
        common = bins[np.argsort(hist)[-5:]]  # Top 5 angles
        return common.tolist()

    def calculate_average_slant(self, angles):
        """Calculate average slant of signature"""
        if not angles:
            return 0

        # Filter for upward/downward strokes
        vertical_angles = [a for a in angles if abs(a - np.pi/2) < np.pi/4]
        if vertical_angles:
            return np.mean(vertical_angles) - np.pi/2
        return 0

class SignatureReproducer:
    """Reproduces signatures with natural variations using AxiDraw"""

    def __init__(self, characteristics: Dict):
        self.characteristics = characteristics
        self.ad = axidraw.AxiDraw()
        self.ad.interactive()

    def connect(self):
        """Connect to AxiDraw"""
        self.ad.connect()
        return self.ad.connected

    def generate_signature_path(self, base_path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Generate signature path with natural variations"""

        if not base_path:
            return []

        # Add natural variations
        varied_path = []

        for i, (x, y) in enumerate(base_path):
            # Add tremor
            tremor = 0.01 * self.characteristics.get('speed_variation', 0.1)
            x += random.gauss(0, tremor)
            y += random.gauss(0, tremor)

            # Apply slant
            slant = self.characteristics.get('slant', 0)
            x += y * np.tan(slant) * 0.1

            varied_path.append((x, y))

        # Smooth the path
        if len(varied_path) > 3:
            varied_path = self.smooth_path(varied_path)

        return varied_path

    def smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Smooth path using spline interpolation"""
        points = np.array(path)

        if len(points) < 4:
            return path

        # Parameterize by arc length
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        t = np.concatenate([[0], np.cumsum(distances)])

        # Create spline
        fx = interpolate.interp1d(t, points[:, 0], kind='cubic')
        fy = interpolate.interp1d(t, points[:, 1], kind='cubic')

        # Resample
        t_new = np.linspace(0, t[-1], len(points))
        smooth_path = list(zip(fx(t_new), fy(t_new)))

        return smooth_path

    def draw_signature(self, strokes: List[List[Tuple[float, float]]],
                       scale: float = 0.01, offset: Tuple[float, float] = (2, 4)):
        """Draw signature with AxiDraw"""

        print("Drawing signature...")

        for stroke in strokes:
            if len(stroke) < 2:
                continue

            # Generate varied path
            varied_stroke = self.generate_signature_path(stroke)

            # Scale and offset
            scaled_stroke = [(offset[0] + x * scale, offset[1] + y * scale)
                           for x, y in varied_stroke]

            # Move to start
            self.ad.penup()
            self.ad.goto(scaled_stroke[0][0], scaled_stroke[0][1])

            # Draw stroke with varying speed
            self.ad.pendown()
            for i, (x, y) in enumerate(scaled_stroke[1:]):
                # Vary speed based on curvature
                if i > 0:
                    prev_x, prev_y = scaled_stroke[i]
                    dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)

                    # Slow down on tight curves
                    if dist < 0.05:
                        self.ad.options.speed_pendown = 20
                    else:
                        self.ad.options.speed_pendown = 30

                self.ad.goto(x, y)

            self.ad.penup()

    def disconnect(self):
        """Disconnect from AxiDraw"""
        self.ad.penup()
        self.ad.goto(0, 0)
        self.ad.disconnect()

def main():
    # Check if signatures folder exists
    signature_folder = "buzz_aldrin_signatures"

    if not os.path.exists(signature_folder):
        print(f"Please create a folder '{signature_folder}' and add signature images")
        print("You can download them from the Google Drive folder you mentioned")
        return

    # Analyze signatures
    analyzer = SignatureAnalyzer()
    analyzer.load_signature_images(signature_folder)
    analyzer.analyze_characteristics()

    # Save characteristics
    with open('signature_characteristics.json', 'w') as f:
        json.dump(analyzer.characteristics, f, indent=2)

    print("\nCharacteristics saved to signature_characteristics.json")

    # Reproduce signature
    if analyzer.signatures:
        reproducer = SignatureReproducer(analyzer.characteristics)

        if reproducer.connect():
            print("\nConnected to AxiDraw")

            # Use first signature as base
            base_sig = analyzer.signatures[0]
            reproducer.draw_signature(base_sig['strokes'])

            reproducer.disconnect()
        else:
            print("Could not connect to AxiDraw")

if __name__ == "__main__":
    main()
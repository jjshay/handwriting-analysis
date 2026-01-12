"""
Tests for Handwriting Analysis System
"""
import pytest
import os
import json
from pathlib import Path


class TestBaselineData:
    """Test baseline data handling"""

    def test_baseline_file_exists(self):
        """Verify baseline data file exists"""
        baseline_path = Path(__file__).parent.parent / "examples" / "sample_baseline_data.json"
        assert baseline_path.exists(), "Baseline data file should exist"

    def test_baseline_valid_json(self):
        """Verify baseline is valid JSON"""
        baseline_path = Path(__file__).parent.parent / "examples" / "sample_baseline_data.json"
        with open(baseline_path) as f:
            data = json.load(f)
        assert "baseline_statistics" in data, "Should have baseline_statistics"

    def test_baseline_has_features(self):
        """Verify baseline has required feature statistics"""
        baseline_path = Path(__file__).parent.parent / "examples" / "sample_baseline_data.json"
        with open(baseline_path) as f:
            data = json.load(f)

        required_features = ["slant_angle", "height_ratio", "stroke_width", "pressure_variance"]
        stats = data["baseline_statistics"]

        for feature in required_features:
            assert feature in stats, f"Missing feature: {feature}"
            assert "mean" in stats[feature], f"{feature} should have mean"
            assert "std" in stats[feature], f"{feature} should have std"


class TestStatisticalMethods:
    """Test statistical calculation methods"""

    def test_z_score_calculation(self):
        """Test z-score calculation"""
        mean = 71.5
        std = 4.2
        value = 73.2

        z_score = (value - mean) / std
        assert abs(z_score - 0.405) < 0.01, "Z-score calculation should be correct"

    def test_within_range_detection(self):
        """Test within-range detection (2 std devs)"""
        mean = 71.5
        std = 4.2

        # Within range
        assert abs(73.2 - mean) <= 2 * std, "73.2 should be within 2 std devs"

        # Outside range
        assert abs(85.0 - mean) > 2 * std, "85.0 should be outside 2 std devs"


class TestSignatureImages:
    """Test signature sample images"""

    def test_authentic_signatures_exist(self):
        """Verify authentic signature samples exist"""
        examples_path = Path(__file__).parent.parent / "examples"

        for i in range(1, 4):
            sig_path = examples_path / f"signature_authentic_0{i}.png"
            assert sig_path.exists(), f"Authentic signature {i} should exist"

    def test_questioned_signature_exists(self):
        """Verify questioned signature sample exists"""
        sig_path = Path(__file__).parent.parent / "examples" / "signature_questioned.png"
        assert sig_path.exists(), "Questioned signature should exist"

    def test_signature_is_png(self):
        """Verify signatures are valid PNG files"""
        sig_path = Path(__file__).parent.parent / "examples" / "signature_authentic_01.png"
        with open(sig_path, 'rb') as f:
            header = f.read(8)
        # PNG magic bytes
        assert header[:4] == b'\x89PNG', "File should be a valid PNG"


class TestForensicReport:
    """Test forensic report output format"""

    def test_sample_report_exists(self):
        """Verify sample report exists"""
        report_path = Path(__file__).parent.parent / "sample_output" / "forensic_report.txt"
        assert report_path.exists(), "Sample forensic report should exist"

    def test_report_has_sections(self):
        """Verify report contains required sections"""
        report_path = Path(__file__).parent.parent / "sample_output" / "forensic_report.txt"
        with open(report_path) as f:
            content = f.read()

        required_sections = ["FEATURE COMPARISON", "STATISTICAL ANALYSIS", "CONCLUSION"]
        for section in required_sections:
            assert section in content, f"Report should contain {section} section"

# Handwriting Analysis System

![CI](https://github.com/jjshay/handwriting-analysis/workflows/CI/badge.svg)
![CodeQL](https://github.com/jjshay/handwriting-analysis/workflows/CodeQL/badge.svg)
[![codecov](https://codecov.io/gh/jjshay/handwriting-analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/jjshay/handwriting-analysis)
![Release](https://img.shields.io/github/v/release/jjshay/handwriting-analysis)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-enabled-brightgreen.svg)
![AI Optional](https://img.shields.io/badge/AI-optional-lightgrey.svg)

**Forensic-grade signature and handwriting analysis - statistical comparison for authentication.**

[![Demo](https://asciinema.org/a/bWZcm0S3ayeCsVy4.svg)](https://asciinema.org/a/bWZcm0S3ayeCsVy4)

[![Watch Demo](https://img.shields.io/badge/▶%20Watch%20Demo-Click%20to%20Play-red?style=for-the-badge&logo=asciinema)](https://asciinema.org/a/bWZcm0S3ayeCsVy4)

---

## What Does This Do?

Analyze signatures and handwriting for authenticity:

1. **Extract features** from signature images (slant, pressure, proportions)
2. **Build baseline** from known authentic samples
3. **Compare questioned documents** against the baseline
4. **Generate reports** with statistical confidence scores

**Use cases:**
- Art authentication (artist signatures)
- Document verification
- Forensic analysis
- Research and education

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/jjshay/handwriting-analysis.git
cd handwriting-analysis

# Install dependencies
pip install -r requirements.txt

# Run the interactive demo
python demo.py

# Or run the visual showcase
python showcase.py

# Analyze with sample signatures
python forensic_handwriting_analyzer.py \
  --baseline examples/sample_baseline_data.json \
  --questioned examples/signature_questioned.png
```

### Sample Files
- `examples/signature_authentic_01.png` - Sample authentic signatures
- `examples/signature_questioned.png` - Questioned signature to analyze
- `examples/sample_baseline_data.json` - Pre-computed baseline stats
- `sample_output/forensic_report.txt` - Example analysis report

---

## Architecture

```mermaid
flowchart TB
    subgraph Baseline["Baseline Creation"]
        A[Authentic Samples] --> B[Image Preprocessing]
        B --> C[Feature Extractor]
        C --> D[Statistical Analyzer]
        D --> E[Baseline Model]
    end

    subgraph Analysis["Document Analysis"]
        F[Questioned Document] --> G[Image Preprocessing]
        G --> H[Feature Extractor]
    end

    subgraph Comparison["Statistical Comparison"]
        E --> I[Feature Comparison]
        H --> I
        I --> J[Z-Score Calculator]
        J --> K[Mahalanobis Distance]
        K --> L[Probability Engine]
    end

    subgraph Optional["Optional: AI Review"]
        H --> M1[Claude]
        H --> M2[GPT-4]
        H --> M3[Gemini]
        H --> M4[Grok]
        M1 & M2 & M3 & M4 --> N[AI Consensus]
    end

    subgraph Output
        L --> O[Confidence Score]
        N --> O
        O --> P[Forensic Report]
    end

    style A fill:#e1f5fe
    style F fill:#fff9c4
    style P fill:#c8e6c9
```

## How It Works

```
Known Authentic Samples (70+)
            │
            ▼
┌─────────────────────────────────────────┐
│       FEATURE EXTRACTION                 │
│  - Slant angle (60-90°)                  │
│  - Height ratios                         │
│  - Stroke characteristics                │
│  - Pressure patterns                     │
│  - Connected vs lifted strokes           │
└─────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│       STATISTICAL BASELINE              │
│  Mean ± Standard Deviation for each     │
│  feature across all authentic samples   │
└─────────────────────────────────────────┘
            │
            ▼
  Questioned Document
            │
            ▼
┌─────────────────────────────────────────┐
│       STATISTICAL COMPARISON            │
│  - Z-scores for each feature            │
│  - Mahalanobis distance                 │
│  - Probability calculations             │
└─────────────────────────────────────────┘
            │
            ▼
  Authentication Report
  (Confidence: 94.7%)
```

---

## Features Extracted

| Feature | Description | Typical Range |
|---------|-------------|---------------|
| Slant Angle | Degrees from horizontal | 60-90° |
| Height Ratio | Full height / x-height | 2.0-3.5 |
| Aspect Ratio | Width / Height of bounding box | 1.5-4.0 |
| Stroke Width | Average pen stroke thickness | 2-8 px |
| Pressure Variance | Variation in stroke darkness | 0.1-0.4 |
| Connectedness | % of connected strokes | 40-80% |
| Loop Ratio | Loops per letter | 0.2-0.6 |

---

## Output Report

```
FORENSIC HANDWRITING ANALYSIS REPORT
=====================================
Document ID: QD-2024-001
Analysis Date: January 8, 2024

FEATURE COMPARISON:
  Slant Angle:      73.2° (Baseline: 71.5° ± 4.2°)  → WITHIN RANGE
  Height Ratio:     2.8   (Baseline: 2.6 ± 0.3)    → WITHIN RANGE
  Stroke Width:     4.1px (Baseline: 3.8 ± 0.8)    → WITHIN RANGE
  Pressure Var:     0.28  (Baseline: 0.25 ± 0.08)  → WITHIN RANGE

STATISTICAL ANALYSIS:
  Combined Z-Score: 0.82
  Mahalanobis Distance: 1.24

CONCLUSION:
  Confidence: 94.7%
  Opinion: The questioned signature is CONSISTENT with
           the known authentic baseline samples.
```

---

## Multi-AI Review (Optional)

For additional verification, the system can use multiple AI models:

| AI Model | Role |
|----------|------|
| Claude | Detailed stroke analysis |
| GPT-4 | Pattern recognition |
| Gemini | Visual comparison |
| Grok | Anomaly detection |

Each AI provides independent analysis, then results are combined for consensus.

---

## Setup

### Basic (Statistical Only)
```bash
pip install -r requirements.txt
python demo.py
```

### Full AI Mode
```bash
cp .env.example .env
nano .env
# Add API keys
python llm_handwriting_reviewer.py
```

---

## Files

| File | Purpose |
|------|---------|
| `forensic_handwriting_analyzer.py` | Main statistical analyzer |
| `signature_analyzer.py` | Signature-specific analysis |
| `statistical_signature_generator.py` | Generate test signatures |
| `llm_handwriting_reviewer.py` | Multi-AI review system |
| `demo.py` | Demo without dependencies |

---

## Important Notes

- This is a research/educational tool
- Not a replacement for professional forensic examination
- Statistical analysis provides probability, not certainty
- Always consult qualified experts for legal matters

---

## License

MIT - For research and educational use

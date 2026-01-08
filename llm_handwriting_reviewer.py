#!/usr/bin/env python3
"""
LLM HANDWRITING REVIEWER
Multi-AI system for analyzing and reviewing handwriting quality using latest models
"""

import os
import requests
import json
import base64
import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from typing import Dict, List, Optional
import asyncio
import aiohttp
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class APIConfig:
    """Configuration for different AI services - loads from environment"""

    # Latest model versions
    CLAUDE_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"

    CHATGPT_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    CHATGPT_MODEL: str = "gpt-4o"

    GROK_API_KEY: str = os.getenv('XAI_API_KEY', '')
    GROK_MODEL: str = "grok-2-1212"

    GEMINI_API_KEY: str = os.getenv('GOOGLE_API_KEY', '')
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"

class LLMHandwritingReviewer:
    """Multi-LLM handwriting analysis and review system"""

    def __init__(self):
        self.config = APIConfig()
        self.analysis_prompt = self._create_analysis_prompt()

    def _create_analysis_prompt(self) -> str:
        """Create comprehensive handwriting analysis prompt"""
        return """
You are an expert handwriting analyst and forensic document examiner. Please analyze this handwritten text image with extreme attention to detail.

Provide analysis in these specific categories:

🖊️ **TECHNICAL QUALITY (1-10 scale)**
- Stroke consistency and smoothness
- Pressure variation naturalness
- Line quality and pen control
- Overall technical execution

✍️ **SPACING & LAYOUT (1-10 scale)**
- Character spacing consistency
- Word spacing appropriateness
- Line spacing and alignment
- Margin utilization

📏 **BASELINE & STRUCTURE (1-10 scale)**
- Baseline consistency and natural drift
- Character size variation
- Ascender/descender proportions
- Overall structural integrity

🎭 **NATURALISM & REALISM (1-10 scale)**
- Human-like variations and imperfections
- Natural tremor and micro-movements
- Realistic fatigue effects
- Authentic handwriting characteristics

💡 **SPECIFIC OBSERVATIONS**
- Most realistic aspects
- Areas that need improvement
- Comparison to human handwriting
- Suggestions for enhancement

⭐ **OVERALL SCORE (1-100)**
- Composite score with reasoning
- Key strengths and weaknesses
- Would this pass as human handwriting?

Please be extremely detailed and technical in your analysis. Focus on micro-details that distinguish human handwriting from synthetic/artificial text.
"""

    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def analyze_with_claude(self, image_path: str) -> Dict:
        """Analyze handwriting using Claude"""
        print("🟣 Analyzing with Claude...")

        try:
            image_base64 = self._image_to_base64(image_path)

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.config.CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01"
            }

            payload = {
                "model": self.config.CLAUDE_MODEL,
                "max_tokens": 1500,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.analysis_prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload
                ) as response:
                    result = await response.json()

                    if response.status == 200:
                        return {
                            "status": "success",
                            "analysis": result['content'][0]['text'],
                            "model": self.config.CLAUDE_MODEL
                        }
                    else:
                        return {
                            "status": "error",
                            "error": result,
                            "model": self.config.CLAUDE_MODEL
                        }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model": self.config.CLAUDE_MODEL
            }

    async def analyze_with_chatgpt(self, image_path: str) -> Dict:
        """Analyze handwriting using ChatGPT"""
        print("🟢 Analyzing with ChatGPT...")

        try:
            image_base64 = self._image_to_base64(image_path)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.CHATGPT_API_KEY}"
            }

            payload = {
                "model": self.config.CHATGPT_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.analysis_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1500
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    result = await response.json()

                    if response.status == 200:
                        return {
                            "status": "success",
                            "analysis": result['choices'][0]['message']['content'],
                            "model": self.config.CHATGPT_MODEL
                        }
                    else:
                        return {
                            "status": "error",
                            "error": result,
                            "model": self.config.CHATGPT_MODEL
                        }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model": self.config.CHATGPT_MODEL
            }

    async def analyze_with_grok(self, image_path: str) -> Dict:
        """Analyze handwriting using Grok"""
        print("🔴 Analyzing with Grok...")

        try:
            image_base64 = self._image_to_base64(image_path)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.GROK_API_KEY}"
            }

            payload = {
                "model": self.config.GROK_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.analysis_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1500,
                "temperature": 0.1
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    result = await response.json()

                    if response.status == 200:
                        return {
                            "status": "success",
                            "analysis": result['choices'][0]['message']['content'],
                            "model": self.config.GROK_MODEL
                        }
                    else:
                        return {
                            "status": "error",
                            "error": result,
                            "model": self.config.GROK_MODEL
                        }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model": self.config.GROK_MODEL
            }

    async def analyze_with_gemini(self, image_path: str) -> Dict:
        """Analyze handwriting using Gemini"""
        print("🔷 Analyzing with Gemini...")

        try:
            image_base64 = self._image_to_base64(image_path)

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.GEMINI_MODEL}:generateContent?key={self.config.GEMINI_API_KEY}"

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": self.analysis_prompt
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1500
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    result = await response.json()

                    if response.status == 200:
                        return {
                            "status": "success",
                            "analysis": result['candidates'][0]['content']['parts'][0]['text'],
                            "model": self.config.GEMINI_MODEL
                        }
                    else:
                        return {
                            "status": "error",
                            "error": result,
                            "model": self.config.GEMINI_MODEL
                        }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model": self.config.GEMINI_MODEL
            }

    async def comprehensive_analysis(self, image_path: str) -> Dict:
        """Run analysis with all available LLMs"""
        print("🤖 Starting comprehensive multi-LLM handwriting analysis...")
        print("=" * 60)

        # Run all analyses concurrently
        tasks = [
            self.analyze_with_claude(image_path),
            self.analyze_with_chatgpt(image_path),
            self.analyze_with_grok(image_path),
            self.analyze_with_gemini(image_path)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Organize results
        analysis_results = {
            "image_path": image_path,
            "analyses": {
                "claude": results[0] if not isinstance(results[0], Exception) else {"status": "error", "error": str(results[0])},
                "chatgpt": results[1] if not isinstance(results[1], Exception) else {"status": "error", "error": str(results[1])},
                "grok": results[2] if not isinstance(results[2], Exception) else {"status": "error", "error": str(results[2])},
                "gemini": results[3] if not isinstance(results[3], Exception) else {"status": "error", "error": str(results[3])}
            }
        }

        return analysis_results

    def generate_comparison_report(self, results: Dict) -> str:
        """Generate a comprehensive comparison report"""
        report = []
        report.append("🖊️ " + "=" * 60)
        report.append("    MULTI-LLM HANDWRITING ANALYSIS REPORT")
        report.append("=" * 62 + " 🖊️")
        report.append("")
        report.append(f"📸 Image: {results['image_path']}")
        report.append("")

        # Analyze each model's response
        model_icons = {
            "claude": "🟣",
            "chatgpt": "🟢",
            "grok": "🔴",
            "gemini": "🔷"
        }

        successful_analyses = []
        failed_analyses = []

        for model_name, result in results['analyses'].items():
            icon = model_icons.get(model_name, "🤖")
            model_display = model_name.upper()

            report.append(f"{icon} {model_display} Analysis")
            report.append("-" * 40)

            if result['status'] == 'success':
                report.append("✅ Analysis completed successfully")
                report.append(f"🔧 Model: {result.get('model', 'Unknown')}")
                report.append("")
                report.append("📋 Analysis Results:")
                report.append(result['analysis'][:500] + "..." if len(result['analysis']) > 500 else result['analysis'])
                successful_analyses.append(model_name)
            else:
                report.append("❌ Analysis failed")
                report.append(f"🚫 Error: {result['error']}")
                failed_analyses.append(model_name)

            report.append("")
            report.append("")

        # Summary section
        report.append("📊 ANALYSIS SUMMARY")
        report.append("=" * 30)
        report.append(f"✅ Successful analyses: {len(successful_analyses)}/4")
        report.append(f"❌ Failed analyses: {len(failed_analyses)}/4")

        if successful_analyses:
            report.append(f"🎯 Working models: {', '.join(successful_analyses).upper()}")
        if failed_analyses:
            report.append(f"💥 Failed models: {', '.join(failed_analyses).upper()}")

        return "\n".join(report)

    def save_analysis_report(self, results: Dict, output_path: str = None):
        """Save the analysis report to file"""
        if not output_path:
            output_path = "/Users/johnshay/handwriting_analysis_report.txt"

        report = self.generate_comparison_report(results)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📄 Analysis report saved to: {output_path}")
        return output_path

def create_sample_handwriting_image(text: str = "The quick brown fox jumps over the lazy dog") -> str:
    """Create a sample handwriting image for testing"""
    from advanced_handwriting_engine import AdvancedHandwritingEngine, create_handwriting_style_presets

    # Use a natural style
    styles = create_handwriting_style_presets()
    engine = AdvancedHandwritingEngine(styles['confident_executive'])
    stroke_data = engine.generate_text_strokes(text)

    # Create image
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    for stroke, pressures in zip(stroke_data['strokes'], stroke_data['pressures']):
        if stroke:
            xs, ys = zip(*stroke)
            # Use pressure to vary line thickness
            for i in range(len(xs)-1):
                thickness = 1 + pressures[i] * 2
                ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                       'k-', linewidth=thickness, alpha=0.8)

    ax.set_xlim(-0.5, max([max([x for x, y in stroke]) for stroke in stroke_data['strokes'] if stroke]) + 0.5)
    ax.set_ylim(min([min([y for x, y in stroke]) for stroke in stroke_data['strokes'] if stroke]) - 0.5, 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.invert_yaxis()

    # Save as high-resolution image
    sample_path = "/Users/johnshay/sample_handwriting_for_analysis.png"
    fig.savefig(sample_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig)

    return sample_path

async def main():
    """Demo the LLM handwriting review system"""
    print("🤖 LLM HANDWRITING REVIEWER DEMO")
    print("=" * 50)

    # Create sample handwriting
    print("📝 Creating sample handwriting...")
    sample_image = create_sample_handwriting_image()
    print(f"✅ Sample created: {sample_image}")

    # Initialize reviewer
    reviewer = LLMHandwritingReviewer()

    # Run comprehensive analysis
    results = await reviewer.comprehensive_analysis(sample_image)

    # Generate and display report
    report = reviewer.generate_comparison_report(results)
    print("\n" + report)

    # Save report
    report_path = reviewer.save_analysis_report(results)

    print(f"\n🎉 Analysis complete!")
    print(f"📸 Sample image: {sample_image}")
    print(f"📄 Report saved: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())
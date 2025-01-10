from setuptools import setup, find_packages
import os
from pathlib import Path

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="video-analyzer",
    version="0.1.0",
    author="Jesse White",
    description="A tool for analyzing videos using Vision models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        'video_analyzer': [
            'config/*.json',
            'prompts/**/*',
        ],
    },
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "video-analyzer=video_analyzer.cli:main",
        ],
    },
    python_requires=">=3.8",
    include_package_data=True
)

from setuptools import setup, find_packages
import os
from pathlib import Path

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Create data_files list for config and prompts
data_files = [
    ('video_analyzer/config', ['video_analyzer/config/default_config.json'])
]

# Recursively add all files from prompts directory
prompts_dir = Path('prompts')
for path in prompts_dir.rglob('*'):
    if path.is_file():
        # Convert path to relative path from prompts directory
        relative_path = path.relative_to(prompts_dir)
        # Create the target directory path
        target_dir = f'video_analyzer/prompts/{relative_path.parent}'
        # Add the file to data_files
        data_files.append((target_dir, [str(path)]))

setup(
    name="video-analyzer",
    version="0.1.0",
    author="Jesse White",
    description="A tool for analyzing videos using Vision models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "video-analyzer=video_analyzer.cli:main",
        ],
    },
    python_requires=">=3.8",
    include_package_data=True,
    data_files=data_files
)

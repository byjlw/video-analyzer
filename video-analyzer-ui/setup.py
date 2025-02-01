from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="video-analyzer-ui",
    version="0.1.0",
    author="Jesse White",
    description="Web interface for the video-analyzer tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        'video_analyzer_ui': [
            'templates/*',
            'static/css/*',
            'static/js/*',
        ],
    },
    install_requires=[
        "flask>=3.0.0",
        "video-analyzer>=0.1.0",
    ],
    entry_points={
        "console_scripts": [
            "video-analyzer-ui=video_analyzer_ui.server:main",
        ],
    },
    python_requires=">=3.8",
    include_package_data=True
)

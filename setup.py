from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='video-analyzer',
    version='0.1.0',
    description='A comprehensive video analysis tool',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='byjlw',
    author_email='',
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': [
            'video-analyzer=video_analyzer.cli:main',
            'video-analyzer-gui=video_analyzer.gui:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Environment :: X11 Applications :: Qt',
    ],
    python_requires='>=3.8',
)
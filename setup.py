# setup.py
from setuptools import setup, find_packages

setup(
    name="prospectPlotter",              # the name people will pip install
    version="0.1.0",            # your initial version
    author="Olivia Curtis",
    description="Plotting functions for Prospector outputs",
    packages=find_packages(),   # auto-discovers the 'myutil' package
    install_requires=[
        "numpy", "matplotlib", "astro-prospector"
	# e.g. "numpy>=1.20", "requests"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)

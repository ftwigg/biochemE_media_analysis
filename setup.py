from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="biochemE_media_analysis",
    version="0.1.0",
    author="Frederick Fairbank Twigg",
    author_email="frederick.twigg@berkeley.edu",
    description="A tool for analyzing fermentation and cell culture media composition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ftwigg/biochemE_media_analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "openpyxl>=3.0.0",
    ],
    package_data={
        "biochemE_media_analysis": ["data/*.json"],
    },
)

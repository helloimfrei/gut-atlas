from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gutatlas",
    version="1.0.0",
    author="Christian Tapp",
    description="Machine learning package for predicting GI disease risk from gut microbiome composition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gutatlas-build=gutatlas.scripts.build_dataset:main",
            "gutatlas-train=gutatlas.scripts.train:main",
            "gutatlas-plot=gutatlas.scripts.generate_plots:main",
            "gutatlas-report=gutatlas.scripts.generate_report:main",
        ],
    },
)

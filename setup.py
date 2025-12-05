from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="graph-novelty",
    version="1.0.0",
    author="Rucha Bhalchandra Joshi, Subhankar Mishra",
    author_email="r.joshi@cyi.ac.cy, smishra@niser.ac.in",
    description="Compositional Novelty Metrics for Graph-Structured Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smlab-niser/compositional-graph-novelty",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.9b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
    },
)

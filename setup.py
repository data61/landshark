#!/usr/bin/env python

from setuptools import setup, find_packages

__version__ = None
exec(open("landshark/__version__.py").read())
readme = open("README.md").read()

setup(
    name="landshark",
    version=__version__,
    description="Large-scale spatial inference with Tensorflow",
    long_description=readme,
    author="Data61",
    author_email="lachlan.mccalman@data61.csiro.au",
    url="https://github.com/determinant-io/landshark",
    packages=find_packages(),
    package_dir={"landshark": "landshark"},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "landshark = landshark.scripts.cli:cli",
            "skshark = landshark.scripts.skcli:cli",
            "landshark-import = landshark.scripts.importers:cli",
        ]
    },
    install_requires=[
        "numpy>=1.13.3",
        "scipy>=1.0.0",
        "click>=6.7",
        "GDAL==2.0.1",
        "rasterio>=1.0a10",
        "tables>=3.4.2",
        "pyshp>=1.2.12",
        "mypy>=0.521",
        "mypy_extensions>=0.3.0",
        "lru-dict>=1.1.6",
        "tqdm>=4.19.6",
        "scikit-learn>=0.19.1",
        "aboleth>=0.7.0"
    ],

    extras_require={
        "dev": [
            "jedi>=0.10.2",
            "pytest>=3.1.3",
            "pytest-flake8>=0.8.1",
            "pytest-mock>=1.6.2",
            "pytest-cov>=2.5.1",
            "pytest-regtest>=0.15.1",
            "pytest-xdist>=1.22.0",
            "flake8-docstrings>=1.1.0",
            "flake8-quotes>=0.11.0",
            "flake8-comprehensions>=1.4.1"
        ]
    },
    license="All Rights Reserved",
    zip_safe=False,
    keywords="landshark",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: POSIX",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.4",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

#!/usr/bin/env python

from setuptools import find_packages, setup

import versioneer

readme = open("README.md").read()

setup(
    name="landshark",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Large-scale spatial inference with Tensorflow",
    long_description=readme,
    author="Data61",
    author_email="lachlan.mccalman@data61.csiro.au",
    url="https://bitbucket.csiro.au/projects/DGEO/repos/landshark/browse",
    packages=find_packages(),
    package_dir={"landshark": "landshark"},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "landshark = landshark.scripts.cli:cli",
            "skshark = landshark.scripts.skcli:cli",
            "landshark-import = landshark.scripts.importers:cli",
            "landshark-extract = landshark.scripts.extractors:cli",
        ]
    },
    install_requires=[
        "numpy>=1.13.3",
        "scipy>=0.19",
        "click>=6.7",
        "pygdal>=2.2.3.3",
        "rasterio>=1.0.2",
        "tables>=3.4.2",
        "pyshp>=1.2.12",
        "mypy>=0.521",
        "mypy_extensions>=0.3.0",
        "lru-dict>=1.1.6",
        "tqdm>=4.19.6",
        "scikit-learn>=0.19.1",
        "tensorflow>=1.7"
    ],
    extras_require={
        "dev": [
            "jedi>=0.10.2",
            "pytest>=3.1.3",
            "pytest-flake8>=0.8.1",
            "pytest-mock>=1.6.2",
            "flake8-bugbear==18.2.0",
            "flake8-builtins==1.4.1",
            "pytest-cov>=2.5.1",
            "flake8-comprehensions>=1.4.1",
            "flake8-docstrings>=1.1.0",
            "flake8-isort>=2.5",
            "flake8-quotes>=0.11.0",
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
        "Programming Language :: Python :: 3.5",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

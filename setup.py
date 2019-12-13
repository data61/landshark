#!/usr/bin/env python

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    author_email="dave.cole@data61.csiro.au",
    url="https://github.com/data61/landshark",
    packages=find_packages(),
    package_dir={"landshark": "landshark"},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "landshark-keras = landshark.scripts.kerascli:cli",
            "landshark = landshark.scripts.cli:cli",
            "skshark = landshark.scripts.skcli:cli",
            "landshark-import = landshark.scripts.importers:cli",
            "landshark-extract = landshark.scripts.extractors:cli",
        ]
    },
    install_requires=[
        "numpy>=1.16",
        "scipy>=0.19",
        "click>=6.7",
        "rasterio>=1.0.2",
        "tables>=3.4.2",
        "pyshp>=1.2.12",
        "mypy>=0.521",
        "mypy_extensions>=0.3.0",
        "lru-dict>=1.1.6",
        "tqdm>=4.19.6",
        "scikit-learn>=0.20.0",
        "tensorflow>=2.0"
        "tensorflow_probability>=0.8"
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
    license="Apache 2.0",
    zip_safe=False,
    keywords="landshark",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

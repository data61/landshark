#!/usr/bin/env python

from setuptools import setup, find_packages

readme = open('README.md').read()
setup(
    name='landshark',
    version='0.1.0',
    description='Large-scale spatial inference with Tensorflow',
    long_description=readme,
    author='Determinant',
    author_email='lachlan.mccalman@data61.csiro.au',
    url='https://github.com/determinant-io/landshark',
    packages=find_packages(),
    package_dir={'landshark': 'landshark'},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'landshark = landshark.scripts.cli:cli',
        ]
    },
    install_requires=[
        'numpy>=1.13.1',
        'scipy>=0.19.1',
        'click==6.7',
        # tensorflow==1.2.1,
        # tensorflow-gpu-1.2.1,
    ],

    extras_require={
        'dev': [
            'neovim>=0.1.13',
            'jedi>=0.10.2',
            'pytest>=3.1.0',
            'pytest-flake8>=0.8.1',
            'pytest-mock>=1.6.0',
            'pytest-cov>=2.5.1',
            'pytest-regtest>=0.15.1',
            'flake8-docstrings>=1.1.0',
        ]
    },
    license="All Rights Reserved",
    zip_safe=False,
    keywords='landshark',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Operating System :: POSIX",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.4",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

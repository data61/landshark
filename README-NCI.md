# Landshark on the NCI

This README is a quick guide to getting the landshark library up and running
in a PBS batch environment that has MPI support. This setup is common in
HPC systems such as the NCI (raijin).

The instructions below should apply to both single- and multi-node runs
on the NCI. Just set ncpus in the PBS  directives in the job submission
script accordingly (e.g. ncpus=32 for 2 nodes).

The instructions assume you are using bash shell.

## Pre-installation

These instructions currently only work with gcc and not the Intel compiler.
Note that on NCI it appears python is compiled against gcc anyway.

1. Unload the icc compiler from the terminal:
```bash
$ module unload intel-cc
$ module unload intel-fc
```
2. Load the modules requried for installation and running:
```bash
module load python3/3.6.2
module load tensorflow/1.8-cudnn7.1-python3.6
module load gdal/2.2.2 git/2.9.5 gcc/4.9.0 openmpi/3.1.0
```
(Alternatively, you may wish to add the above lines to your ~/.profile (or ~/
.bashrc))

2. Now add the following lines to the end of your ~/.profile (or ~/.bashrc):

```bash
export PATH=$HOME/.local/bin:$PATH
export VIRTUALENVWRAPPER_PYTHON=/apps/python3/3.6.2/bin/python3
export PYTHONPATH=/home/547/sxb547/.local/lib/python3.6/site-packages:$PYTHONPATH
export LC_ALL=en_AU.UTF-8
export LANG=en_AU.UTF-8
source $HOME/.local/bin/virtualenvwrapper.sh
``` 

4. Install virtualenv and virtualenvwrapper by running the following command
on the terminal:
```bash
$ pip3 install  --user virtualenv virtualenvwrapper
```

5. Refresh your environment by reloading your profile:
```bash
$ source ~/.profile
# or 
$ source ~/.profile
```

## Installation

1. Create a new virtualenv for `landshark`:
```bash
$ mkvirtualenv --system-site-packages landshark
```

2. Make sure the virtualenv is activated:
```bash
$ workon landshark
```

3. Clone the `landshark` repo into your home directory:
```bash
$ cd ~
$ git clone https://github.com/basaks/landshark.git
```

4. Install `landshark`:
```bash
$ cd 
$ pip install -e .[dev]
```

5. Once installation has completed, you can run the tests to verify everything
has gone correctly:
```bash
$ make test
```
Follow the above by running the integration tests 
 
```bash
$ make integration
```

## Updating the Code
To update the code, first make sure you are in the `landshark` virtual 
environment:
```bash
$ workon landshark
```
Next, pull the latest commit from the master branch, and install:
```bash
$ cd ~/landshark
$ git pull origin
$ pip install -e .[dev]
```
If the pull and the installation complete successfully, the code is ready to run!

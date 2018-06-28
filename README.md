# landshark

Large-scale spatial inference with Tensorflow.

Designed to work with Aboleth (github.com/data61/aboleth).


## Introduction

Landshark is a set of python command line tools that for supervised learning
problems on large spatial raster datasets (with sparse targets).  It solves
problems in which the user has a set of target point measurements (such as
geochemistry, soil classification, or depth to basement) and wants to relate
those to a number of raster covariates (like satellite imagery or geophysics)
to predict the targets on the raster grid.

Many such tools exist already for this problem. Landshark fills a particular
niche: where we want to efficiently learn models with very large numbers of
training points and/or very large covariate images using tensorflow. Landshark
is particularly useful for the case when the training data itself will not fit
in memory, and must be streamed to a minibatch stochastic gradient descent
algorithm for model learning.

There are no actual models in landshark. It is really just a training and query
data conversion system to get shapefiles and geotiffs to and from 'tfrecord'
files which can be read efficiently by tensorflow.

The choice of models is up to the user: arbitrary tensorflow models are fine, as
is anything built on top of tensorflow like Keras or Aboleth. There's also
a scikit-learn interface for problems that do fit in memory (although this is
mainly for validation purposes).

## Installation

### Prerequisites

 The following will need to be installed before installing Landshark:

 - Python 3.5+
 - GDAL 2.0.1+
 - Tensorflow 1.6+

Note that if tensorflow isn't found during Landshark installation, 
pip will install the pre-compiled version which has horrible performance. 
Make sure you compile tensorflow yourself!

### Installing Landshark

Once you have cloned the repository, simply run

```bash
$ pip install .
```

If you're planning to do development, or even run the tests, you probably want
to run

```bash
$ pip install -e .[dev]
```
to install the development dependencies and link to the actual source files.

### Testing

Once you've installed the development dependencies, you can run:

 - `make test` for the unit tests.
 - `make integration` for integration tests. Note these can be computationally
   expensive, they're doing 48 combinations of the full pipeline with different
   settings (on a small example dataset).

If you want to check some code you're writing:
 - `make lint` will lint the code according to PEP8.
 - `make typecheck` will type-check the code using mypy.

## Outline

The basic steps in using Landshark are:

1. Import geotiffs and a target shapefile with `landshark-import`,
2. Extract training/testing and query data with `landshark-extract`,
3. Train a model and predict with the `landshark` command, 
   or `skshark` for scikit-learn models.

If you would like to use landshark outputs for other tools, you can
export training, testing and query data with the `landshark-dump` command.


## Data Prerequisites

Landshark has not tried to replicate features that exist in other tools,
especially image processing and GIS. Therefore, it has quite strict
requirements for data input:

1. Targets are stored as points in a shapefile
2. Covariates are stored as a set of geotiffs (1 or more)
3. All geotiffs have the same projection, resolution and bounding box
4. The projection of the targets is the same as that of the geotiffs
4. All target points are inside the bounding box of the geotiffs
5. The covariate images are also those used for prediction (i.e, the prediction
   image will come out with the same resultion and bounding box as the
   covariate images)
6. Geotiffs may have multiple bands, and different geotiffs may have different
   datatypes (eg uint8, float32), but within a single geotiff all bands must
   have the same datatype
7. Geotiffs have been categorized into continuous data (referred to as 'ordinal'), 
   and categorical data. These two sets of geotiffs are stored in separate
   folders.


## Usage Example

We have say a dozen `.tif` covariates stored between our `./ord_images` and
`./cat_images` folder. We have a target shapefile in our `./targets` folder. 
We're going to use the `landshark-import` command to get our data into a format
useable by landshark.

### 1. Import the data

We start by creating a tiff stack called "murray".

```bash
$ landshark-import tifs --ordinal ord_images/ --categorical cat_images --name murray
```

The result of this command will be a file in the current directory called 
`features_murray.hdf5`. Similarly, we import some "Sodium" targets from
a shapefile. Note we can import as many records as we like using multiple
`--record` flags:

```bash
$ landshark-import targets --name sodium --shapefile ./targets/Na.shp --dtype ordinal --record Na_conc --record meas_error
```

We've also specified that the type of the records is ordinal (ie continuous).
From this command, landshark will output `targets_sodium.hdf5`.

### 2. Extract Train/test and Query Data

Let's try putting together a regression problem from the data we just imported.
We're going to use the `landshark-extract` command  for this. 
Starting with a train/test set, we use the 'traintest' sub-command:

```bash
$ landshark-extract traintest --features features_murray.hdf5 --targets targets_sodium.hdf5 --name myproblem
```

This command will create a folder in the new directory called
`traintest_myproblem_fold1of10`. The 'fold1of10' part of that folder name is
indicating how the test data were selected. For more information see the
landshark-extract options, but the default is fine for now.

Similarly, we can extract the query data:

```bash
$ landshark-extract query --features features_murray.hdf5 --name myproblem
```

This command will create a folder in a new directory called
`query_myproblem_strip1of1`. The 'strip1of1' indicates the whole image has been
extracted. For large images it is possible to extract only a small (horizontal)
window of the image for iterative processing. See the landshark-extract docs
for details on this.


### 3. Train a Model
We're finally ready to actually train a model. We've set up our model as per
the documentation, in a file called `dnn.py`. There are a couple of model file
examples in the `config_files` folder in this repo.

```bash
$ landshark train dnn.py traintest_myproblem_fold1of10
```
This will start the training off, first creating a folder to store the model
checkpoints called `model_dnn`

### 4. Predict with the trained model
Because we've already extracted the query data, this is as simple as

```bash
$ landshark query model_dnn query_myproblem_strip1of1
```
The prediction images will be saved to the model folder.


## Landshark Commands

The Following section describes all landshark commands, sub-commands and
options. The major commands in landshark are:

Command | Description
| --- | --- |
`landshark-import` | Import geotiff features and shapefile targets into landshark-compatible formats
`landshark-extract` | Extract train/test data and query data from imported features and targets
`landshark` | Train a model and make predictions
`skshark` | Train a scikit-learn model for comparison/baseline purposes
`landshark-dump` | Dump train/test and query data into HDF5 for use in external tools

There are two global options for all these commands:

Option | Argument | Description
| --- | --- | --- |
`-v,--verbosity` | `DEBUG\|INFO\|WARNING\|ERROR` | Level of logging
`--help` | | Print command help including option descriptions


### landshark-import

`landshark-import` is the first stage of building models with Landshark. It
takes the input data for the problem (features and targets), and performs some
light preliminary processing to make it easier to handle further down the
pipeline. It has two subcommands, `landshark-import tifs` and `landshark-import
targets`.

Optional Arguments:

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--nworkers` | `INT>=0` | number of cores | The number of *additional* worker processes beyond the parent process. Setting this value to 0 disables multiprocessing entirely. The default is the number of logical CPUs python has detected.
`--batch-mb` | `FLOAT>0` | 100 | The approximate size, in megabytes of data read per worker and per iteration. See Memory Usage for details.

#### tifs

The `tifs` subcommand takes a set of geotiff files and builds a single image stack
for fast reading by Landshark.

The output of this operation is a 'feature stack' called
`features_<name>.hdf5`.

Required flags:

Flag | Argument | Description
| --- | --- | --- |
`--name` | `STRING` | A name describing the feature set being constructed.
`--ordinal` | `DIRECTORY` | A directory containing ordinal (continuous-valued) geotiffs. This argument can be given multiple times with different folders. May be omitted, but at least one of `--ordinal` or `--categorical` must be given.
`--categorical` | `DIRECTORY` | A directory containing categorical geotiffs. This argument can be given multiple times with different folders. May be omitted, but at least one of `--ordinal` or `--categorical` must be given.

Optional arguments:

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--normalise/--no-normalise` | | `TRUE` | Whether to normalise each ordinal tif band to have mean 0 and standard deviation 1. Normalising is highly recommended for learning.
`--ignore-crs/--no-ignore-crs` | | `FALSE` | Whether to enforce the CRS data being identical for all images. Default is no-ignore, but if you know what you're doing...


#### targets

The `targets` subcommand takes a shapefile and extracts a set of points and
records from it to use as targets. At the moment the datatype of the records
must be the same, i.e. all categorical or ordinal. The points are also shuffled
on import (deterministically).

The output of the operation is a  target file called `targets_<name>.hdf5`.

Required Flags:

Flag | Argument | Description
| --- | --- | --- |
`--name` | `STRING` | A name describing the target set being constructed.
`--shapefile` | `SHAPEFILE` | The shapefile from which to extract. Use the actual `.shp` file here.
`--record` | `STRING` | A record to extract for each point as a target. This argument can be given multiple times to extract multiple records.
`--dtype` | `[ordinal\|categorical]` | The type of target, either ordinal for regression or categorical for classification.

Optional Arguments:

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--normalise` | | `FALSE` | Whether to normalise each target column to have mean 0 and standard deviation 1.
`--random_seed` | `INT` | 666 | The initial state of the random number generator used to shuffle the targets on import.
`--every` | `INT>0` | 1 | Factor by which to subsample the data (after shuffling). For example `--every 2` will extract half the targets.

### landshark-extract


Option | Argument | Default | Description
| --- | --- | --- | --- |
`--nworkers` | `INT>=0` | number of cores | The number of *additional* worker processes beyond the parent process. Setting this value to 0 disables multiprocessing entirely. The default is the number of logical CPUs python has detected.
`--batch-mb` | `FLOAT>0` | 100 | The approximate size in megabytes of data read per worker and per iteration. See Memory Usage for details.


#### traintest

Required Flags:

Flag | Argument | Description
| --- | --- | --- |
`--name` | `STRING` | A name describing the training dataset being constructed.
`--features` | `FILE` | The landshark HDF5 feature file from which to extract
`--targets` | `FILE` | The landshark HDF5 target file from which to extract


Optional Arguments:

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--split` | `INT>0` `INT>0` | 1 10 | The specification of folds for the train/test split.  For example, `--split 1 10` uses fold 1 of 10 for testing. Repeated extractions with different folds allows for k-fold cross validation.
`--halfwith` | `INT>=0` | 0 | The size of the patch to extract around each target, such that 0 is no patch, 1 is a 3x3 patch, 2 is 5x5 etc...
`--withfeat` | `STRING` | | Extract the given feature from the feature file.  Can be used multiple times for multiple features. Incompatible with `--withoutfeat` and `--withlist`.
`--withoutfeat` | `STRING` | | Don't extract the given feature from the feature file. Can be used multiple times for multiple features. Incompatible with `--withfeat` and `--withlist`.
`--withlist` | `TXTFILE` | | Provide a newline-separated  (and \#-commented) text file with a list of features to extract. Incompatible with `--withfeat` and `--withoutfeat`.

#### query


Required Flags:

Flag | Argument | Description
| --- | --- | --- |
`--name` | `STRING` | A name describing the training dataset being constructed.
`--features` | `FILE` | The landshark HDF5 feature file from which to extract


Optional Arguments:

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--strip` | `INT>0` `INT>0` | 1 1 | The horizontal strip of the image to extract.  The second argument is the number of horizontal strips to divide the image, the first argument is the index (from 1) of those strips. For example, `--strip 3 5` is the 3rd strip of 5. 
`--halfwith` | `INT>=0` | 0 | The size of the patch to extract around each target, such that 0 is no patch, 1 is a 3x3 patch, 2 is 5x5 etc...
`--withfeat` | `STRING` | | Extract the given feature from the feature file.  Can be used multiple times for multiple features. Incompatible with `--withoutfeat` and `--withlist`.
`--withoutfeat` | `STRING` | | Don't extract the given feature from the feature file. Can be used multiple times for multiple features. Incompatible with `--withfeat` and `--withlist`.
`--withlist` | `TXTFILE` | | Provide a newline-separated  (and \#-commented) text file with a list of features to extract. Incompatible with `--withfeat` and `--withoutfeat`.


### landshark

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--batch-mb` | `FLOAT>0` | 100 | The approximate size in megabytes of data read per worker and per iteration. See Memory Usage for details.
`--gpu/--no-gpu` | | `FALSE` | Whether to use the GPU (rather than the CPU) as the primary tensorflow device.


#### train


Required Flags:

Flag | Argument | Description
| --- | --- | --- |
`--data` | `DIRECTORY` | The traintest data directory containing the training and testing data.
`--config` | `FILE` | The path to the model configuration file.


Optional Arguments:

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--epochs` | `INT>0` | 1 | The number of epochs to train before evaluating the current parameters on the test set.
`--batchsize` | `INT>0` | 1000 | The size of the minibatch for one iteration of stochastic gradient descent.
`--samples` | `INT>0` | 5 | The number of times to sample the parameter distributions for training.
`--test_samples` | `INT>0` | 20| The number of times to sample the parameter distributions for testing.
`--learnrate` | `FLOAT>0` | 0.01 | The learning rate to pass to the ADAM optimiser.
`--test_batchsize` | `INT>0` | 1000 | The size of the batch to evalue the test data.
`--iterations` | `INT>0` |  | If specified, limits the training to the supplied number of  train/test iterations. Default is to train indefinitely.


#### predict

Required Flags:

Flag | Argument | Description
| --- | --- | --- |
`--data` | `DIRECTORY` | The directory containing the query data.
`--model` | `DIRECTORY` | The trained model directory.


Optional Arguments:

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--samples` | `INT>0` | 20 | The number of times to sample the parameter distributions for prediction.
`--lower` | `0<INT<100` | 10 | The lower percentile bound of the predictive posterior to output.
`--upper` | `0<INT<100` | 90 | The upper percentile bound of the predictive posterior to output.


### skshark

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--batch-mb` | `FLOAT>0` | 100 | The approximate size in megabytes of data read per worker and per iteration. See Memory Usage for details.


#### train

Required Flags:

Flag | Argument | Description
| --- | --- | --- |
`--data` | `DIRECTORY` | The traintest data directory containing the training and testing data.
`--config` | `FILE` | The path to the model configuration file.


Optional Arguments:

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--maxpoints` | `INT>0` |  | If supplied, limits the number of training points going to the sklearn interface. Useful for very big datasets.
`--random_seed` | `INT` | 666 | A random seed supplied to the sklearn configuration. It is up to the configuration to use it or not, but useful for algorithms like random forest.


#### predict


Required Flags:

Flag | Argument | Description
| --- | --- | --- |
`--data` | `DIRECTORY` | The directory containing the query data.
`--model` | `DIRECTORY` | The trained model directory.

Optional Arguments:

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--lower` | `0<INT<100` | 10 | The lower percentile bound of the predictive posterior to output. Provided to the sklearn configuration which may or may not implement it.
`--upper` | `0<INT<100` | 90 | The upper percentile bound of the predictive posterior to output. Provided to the sklearn configuration which may or may not implement it.


### landshark-dump

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--nworkers` | `INT>=0` | number of cores | The number of *additional* worker processes beyond the parent process. Setting this value to 0 disables multiprocessing entirely. The default is the number of logical CPUs python has detected.
`--batch-mb` | `FLOAT>0` | 100 | The approximate size in megabytes of data read per worker and per iteration. See Memory Usage for details.


#### traintest

Required Flags:

Flag | Argument | Description
| --- | --- | --- |
`--name` | `STRING` | A name describing the training dataset being constructed.
`--features` | `FILE` | The landshark HDF5 feature file from which to extract
`--targets` | `FILE` | The landshark HDF5 target file from which to extract


Optional Arguments:

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--nfolds` | `INT>0` | 10 | The number of folds into which to assign training folds. The fold number gets written into the HDF5 so that train/test splits can be directly compared with other tools.
`--halfwith` | `INT>=0` | 0 | The size of the patch to extract around each target, such that 0 is no patch, 1 is a 3x3 patch, 2 is 5x5 etc...

#### query


Required Flags:

Flag | Argument | Description
| --- | --- | --- |
`--name` | `STRING` | A name describing the training dataset being constructed.
`--features` | `FILE` | The landshark HDF5 feature file from which to extract


Optional Arguments:

Option | Argument | Default | Description
| --- | --- | --- | --- |
`--strip` | `INT>0` `INT>0` | 1 1 | The horizontal strip of the image to dump.  The second argument is the number of horizontal strips to divide the image, the first argument is the index (from 1) of those strips. For example, `--strip 3 5` is the 3rd strip of 5. 
`--halfwith` | `INT>=0` | 0 | The size of the patch to extract around each target, such that 0 is no patch, 1 is a 3x3 patch, 2 is 5x5 etc...



## Design choices

### Intermediate HDF5 Format

Reading a few pixels from 100 (often compressed) geotiffs is slow. Reading
the same data from an HDF5 file where the 100 bands are contiguous is fast.
Therefore, if we need to do more than a couple of reads from that big geotiff
stack, it saves a lot of time to first re-order all that data to
'band-interleaved by pixel'. We could do this back to geotiff, but HDF5 is
convenient and allows us to store some extra data (and reading is higher
performance too). 


### Configuration as Code

The model specification files in landshark are python functions (or classes in
the case of the sklearn interface). Whilst this presents a larger barrier for
new users if they're not familiar with Python, it allows users to customise
models completely to their particular use-case. A classic example the authors
often come across are custom likelihood functions in Bayesian algorithms.


## Memory Usage

### Importing, Extracting and Dumping

### Training and Predicting


## Writing Model Configuration Files


### landshark


### skshark






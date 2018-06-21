# landshark

Large-scale spatial inference with Tensorflow.

Designed to work with Aboleth (github.com/data61/aboleth).


## Introduction

Landshark is a set of python command line tools that for supervised learning problems on large spatial raster datasets (with sparse targets).
It solves problems in which the user has a set of target point measurements (such as geochemistry, soil classification, or depth to basement) and wants to relate those to a number of raster covariates (like satellite imagery or geophysics) to predict the targets on the raster grid.

Many such tools exist already for this problem. Landshark fills a particular
niche: where we want to efficiently learn models with very large numbers of
training points and/or very large covariate images using tensorflow. Landshark
is particularly useful for the case when the training data itself will not fit
in memory, and must be streamed to a minibatch stochastic gradient descent
algorithm for model learning.

There are no actual models in landshark. It is really just a training and query
data conversion system, to get shapefiles and geotifs to and from 'tfrecord'
files which can be read efficiently by tensorflow.

The choice of models is up to the user: arbitary tensorflow models are fine, as
is anything built on top of tensorflow like Keras or Aboleth. There's also
a scikit-learn interface for problems that do fit in memory (although this is
mainly for validation purposes).


## Data Prerequisites

Landshark has not tried to replicate features that exist in other tools,
especilaly image processing and GIS. Therefore, it has quite strict
requirements for data input:

1. Targets are stored as points in a shapefile
2. Covariates are stored as a set of geotifs (1 or more)
3. All geotifs have the same resolution and bounding box
4. All target points are inside the bounding box of the geotifs
5. The covariate images are also those used for prediction (i.e, the prediction
   image will come out with the same resultion and bounding box as the
   covariate images)
6. Geotiffs may have multiple bands, and different geotiffs may have different
   datatypes (eg uint8, float32), but within a single geotiff all bands must
   have the same datatype
7. Geotiffs have been categorized into continous data (referred to as 'ordinal'), 
   and categorical data. These two sets of geotiffs are stored in separate
   folders.

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


## Usage Example

We have say a dozen ".tif" covariates stored between our 'ord_images' and
'cat_images' folder. We have a target shapefile in our 'targets' folder.

1. Import the data

We run
$ landshark-import tifs --ordinal ord_images/ --categorical cat_images --name myfeats

now landshark runs and spits out "myfeats_features.hdf5". Similarly for our
targets, we run
$ landshark-import targets --name mytarg --dtype categorical --record ROCK_CLASS --record ROCK_TYPE

This command indicates that we're interested in the ROCK_CLASS and ROCK_TYPE
records in our shapefile, and that these values are categorical. This command
outputs "mytarg_targets.hdf5".

2. Extract train/test and query data

We run
$ landshark-extract --features features_myfeats.hdf5 traintest --split 1 10
--halfwidth 0





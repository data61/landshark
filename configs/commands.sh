#! /usr/bin/env bash

# Commands from integration tests

# keras-model regression
landshark-import --batch-mb 0.001 targets \
    --shapefile ../integration/data/targets/geochem_sites.shp \
    --name NaCu \
    --record Na_ppm_i_1 \
    --record Cu_ppm_i_1 \
    --dtype continuous

landshark-import --nworkers 0 --batch-mb 0.001 tifs \
    --continuous ../integration/data/continuous \
    --name sirsam \
    --ignore-crs

landshark-extract --nworkers 0 --batch-mb 0.001 traintest \
    --features features_sirsam.hdf5 \
    --split 1 10 \
    --targets targets_NaCu.hdf5 \
    --name SirsamNaCu \
    --halfwidth 1

landshark-extract query \
    --features features_sirsam.hdf5 \
    --strip 5 10 \
    --name sirsam1 \
    --halfwidth 1

landshark --keras-model --no-gpu train \
    --config ../configs/nn_regression_keras.py \
    --data traintest_SirsamNaCu_fold1of10 \
    --epochs 3 \
    --iterations 5


# keras-model classification
landshark-import --batch-mb 0.001 targets \
    --shapefile ../integration/data/targets/geochem_sites.shp \
    --name NaCu \
    --record Na_ppm_i_1 \
    --record Cu_ppm_i_1 \
    --dtype continuous

landshark-import --batch-mb 0.001 targets \
    --shapefile ../integration/data/targets/geochem_sites.shp \
    --name SAMPLETYPE \
    --record SAMPLETYPE \
    --dtype categorical

landshark-extract --nworkers 0 --batch-mb 0.001 traintest \
    --features features_sirsam.hdf5 \
    --split 1 10 \
    --targets targets_SAMPLETYPE.hdf5 \
    --name sirsam

landshark-extract --nworkers 0 --batch-mb 0.001 query \
    --features features_sirsam.hdf5 \
    --strip 5 10 \
    --name sirsam

landshark --keras-model train \
    --data traintest_sirsam_fold1of10 \
    --config /home/col540/dev/ml/landshark2/configs/nn_classification_keras.py \
    --epochs 200 \
    --iterations 5

landshark --keras-model --batch-mb 0.001 predict \
    --config /home/col540/dev/ml/landshark2/configs/nn_classification_keras.py \
    --checkpoint nn_classification_keras_model_1of10 \
    --data query_sirsam_strip5of10

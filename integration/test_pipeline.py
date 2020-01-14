"""Tests for the whole pipeline."""

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

import os
import shutil
import subprocess
from glob import glob
from typing import Any

import pytest
from _pytest.fixtures import FixtureRequest

# these data are tiny so we need a really
# small batch size to emulate normal use
BATCH_MB = 0.001


commands = {
    "landshark": ["landshark"],
    "landshark-keras": ["landshark", "--keras-model"],
    "skshark": ["skshark"],
}

model_files = {
    "regression": {
        "landshark": "nn_regression.py",
        "landshark-keras": "nn_regression_keras.py",
        "skshark": "sklearn_regression.py"
    },
    "classification": {
        "landshark": "nn_classification.py",
        "landshark-keras": "nn_classification_keras.py",
        "skshark": "sklearn_classification.py"
    }
}

training_args = {
    "landshark": ["--epochs", "200", "--iterations", "5"],
    "landshark-keras": ["--epochs", "200", "--iterations", "5"],
    "skshark": []
}

target_files = {
    "regression": {
        "target": "Na_ppm_i_1",
        "args": ["--dtype", "continuous"]
    },
    "classification": {
        "target": "SAMPLETYPE",
        "args": ["--dtype", "categorical"]
    }
}


@pytest.fixture(params=["continuous-only", "categorical-only", "both"])
def whichfeatures(request: FixtureRequest) -> Any:
    return request.param


@pytest.fixture(params=["regression", "classification"])
def whichproblem(request: FixtureRequest) -> Any:
    return request.param


@pytest.fixture(params=[0, 2])
def number_of_cpus(request: FixtureRequest) -> Any:
    return request.param


@pytest.fixture(params=[0, 1])
def half_width(request: FixtureRequest) -> Any:
    return request.param


@pytest.fixture(params=["landshark", "skshark", "landshark-keras"])
def whichalgo(request: FixtureRequest) -> Any:
    return request.param


def import_tifs(cat_dir, con_dir, feature_string, ncpus):
    tif_import_args = ["--categorical", cat_dir, "--continuous", con_dir,
                       "--ignore-crs"]
    if feature_string == "continuous-only":
        tif_import_args = tif_import_args[2:]
    elif feature_string == "categorical-only":
        tif_import_args = tif_import_args[:2] + ["--ignore-crs"]
    _run(["landshark-import", "--nworkers", ncpus,
          "--batch-mb", BATCH_MB, "tifs", "--name", "sirsam"
          ] + tif_import_args)
    feature_file = "features_sirsam.hdf5"
    assert os.path.isfile(feature_file)
    return feature_file


def import_targets(target_dir, target_name, target_flags, ncpus):
    target_file = os.path.join(target_dir, "geochem_sites.shp")
    _run(["landshark-import", "--batch-mb", BATCH_MB, "targets",
          "--shapefile", target_file, "--name", target_name,
          "--record", target_name] + target_flags)
    target_file = "targets_{}.hdf5".format(target_name)
    assert os.path.isfile(target_file)
    return target_file


def extract_training_data(target_file, target_name, ncpus):
    _run(["landshark-extract", "--nworkers", ncpus, "--batch-mb", BATCH_MB,
          "traintest", "--features", "features_sirsam.hdf5", "--split", 1, 10,
          "--targets", target_file, "--name", "sirsam"])
    trainingdata_folder = "traintest_sirsam_fold1of10"
    assert os.path.isdir(trainingdata_folder)
    return trainingdata_folder


def extract_query_data(feature_file, ncpus):
    _run(["landshark-extract", "--nworkers", ncpus, "--batch-mb", BATCH_MB,
          "query", "--features", feature_file, "--strip", 5, 10,
          "--name", "sirsam"])
    querydata_folder = "query_sirsam_strip5of10"
    assert os.path.isdir(querydata_folder)
    return querydata_folder


def train(cmd, model_dir, model_filename, trainingdata_folder,
          training_args):
    _run(cmd + ["train", "--data", trainingdata_folder,
                 "--config", model_filename] + training_args)
    trained_model_dir = "{}_model_1of10".format(
        os.path.basename(model_filename).split(".py")[0])
    assert os.path.isdir(trained_model_dir)
    return trained_model_dir


def predict(cmd, model_filename, trained_model_dir,
            querydata_folder, target_name):
    _run(cmd + ["--batch-mb", BATCH_MB, "predict",
                  "--config", model_filename,
                  "--checkpoint", trained_model_dir,
                  "--data", querydata_folder])
    image_filename = "predictions_{}_5of10.tif".format(target_name)
    image_path = os.path.join(trained_model_dir, image_filename)
    assert os.path.isfile(image_path)


def _run(cmd):
    """Execute CLI command  using subprocess."""
    cmd_str = [str(k) for k in cmd]
    print("Running command: {}".format(" ".join(cmd_str)))
    proc = subprocess.run(cmd_str, stdout=subprocess.PIPE)
    assert proc.returncode == 0


def test_full_pipeline(tmpdir, data_loc, whichfeatures, whichproblem,
                       whichalgo, number_of_cpus, half_width):
    con_dir, cat_dir, target_dir, model_dir, result_dir = data_loc
    os.chdir(os.path.abspath(tmpdir))
    ncpus = number_of_cpus
    thisrun = "{}_{}_{}_{}cpus_hw{}".format(whichalgo, whichproblem,
                                            whichfeatures, ncpus, half_width)
    print("Current run: {}".format(thisrun))

    target_name = target_files[whichproblem]["target"]
    target_flags = target_files[whichproblem]["args"]
    model_filename = model_files[whichproblem][whichalgo]
    train_args = training_args[whichalgo]
    model_path = os.path.join(model_dir, model_filename)

    # need to make isolated filesystem
    print("Importing tifs...")
    feature_file = import_tifs(cat_dir, con_dir, whichfeatures, ncpus)
    print("Importing targets...")
    target_file = import_targets(target_dir, target_name, target_flags,
                                 ncpus)
    print("Extracting training data...")
    trainingdata_folder = extract_training_data(target_file,
                                                target_name, ncpus)

    print("Extracting query data...")
    querydata_folder = extract_query_data(feature_file, ncpus)
    print("Training...")
    trained_model_dir = train(commands[whichalgo], model_dir, model_path,
                              trainingdata_folder, train_args)
    print("Predicting...")
    predict(commands[whichalgo], model_path, trained_model_dir,
            querydata_folder, target_name)
    print("Cleaning up...")

    this_result_dir = os.path.join(result_dir, thisrun)
    shutil.rmtree(this_result_dir, ignore_errors=True)
    os.makedirs(this_result_dir)

    images = glob(os.path.join(trained_model_dir, "*.tif"))
    for im in images:
        shutil.move(im, this_result_dir)
    shutil.rmtree(trained_model_dir, ignore_errors=True)

"""Tests for the whole pipeline."""
import os
import shutil
import subprocess
import traceback
from glob import glob

import pytest

from landshark.scripts import cli, extractors, importers, skcli

# these data are tiny so we need a really
# small batch size to emulate normal use
BATCH_MB = 0.001

model_files = {"regression": {"landshark": "nnr_aboleth.py",
                              "skshark": "sklearn_rfr.py"},
               "classification": {"landshark": "nnc_aboleth.py",
                                  "skshark": "sklearn_rfc.py"}}

training_args = {"landshark": ["--epochs", "200", "--iterations", "5"],
                 "skshark": []}

target_files = {"regression": {"target": "Na_ppm_i_1",
                               "args": ["--dtype", "continuous"]},
                "classification": {"target": "SAMPLETYPE",
                                   "args": ["--dtype", "categorical"]}}


@pytest.fixture(params=["continuous-only", "categorical-only", "both"])
def whichfeatures(request):
    return request.param

@pytest.fixture(params=["regression", "classification"])
def whichproblem(request):
    return request.param

@pytest.fixture(params=[0, 2])
def number_of_cpus(request):
    return request.param

@pytest.fixture(params=[0, 1])
def half_width(request):
    return request.param


@pytest.fixture(params=["landshark", "skshark"])
def whichalgo(request):
    return request.param


def _run(cmd):
    cmd_str = [str(k) for k in cmd]
    print("Runinng command: {}".format(cmd_str))
    proc = subprocess.run(cmd_str, stdout=subprocess.PIPE)
    assert proc.returncode == 0


def import_tifs(cat_dir, con_dir, feature_string, ncpus):
    tif_import_args = ["--categorical", cat_dir, "--continuous", con_dir,
                       "--ignore-crs"]
    if feature_string == "continuous-only":
        tif_import_args = tif_import_args[2:]
    elif feature_string == "categorical-only":
        tif_import_args = tif_import_args[:2] + ["--ignore-crs"]
    _run(["landshark-import", "--nworkers", ncpus, "--batch-mb", BATCH_MB,
          "tifs", "--name", "sirsam"] + tif_import_args)
    feature_file = "features_sirsam.hdf5"
    assert os.path.isfile(feature_file)
    return feature_file


def import_targets(target_dir, target_name, target_flags, ncpus):
    target_file = os.path.join(target_dir, "geochem_sites.shp")
    _run(["landshark-import", "--batch-mb", BATCH_MB, "targets",
          "--shapefile", target_file, "--name", target_name] +
         target_flags + ["--record", target_name])
    target_file = "targets_{}.hdf5".format(target_name)
    assert os.path.isfile(target_file)
    return target_file


def extract_training_data(target_file, target_name, ncpus):
    _run(["landshark-extract", "--nworkers", ncpus, "--batch-mb",
          BATCH_MB, "traintest", "--features", "features_sirsam.hdf5",
          "--split", 1, 10, "--targets", target_file, "--name", "sirsam"])
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


def train(cmd, model_dir, model_filename, trainingdata_folder, training_args):
    _run([cmd] + ["train"] + training_args +
         ["--data", trainingdata_folder, "--config", model_filename])
    trained_model_dir = "{}_model_1of10".format(
        os.path.basename(model_filename).split(".py")[0])
    assert os.path.isdir(trained_model_dir)
    return trained_model_dir


def predict(cmd, model_filename, trained_model_dir,
            querydata_folder, target_name):
    _run([cmd] + ["--batch-mb", BATCH_MB, "predict",
                  "--config", model_filename,
                  "--checkpoint", trained_model_dir,
                  "--data", querydata_folder])
    image_filename = "predictions_5of10.tif"
    image_path = os.path.join(trained_model_dir, image_filename)
    assert os.path.isfile(image_path)


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
    target_file = import_targets(target_dir, target_name, target_flags, ncpus)
    print("Extracting training data...")
    trainingdata_folder = extract_training_data(target_file,
                                                target_name, ncpus)
    print("Extracting query data...")
    querydata_folder = extract_query_data(feature_file, ncpus)
    print("Training...")
    trained_model_dir = train(whichalgo, model_dir, model_path,
                              trainingdata_folder, train_args)
    print("Predicting...")
    predict(whichalgo, model_path, trained_model_dir,
            querydata_folder, target_name)
    print("Cleaning up...")

    this_result_dir = os.path.join(result_dir, thisrun)
    shutil.rmtree(this_result_dir, ignore_errors=True)
    os.makedirs(this_result_dir)

    images = glob(os.path.join(trained_model_dir, "*.tif"))
    for im in images:
        shutil.move(im, this_result_dir)
    shutil.rmtree(trained_model_dir, ignore_errors=True)

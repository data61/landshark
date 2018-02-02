"""Tests for the whole pipeline."""
import traceback
import os
import shutil
from glob import glob

from click.testing import CliRunner
import pytest
import tensorflow as tf

from landshark.scripts import importers, cli, skcli

NCPUS = 2


model_files = {"regression": {"landshark": "nnr.py",
                              "skshark": "sklearn_rfr.py"},
               "classification": {"landshark": "nnc.py",
                                  "skshark": "sklearn_rfc.py"}}

target_files = {"regression": {"target": "Na_ppm_i_1",
                               "args": []},
                "classification": {"target": "LITHNAME",
                                   "args": ["--categorical"]}}

training_args = {"landshark": ["--epochs", "200", "--iterations", "5"],
                 "skshark": []}

@pytest.fixture(params=["ordinal-only", "categorical-only", "both"])
def whichfeatures(request):
    return request.param


@pytest.fixture(params=["regression", "classification"])
def whichproblem(request):
    return request.param


@pytest.fixture(params=["landshark", "skshark"])
def whichalgo(request):
    return request.param


def _run(runner, cmd, args):
    results = runner.invoke(cmd, args)
    for line in results.output.split("\n"):
        print(line)
    if results.exit_code != 0:
        print("command {} failed with exception {}. Traceback:\n".format(
            args, results.exception))
        traceback.print_tb(results.exc_info[2])
    assert results.exit_code == 0


def import_tifs(runner, cat_dir, ord_dir, feature_string):
    # Import tifs
    tif_import_args = ["--categorical", cat_dir, "--ordinal", ord_dir]
    if feature_string == "ordinal-only":
        tif_import_args = tif_import_args[2:]
    elif feature_string == "categorical-only":
        tif_import_args = tif_import_args[:2]
    _run(runner, importers.cli, ["tifs", "--nworkers", NCPUS,
                                 "--name", "sirsam"] + tif_import_args)
    feature_file = "sirsam_features.hdf5"
    assert os.path.isfile(feature_file)
    return feature_file


def import_targets(runner, target_dir, target_name, target_flags):
    # Import targets
    target_file = os.path.join(target_dir, "geochem_sites.shp")
    _run(runner, importers.cli, ["targets", "--nworkers", NCPUS,
                                 "--shapefile", target_file,
                                 "--name", target_name] +
         target_flags + [target_name])
    target_file = "{}_targets.hdf5".format(target_name)
    assert os.path.isfile(target_file)
    return target_file


def import_training_data(runner, target_file, target_name):
    # Import training data
    _run(runner, importers.cli,
         ["trainingdata", "--nworkers", NCPUS,
          "sirsam_features.hdf5", target_file])
    trainingdata_folder = "sirsam-{}_trainingdata".format(target_name)
    assert os.path.isdir(trainingdata_folder)
    return trainingdata_folder


def import_query_data(runner, feature_file):
    # Import query data
    _run(runner, importers.cli, ["querydata", "--nworkers", NCPUS,
                                 "--features", feature_file, "5", "10"])
    querydata_folder = "sirsam_features_query5of10"
    assert os.path.isdir(querydata_folder)
    return querydata_folder

def train(runner, module, model_dir, model_filename, trainingdata_folder,
          training_args):
    model_file = os.path.join(model_dir, model_filename)
    _run(runner, module.cli, ["train"] +
         training_args + [trainingdata_folder, model_file])
    trained_model_dir = "{}_model".format(model_filename.split(".py")[0])
    assert os.path.isdir(trained_model_dir)
    return trained_model_dir


def predict(runner, module, model_dir, trained_model_dir,
            querydata_folder, target_name):
    _run(runner, module.cli, ["predict", trained_model_dir, querydata_folder])
    image_filename = "{}_5of10.tif".format(target_name)
    image_path = os.path.join(trained_model_dir, image_filename)
    assert os.path.isfile(image_path)


def test_full_pipeline(data_loc, whichfeatures, whichproblem, whichalgo):
    ord_dir, cat_dir, target_dir, model_dir, result_dir = data_loc

    thisrun = "{}_{}_{}".format(whichalgo, whichproblem, whichfeatures)
    print("Current run: {}".format(thisrun))

    target_name = target_files[whichproblem]["target"]
    target_flags = target_files[whichproblem]["args"]
    model_filename = model_files[whichproblem][whichalgo]
    train_args = training_args[whichalgo]

    module = cli if whichalgo == "landshark" else skcli

    runner = CliRunner()
    with runner.isolated_filesystem():
        tf.reset_default_graph()
        feature_file = import_tifs(runner, cat_dir, ord_dir, whichfeatures)
        target_file = import_targets(runner, target_dir, target_name,
                                     target_flags)
        trainingdata_folder = import_training_data(runner, target_file,
                                                   target_name)
        querydata_folder = import_query_data(runner, feature_file)
        trained_model_dir = train(runner, module, model_dir, model_filename,
                                  trainingdata_folder, train_args)
        predict(runner, module, model_dir, trained_model_dir,
                querydata_folder, target_name)

        this_result_dir = os.path.join(result_dir, thisrun)
        shutil.rmtree(this_result_dir, ignore_errors=True)
        os.makedirs(this_result_dir)

        images = glob(os.path.join(trained_model_dir, "*.tif"))
        for im in images:
            shutil.move(im, this_result_dir)
        shutil.rmtree(trained_model_dir, ignore_errors=True)


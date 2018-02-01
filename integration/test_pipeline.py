"""Tests for the whole pipeline."""
import os
import shutil
from glob import glob

from click.testing import CliRunner
import pytest

from landshark.scripts import importers, cli, skcli


@pytest.fixture(params=["ordinal-only", "categorical-only", "both"])
def whichfeatures(request):
    return request.param

@pytest.fixture(params=[("Na_ppm_i_1", "Na", [], "sklearn_rfr.py"),
                        ("LITHNAME", "lith", ["--categorical"],
                         "sklearn_rfc.py")])
def whichtarget(request):
    return request.param

@pytest.fixture(params=["skshark"])
def whichalgo(request):
    return request.param


def _run(runner, cmd, args):
    results = runner.invoke(cmd, args)
    for line in results.output.split("\n"):
        print(line)
    assert results.exit_code == 0

def import_tifs(runner, cat_dir, ord_dir):
    # Import tifs
    tif_import_args = ["--categorical", cat_dir, "--ordinal", ord_dir]
    if whichfeatures == "ordinal-only":
        tif_import_args = tif_import_args[2:]
    elif whichfeatures == "categorical-only":
        tif_import_args = tif_import_args[:2]
    _run(runner, importers.cli, ["tifs", "--name", "sirsam"] +
         tif_import_args)
    feature_file = "sirsam_features.hdf5"
    assert os.path.isfile(feature_file)
    return feature_file


def import_targets(runner, target_dir, target_name, target_flags, target_lbl):
    # Import targets
    target_file = os.path.join(target_dir, "geochem_sites.shp")
    _run(runner, importers.cli, ["targets", "--shapefile",
                                 target_file, "--name", target_name] +
         target_flags + [target_lbl])
    target_file = "{}_targets.hdf5".format(target_name)
    assert os.path.isfile(target_file)
    return target_file


def import_training_data(runner, target_file, target_name):
    # Import training data
    _run(runner, importers.cli,
         ["trainingdata", "sirsam_features.hdf5", target_file])
    trainingdata_folder = "sirsam-{}_trainingdata".format(target_name)
    assert os.path.isdir(trainingdata_folder)
    return trainingdata_folder


def import_query_data(runner, feature_file):
    # Import query data
    _run(runner, importers.cli, ["querydata", "--features",
                                 feature_file, "5", "10"])
    querydata_folder = "sirsam_features_query5of10"
    assert os.path.isdir(querydata_folder)
    return querydata_folder

def train_sklearn(runner, model_dir, model_filename, trainingdata_folder):
    # train a model
    model_file = os.path.join(model_dir, model_filename)
    _run(runner, skcli.cli, ["train", trainingdata_folder, model_file])
    trained_model_dir = "{}_model".format(model_filename.split(".py")[0])
    assert os.path.isdir(trained_model_dir)
    return trained_model_dir


def predict_sklearn(runner, model_dir, trained_model_dir,
                    querydata_folder, target_lbl):
    # make a prediction
    # model_file = os.path.join(model_dir, model_filename)
    _run(runner, skcli.cli, ["predict", trained_model_dir, querydata_folder])
    image_filename = "{}_5of10.tif".format(target_lbl)
    image_path = os.path.join(trained_model_dir, image_filename)
    assert os.path.isfile(image_path)


def test_full_pipeline(data_loc, whichfeatures, whichtarget, whichalgo):
    ord_dir, cat_dir, target_dir, model_dir, result_dir = data_loc
    target_lbl, target_name, target_flags, model_filename = whichtarget

    runner = CliRunner()
    with runner.isolated_filesystem():
        feature_file = import_tifs(runner, cat_dir, ord_dir)
        target_file = import_targets(runner, target_dir, target_name,
                                     target_flags, target_lbl)
        trainingdata_folder = import_training_data(runner, target_file,
                                                   target_name)
        querydata_folder = import_query_data(runner, feature_file)
        trained_model_dir = train_sklearn(runner, model_dir, model_filename,
                                          trainingdata_folder)
        predict_sklearn(runner, model_dir, trained_model_dir,
                        querydata_folder, target_lbl)
        pass

def train():
    pass


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

@pytest.fixture(params=["landshark", "skshark"])
def whichalgo(request):
    return request.param

def test_train_predict(data_loc, whichfeatures, whichtarget, whichalgo):
    ord_dir, cat_dir, target_dir, model_dir, result_dir = data_loc
    target_lbl, target_name, target_flags, model_filename = whichtarget

    runner = CliRunner()
    with runner.isolated_filesystem():

        # Import tifs
        tif_import_args = ["--categorical", cat_dir, "--ordinal", ord_dir]
        if whichfeatures == "ordinal-only":
            tif_import_args = tif_import_args[2:]
        elif whichfeatures == "categorical-only":
            tif_import_args = tif_import_args[:2]
        results = runner.invoke(importers.cli, ["tifs", "--name", "sirsam"] +
                                tif_import_args)
        for line in results.output.split("\n"):
            print(line)
        assert results.exit_code == 0
        assert os.path.isfile("sirsam_features.hdf5")

        # Import targets
        target_file = os.path.join(target_dir, "geochem_sites.shp")
        results = runner.invoke(importers.cli, ["targets",
                                                "--shapefile", target_file,
                                                "--name", target_name] +
                                                target_flags + [target_lbl])
        for line in results.output.split("\n"):
            print(line)
        assert results.exit_code == 0
        target_file = "{}_targets.hdf5".format(target_name)
        assert os.path.isfile(target_file)

        # Import training data
        results = runner.invoke(importers.cli, ["trainingdata",
                                                "sirsam_features.hdf5",
                                                target_file])
        for line in results.output.split("\n"):
            print(line)
        assert results.exit_code == 0
        trainingdata_folder = "sirsam-{}_trainingdata".format(target_name)
        assert os.path.isdir(trainingdata_folder)

        # Import query data
        results = runner.invoke(importers.cli, ["querydata",
                                                "--features",
                                                "sirsam_features.hdf5",
                                                "5", "10"])
        for line in results.output.split("\n"):
            print(line)
        assert results.exit_code == 0
        querydata_folder = "sirsam_features_query5of10"
        assert os.path.isdir(querydata_folder)

        # train a model
        model_file = os.path.join(model_dir, model_filename)
        results = runner.invoke(skcli.cli, ["train", trainingdata_folder,
                                            model_file])
        for line in results.output.split("\n"):
            print(line)
        assert results.exit_code == 0
        trained_model_dir = "{}_model".format(model_filename.split(".py")[0])
        assert os.path.isdir(trained_model_dir)

        # make a prediction
        model_file = os.path.join(model_dir, model_filename)
        results = runner.invoke(skcli.cli, ["predict", trained_model_dir,
                                            querydata_folder])
        for line in results.output.split("\n"):
            print(line)
        assert results.exit_code == 0
        image_filename = "{}_5of10.tif".format(target_lbl)
        image_path = os.path.join(trained_model_dir, image_filename)
        assert os.path.isfile(image_path)

        # outputs
        # images = glob(os.path.join(trained_model_dir, "*.tif"))
        # for im in images:
        #     shutil.move(im, result_dir)

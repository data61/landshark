"""Tests for the whole pipeline."""
import traceback
import os
import shutil
from glob import glob

from click.testing import CliRunner
import pytest
import tensorflow as tf

from landshark.scripts import importers, extractors, cli, skcli, dumpers

BATCH_MB = 1

model_files = {"regression": {"landshark": "nnr.py",
                              "skshark": "sklearn_rfr.py"},
               "classification": {"landshark": "nnc.py",
                                  "skshark": "sklearn_rfc.py"}}

target_files = {"regression": {"target": "Na_ppm_i_1",
                               "args": ["--dtype", "ordinal"]},
                "classification": {"target": "SAMPLETYPE",
                                   "args": ["--dtype", "categorical"]}}

training_args = {"landshark": ["--epochs", "200", "--iterations", "5"],
                 "skshark": []}

@pytest.fixture(params=["ordinal-only", "categorical-only", "both"])
def whichfeatures(request):
    return request.param


@pytest.fixture(params=["regression", "classification"])
def whichproblem(request):
    return request.param


@pytest.fixture(params=[0, 1])
def number_of_cpus(request):
    return request.param

@pytest.fixture(params=[0, 1])
def half_width(request):
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


def import_tifs(runner, cat_dir, ord_dir, feature_string, ncpus):
    tif_import_args = ["--categorical", cat_dir, "--ordinal", ord_dir,
                       "--ignore-crs"]
    if feature_string == "ordinal-only":
        tif_import_args = tif_import_args[2:]
    elif feature_string == "categorical-only":
        tif_import_args = tif_import_args[:2] + ["--ignore-crs"]
    _run(runner, importers.cli, ["--nworkers", ncpus, "--batch-mb", BATCH_MB,
                                 "tifs", "--name", "sirsam"] +
         tif_import_args)
    feature_file = "features_sirsam.hdf5"
    assert os.path.isfile(feature_file)
    return feature_file


def import_targets(runner, target_dir, target_name, target_flags, ncpus):
    target_file = os.path.join(target_dir, "geochem_sites.shp")
    _run(runner, importers.cli, ["--batch-mb", BATCH_MB, "targets",
                                 "--shapefile", target_file,
                                 "--name", target_name] +
         target_flags + ["--record", target_name])
    target_file = "targets_{}.hdf5".format(target_name)
    assert os.path.isfile(target_file)
    return target_file


def extract_training_data(runner, target_file, target_name, ncpus):
    _run(runner, extractors.cli,
         ["--nworkers", ncpus, "--batch-mb", BATCH_MB, "traintest",
          "--features", "features_sirsam.hdf5", "--split", 1, 10,
          "--targets", target_file, "--name", "sirsam"])
    trainingdata_folder = "traintest_sirsam_fold1of10"
    assert os.path.isdir(trainingdata_folder)
    return trainingdata_folder

def dump_training_data(runner, target_file, target_name, ncpus):
    _run(runner, dumpers.cli,
         ["--nworkers", ncpus, "--batch-mb", BATCH_MB, "traintest",
          "--features", "features_sirsam.hdf5", "--targets", target_file,
          "--name", "sirsam"])
    trainingdata_dump = "dump_traintest_sirsam.hdf5"
    assert os.path.isfile(trainingdata_dump)


def extract_query_data(runner, feature_file, ncpus):
    _run(runner, extractors.cli, ["--nworkers", ncpus, "--batch-mb", BATCH_MB,
                                  "query", "--features", feature_file,
                                  "--strip", 5, 10, "--name", "sirsam"])
    querydata_folder = "query_sirsam_strip5of10"
    assert os.path.isdir(querydata_folder)
    return querydata_folder


def dump_query_data(runner, feature_file, ncpus):
    _run(runner, dumpers.cli, ["--nworkers", ncpus, "--batch-mb", BATCH_MB,
                               "query", "--features", feature_file,
                               "--strip", 5, 10, "--name", "sirsam"])
    querydata_dump = "dump_query_sirsam_strip5of10.hdf5"
    assert os.path.isfile(querydata_dump)


def train(runner, module, model_dir, model_filename, trainingdata_folder,
          training_args):
    model_file = os.path.join(model_dir, model_filename)
    _run(runner, module.cli, ["train"] +
         training_args + ["--data", trainingdata_folder,
                          "--config", model_file])
    trained_model_dir = "{}_model_1of10".format(model_filename.split(".py")[0])
    assert os.path.isdir(trained_model_dir)
    return trained_model_dir

def predict(runner, module, model_dir, trained_model_dir,
            querydata_folder, target_name):
    _run(runner, module.cli, ["--batch-mb", BATCH_MB, "predict", "--model",
                              trained_model_dir, "--data", querydata_folder])
    image_filename = "{}_5of10.tif".format(target_name)
    image_path = os.path.join(trained_model_dir, image_filename)
    assert os.path.isfile(image_path)


def test_full_pipeline(data_loc, whichfeatures, whichproblem, whichalgo,
                       number_of_cpus, half_width):
    ord_dir, cat_dir, target_dir, model_dir, result_dir = data_loc
    ncpus = number_of_cpus

    thisrun = "{}_{}_{}_{}cpus".format(whichalgo, whichproblem, whichfeatures,
                                       ncpus)
    print("Current run: {}".format(thisrun))

    target_name = target_files[whichproblem]["target"]
    target_flags = target_files[whichproblem]["args"]
    model_filename = model_files[whichproblem][whichalgo]
    train_args = training_args[whichalgo]

    module = cli if whichalgo == "landshark" else skcli

    runner = CliRunner()
    with runner.isolated_filesystem():
        tf.reset_default_graph()
        feature_file = import_tifs(runner, cat_dir, ord_dir, whichfeatures,
                                   ncpus)
        target_file = import_targets(runner, target_dir, target_name,
                                     target_flags, ncpus)
        trainingdata_folder = extract_training_data(runner, target_file,
                                                    target_name, ncpus)
        dump_training_data(runner, target_file, target_name, ncpus)
        querydata_folder = extract_query_data(runner, feature_file, ncpus)
        dump_query_data(runner, feature_file, ncpus)
        trained_model_dir = train(runner, module, model_dir, model_filename,
                                  trainingdata_folder, train_args)
        predict(runner, module, model_dir, trained_model_dir,
                querydata_folder, target_name)

        this_result_dir = os.path.join(result_dir, thisrun)
        shutil.rmtree(this_result_dir, ignore_errors=True)
        os.makedirs(this_result_dir)

        images = glob(os.path.join(trained_model_dir, "*.tif"))
        dumps = glob("dump_*.hdf5")
        for im in images:
            shutil.move(im, this_result_dir)
        for d in dumps:
            shutil.move(d, this_result_dir)
        shutil.rmtree(trained_model_dir, ignore_errors=True)


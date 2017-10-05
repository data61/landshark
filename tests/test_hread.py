import numpy as np
import tables

from landshark import hread
from landshark import rowcache


def test_image_features(mocker):
    """Make sure we get two types (ord, cat) of features from ImageFeatures"""
    root = mocker.MagicMock()
    mocker.patch.object(tables, 'open_file')
    tables.open_file.return_value = root
    feat = mocker.patch.object(hread, 'Features')

    filename = "path"
    imfeat = hread.ImageFeatures(filename, cache_blocksize=20,
                                 cache_nblocks=10)

    tables.open_file.assert_called_with(filename)
    feat.assert_called_with(
        tables.open_file().root.categorical_data,
        tables.open_file().root.categorical_data.attrs.missing_values,
        20, 10
        )

    assert hasattr(imfeat, 'ord')
    assert hasattr(imfeat, 'cat')


def test_features(mocker):
    """Test that we get expected properties from a Features object."""
    carray = mocker.MagicMock()
    carray.atom.dtype.base = np.float32
    carray.atom.shape = [10]
    missing_values = [None, None]

    rcache = mocker.patch("landshark.hread.RowCache")
    feat = hread.Features(carray, missing_values, 20, 10)

    rcache.assert_called_with(carray, 20, 10)
    assert feat.dtype == np.float32
    assert feat.nfeatures == 10
    assert feat.missing_values == missing_values


def test_target(mocker):
    """Test we keep coordinates aligned with labels."""
    root = mocker.MagicMock()
    mocker.patch.object(tables, 'open_file')
    tables.open_file.return_value = root
    imcoords = mocker.patch("landshark.image.coords_training")
    batches_one = np.stack((np.arange(10), np.arange(10))).T[:, :, np.newaxis]
    imcoords.return_value = batches_one

    targs = hread.Targets("path", "label")
    targs._data = np.arange(10)

    batches = targs.training(mocker.MagicMock(), "batch_size")
    for x, y, t in batches:
        assert x == y == t

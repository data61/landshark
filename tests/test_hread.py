import numpy as np

from landshark import hread


def test_image_features(mocker):
    """Make sure we get two types (ord, cat) of features from ImageFeatures"""
    p_openfile = mocker.patch('landshark.hread.tables.open_file')
    feat = mocker.patch('landshark.hread.Features')
    mocker.patch('landshark.hread.image.ImageSpec')

    hfile = p_openfile.return_value
    hfile.root._v_attrs.width = 5
    hfile.root.x_coordinates.read.return_value.__len__.return_value = 6
    hfile.root._v_attrs.height = 5
    hfile.root.y_coordinates.read.return_value.__len__.return_value = 6

    filename = "path"
    imfeat = hread.ImageFeatures(filename, cache_blocksize=20,
                                 cache_nblocks=10)

    p_openfile.assert_called_with(filename)
    feat.assert_called_with(hfile.root.categorical_data, 20, 10)

    assert hasattr(imfeat, 'ord')
    assert hasattr(imfeat, 'cat')


def test_features(mocker):
    """Test that we get expected properties from a Features object."""
    carray = mocker.MagicMock()
    carray.atom.dtype.base = np.float32
    carray.atom.shape = [10]
    missing_values = [None, None]
    carray.attrs.missing_values = missing_values

    rcache = mocker.patch("landshark.hread.RowCache")
    feat = hread.Features(carray, 20, 10)

    rcache.assert_called_with(carray, 20, 10)
    assert feat.dtype == np.float32
    assert feat.nfeatures == 10
    assert feat.missing_values == missing_values


def test_target(mocker):
    """Test we keep coordinates aligned with labels."""
    mocker.patch('landshark.hread.tables.open_file')
    imcoords = mocker.patch("landshark.image.coords_training")
    batches_one = np.stack((np.arange(10), np.arange(10))).T[:, :, np.newaxis]
    imcoords.return_value = batches_one

    targs = hread.Targets("path", "label")
    targs._data = np.arange(10)

    batches = targs.training(mocker.MagicMock(), "batch_size")
    for x, y, t in batches:
        assert x == y == t

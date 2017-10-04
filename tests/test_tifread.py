"""Tests for the tif reading importer module."""

from collections import namedtuple

import numpy as np
import rasterio.transform
import pytest

from landshark.importers import tifread
def test_match():
    """
    Checks that _match can pull out a property from a bunch of image-like
    objects when that property is the same for each (default behaviour).
    """
    Im = namedtuple('Im', ['prop'])
    name = 'myprop'
    true_answer = 1
    images = [Im(prop=true_answer) for k in range(10)]
    prop = tifread._match(lambda x: x.prop, images, name)
    assert prop == true_answer


def test_match_nomatch(mocker):
    """
    Checks that _match correctly identifies a non-matching property and
    calls the right error functions.
    """
    Im = namedtuple('Im', ['prop'])
    name = 'myprop'
    images = [Im(prop=k) for k in range(10)]
    mocked_mismatch = mocker.patch('landshark.importers.tifread._fatal_mismatch')
    tifread._match(lambda x: x.prop, images, name)
    mocked_mismatch.assert_called_once_with(list(range(10)), images, name)


def test_fatal_mismatch(mocker):
    """Checks fatal mismatch calls log.fatal with some sensible text."""
    mock_error = mocker.patch('landshark.importers.tifread.log.error')
    property_list = list(range(3))
    Im = namedtuple('Im', ['name'])
    images = [Im(name="n{}".format(i)) for i in range(3)]
    name = "myname"
    with pytest.raises(Exception):
        tifread._fatal_mismatch(property_list, images, name)
    true_answer = 'No match for myname:\nn0: 0\nn1: 1\nn2: 2'
    mock_error.assert_called_once_with(true_answer)


def test_names():
    """Checks names are generated sanely for bands."""
    Im = namedtuple('Im', ['name', 'count'])
    im1 = Im(name="A", count=1)
    im2 = Im(name="B", count=2)
    bands = [tifread.Band(image=im1, index=1),
             tifread.Band(image=im2, index=1),
             tifread.Band(image=im2, index=2)]

    name = tifread._names(bands)
    true_answer = ["A", "B_1", "B_2"]
    assert name == true_answer


def test_missing():
    """Checks missing correctly converts types of nodatavals."""
    Im = namedtuple('Im', ['nodatavals', 'count'])
    im1 = Im(count=1, nodatavals=[1.0])
    im2 = Im(count=2, nodatavals=[2.0, 3.0])
    bands = [tifread.Band(image=im1, index=1),
             tifread.Band(image=im2, index=1),
             tifread.Band(image=im2, index=2)]

    res = tifread._missing(bands, np.int32)
    true_answer = [1, 2, 3]
    assert res == true_answer

    im3 = Im(count=2, nodatavals=[1.0, None])
    bands[0] = tifread.Band(image=im3, index=2)
    res2 = tifread._missing(bands, np.int32)
    true_answer2 = [None, 2, 3]
    assert res2 == true_answer2


def test_bands():
    """Checks that bands are correctly listed from images."""
    Im = namedtuple('Im', ['dtypes'])
    im1 = Im(dtypes=[np.float32, np.int32, np.uint8])
    im2 = Im(dtypes=[np.float64, np.uint64])

    true_cat = [tifread.Band(image=im1, index=2),
                tifread.Band(image=im1, index=3),
                tifread.Band(image=im2, index=2)]

    true_ord = [tifread.Band(image=im1, index=1),
                tifread.Band(image=im2, index=1)]

    true_res = tifread.BandCollection(categorical=true_cat, ordinal=true_ord)
    res = tifread._bands([im1, im2])
    assert res == true_res


def test_blockrows():
    """Checks blocksize does something sane."""
    Im = namedtuple('Im', ['block_shapes'])
    im1 = Im(block_shapes=[(1, 10), (2, 100)])
    im2 = Im(block_shapes=[(3, 30)])
    bands = [tifread.Band(image=im1, index=1),
             tifread.Band(image=im1, index=2),
             tifread.Band(image=im2, index=1)]

    blocksize = tifread._block_rows(bands)
    assert blocksize == 3


def test_windows():
    """Checks window list covers whole image."""
    w_list = tifread._windows(1024, 768, 10)
    assert np.all([k[1] == (0, 1024) for k in w_list])
    assert np.all([k[0][1] - k[0][0] == 10 for k in w_list[:-1]])
    assert w_list[-1][0][0] < 768
    assert w_list[-1][0][1] == 768

    w_list = tifread._windows(1024, 450, 5)
    assert np.all([k[1] == (0, 1024) for k in w_list])
    assert np.all([k[0][1] - k[0][0] == 5 for k in w_list])
    assert w_list[-1][0][0] == 445
    assert w_list[-1][0][1] == 450


def test_read(mocker):
    """Checks that read calls the right image functions in the right order."""

    a1 = [np.random.rand(10, 25) * 100,
          np.random.rand(10, 25) * 50,
          np.random.rand(10, 25) * 10]
    a2 = [np.random.rand(10, 25) * 100,
          np.random.rand(10, 25) * 50,
          np.random.rand(10, 25) * 10]
    answers = [np.concatenate((i1[..., np.newaxis],
                               i2[..., np.newaxis]), axis=-1).astype(np.int32)
               for i1, i2 in zip(a1, a2)]
    im = mocker.Mock()
    im.read = mocker.Mock(side_effect=a1)
    im2 = mocker.Mock()
    im2.read = mocker.Mock(side_effect=a2)
    bands = [tifread.Band(image=im, index=1), tifread.Band(image=im2, index=2)]
    windows = [((0, 10), (0, 25)), ((10, 20), (0, 25)), ((20, 30), (0, 25))]
    it = tifread._read(bands, windows, dtype=np.int32)
    for res, ans in zip(it, answers):
        assert np.all(res == ans)

    assert im.read.call_count == 3
    assert im2.read.call_count == 3
    for im_calls, im2_calls, w in zip(im.read.call_args_list,
                                      im2.read.call_args_list,
                                      windows):
        assert im_calls[0][0] == 1
        assert im2_calls[0][0] == 2
        assert im_calls[1] == {'window': w}
        assert im2_calls[1] == {'window': w}


@pytest.mark.parametrize("block_rows", [None, 3])
def test_imagestack(mocker, block_rows):
    """Constructs and image stack ensuring it calls all the right fns."""
    call = mocker.mock_module.call

    m_open = mocker.patch('landshark.importers.tifread.rasterio.open')
    m_open.return_value = [mocker.Mock(), mocker.Mock()]

    width = 10
    height = 20
    affine = rasterio.transform.IDENTITY
    m_match = mocker.patch('landshark.importers.tifread._match')
    m_match.side_effect = [width, height, affine]

    m_bands = mocker.patch('landshark.importers.tifread._bands')
    m_bands.return_value = tifread.BandCollection(ordinal=[mocker.Mock()],
                                                categorical=[mocker.Mock()])

    m_names = mocker.patch('landshark.importers.tifread._names')
    m_names.return_value = mocker.Mock()

    m_block_rows = mocker.patch('landshark.importers.tifread._block_rows')
    m_block_rows.return_value = 2
    m_missing = mocker.patch('landshark.importers.tifread._missing')
    m_missing.return_value = mocker.Mock()
    m_windows = mocker.patch('landshark.importers.tifread._windows')
    m_windows.return_value = mocker.Mock()
    m_pixels = mocker.patch('landshark.importers.tifread.pixel_coordinates')
    m_pixels.return_value = (np.zeros((10, 2)), np.zeros((10, 2)))
    paths = ['my/path', 'my/other/path']
    stack = tifread.ImageStack(paths, block_rows)
    m_open_calls = [call(paths[0], 'r'), call(paths[1], 'r')]
    m_open.assert_has_calls(m_open_calls, any_order=False)

    assert stack.width == width
    assert stack.height == height
    assert stack.affine == affine
    assert stack.ordinal_bands == m_bands.return_value.ordinal
    assert stack.categorical_bands == m_bands.return_value.categorical
    assert stack.ordinal_names == m_names.return_value
    assert stack.categorical_names == m_names.return_value
    assert stack.ordinal_dtype == np.float32
    assert stack.categorical_dtype == np.int32
    assert stack.windows == m_windows.return_value
    assert stack.block_rows == (block_rows if block_rows
                                else m_block_rows.return_value)
    m_missing_calls = [call(m_bands.return_value.ordinal,
                            dtype=stack.ordinal_dtype),
                       call(m_bands.return_value.categorical,
                            dtype=stack.categorical_dtype)]
    m_missing.assert_has_calls(m_missing_calls, any_order=True)

    m_read = mocker.patch('landshark.importers.tifread._read')
    stack.categorical_blocks()
    m_read.assert_called_with(stack.categorical_bands,
                              stack.windows,
                              stack.categorical_dtype)

    stack.ordinal_blocks()
    m_read.assert_called_with(stack.ordinal_bands,
                              stack.windows,
                              stack.ordinal_dtype)


class FakeImage:
    def __init__(self, name, width, height, affine, dtypes, block_rows):
        self.name = name
        self.width = width
        self.affine = affine
        self.height = height
        self.dtypes = dtypes
        self.count = len(self.dtypes)
        self.nodatavals = [-1.0 for i in range(self.count)]
        self.block_shapes = [(block_rows, width) for w in range(self.count)]


def test_imagestack_real(mocker):
    affine = rasterio.transform.IDENTITY
    im1 = FakeImage(name='im1', width=10, height=5, affine=affine,
                    dtypes=[np.dtype('float32'), np.dtype('int32')],
                    block_rows=2)
    im2 = FakeImage(name='im2', width=10, height=5, affine=affine,
                    dtypes=[np.dtype('uint8'), np.dtype('float64')],
                    block_rows=3)

    m_open = mocker.patch('landshark.importers.tifread.rasterio.open')
    m_open.side_effect = [im1, im2]
    paths = ['path1', 'path2']
    stack = tifread.ImageStack(paths)

    cat_bands = [tifread.Band(image=im1, index=2),
                 tifread.Band(image=im2, index=1)]
    ord_bands = [tifread.Band(image=im1, index=1),
                 tifread.Band(image=im2, index=2)]

    assert stack.affine == affine
    assert stack.width == 10
    assert stack.height == 5
    assert stack.block_rows == 3
    assert stack.categorical_bands == cat_bands
    assert stack.ordinal_bands == ord_bands
    assert stack.categorical_dtype == np.int32
    assert stack.ordinal_dtype == np.float32
    assert stack.categorical_missing == [-1, -1]
    assert stack.ordinal_missing == [-1., -1.]
    assert stack.categorical_names == ['im1_2', 'im2_1']
    assert stack.ordinal_names == ['im1_1', 'im2_2']
    assert stack.windows == [((0, 3), (0, 10)), ((3, 5), (0, 10))]
    assert np.all(stack.coordinates_x == np.arange(10 + 1, dtype=float))
    assert np.all(stack.coordinates_y == np.arange(5 + 1, dtype=float))


def test_block_shape():
    """Checks the (simple) multiplication for total size."""
    width = 4
    height = 5
    nbands = 3
    w = ((1, 1 + height), (3, 3 + width))
    r = tifread._block_shape(w, nbands)
    assert r == (height, width, nbands)

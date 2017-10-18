"""Tests for the feature writing importer module."""

import numpy as np
import tables
import pytest

from landshark.importers.tifread import _block_shape
from landshark.importers import featurewrite

@pytest.mark.parametrize("standardise", [True, False])
def test_write_datafile(mocker, standardise):
    """Checks that write_datafile calls all the right functions."""
    call = mocker.mock_module.call
    m_open = mocker.patch('landshark.importers.featurewrite.tables.open_file')
    m_write = mocker.patch('landshark.importers.featurewrite._write')
    m_std_write = mocker.patch(
        'landshark.importers.featurewrite._standardise_write')
    m_get_stats = mocker.patch(
        'landshark.importers.featurewrite._get_stats')
    m_get_stats.return_value = (mocker.Mock(), mocker.Mock())
    m_hfile = mocker.MagicMock()
    m_open.return_value = m_hfile
    m_size = mocker.patch('landshark.importers.featurewrite.os.path.getsize')
    m_size.return_value = 1000
    width = 10
    height = 5
    im_shape = (height, width)
    image_stack = mocker.Mock()
    image_stack.coordinates_x = np.arange(width + 1, dtype=np.float64)
    image_stack.coordinates_y = np.arange(height + 1, dtype=np.float64)
    image_stack.categorical_bands = ['b1', 'b2']
    image_stack.ordinal_bands = ['b3', 'b4']
    image_stack.width = width
    image_stack.height = height
    image_stack.block_rows = 2
    image_stack.ordinal_names = ['name1', 'name2']
    image_stack.categorical_names = ['cat1', 'cat2']
    image_stack.ordinal_missing = [None, 0.0]
    image_stack.categorical_missing = [None, 0]
    image_stack.windows = [((0, 2), (0, 10)), ((2, 4), (0, 10)), ((4, 5), (0, 10))]
    block_it_ord = (np.zeros(_block_shape(w, 2), dtype=np.float32)
                    for i, w in enumerate(image_stack.windows))
    block_it_cat = (np.zeros(_block_shape(w, 2), dtype=np.int32)
                    for i, w in enumerate(image_stack.windows))

    image_stack.categorical_blocks = mocker.Mock(return_value=block_it_ord)
    image_stack.ordinal_blocks = mocker.Mock(return_value=block_it_cat)

    m_cat_array = mocker.MagicMock()
    m_ord_array = mocker.MagicMock()
    m_hfile.create_carray.side_effect = [m_cat_array, m_ord_array]

    filename = 'filename'
    featurewrite.write_datafile(image_stack, filename, standardise)

    m_open.assert_called_once_with(filename, mode='w',
                                   title='Landshark Image Stack')

    assert m_hfile.root._v_attrs.height == height
    assert m_hfile.root._v_attrs.width == width

    coord_calls = [call(m_hfile.root, name='x_coordinates',
                        obj=image_stack.coordinates_x),
                   call(m_hfile.root, name='y_coordinates',
                        obj=image_stack.coordinates_y)]
    m_hfile.create_array.assert_has_calls(coord_calls, any_order=True)

    cat_atom = tables.Int32Atom(shape=(2,))
    ord_atom = tables.Float32Atom(shape=(2,))
    filters = tables.Filters(complevel=1, complib='blosc:lz4')
    carray_calls =[call(m_hfile.root, name='categorical_data', shape=im_shape,
                        filters=filters, atom=cat_atom),
                   call(m_hfile.root, name='ordinal_data', shape=im_shape,
                        filters=filters, atom=ord_atom)]

    m_hfile.create_carray.assert_has_calls(carray_calls)

    assert m_cat_array.attrs.labels == image_stack.categorical_names
    assert m_cat_array.attrs.missing_values == image_stack.categorical_missing
    assert m_ord_array.attrs.labels == image_stack.ordinal_names
    assert m_ord_array.attrs.missing_values == image_stack.ordinal_missing

    assert m_cat_array.attrs.mean is None
    assert m_cat_array.attrs.variance is None

    if standardise:
        assert m_ord_array.attrs.mean == m_get_stats.return_value[0]
        assert m_ord_array.attrs.variance == m_get_stats.return_value[1]
        m_get_stats.assert_called_with(m_ord_array, image_stack.ordinal_blocks,
                                       image_stack.ordinal_missing)
        m_write.assert_has_calls([call(m_cat_array,
                                       image_stack.categorical_blocks)])
        m_std_write.assert_called_with(m_ord_array,
                                       image_stack.ordinal_blocks,
                                       image_stack.ordinal_missing,
                                       m_ord_array.attrs.mean,
                                       m_ord_array.attrs.variance)
    else:
        assert m_ord_array.attrs.mean is None
        assert m_ord_array.attrs.variance is None
        m_write.assert_has_calls([call(m_cat_array,
                                       image_stack.categorical_blocks),
                                  call(m_ord_array,
                                       image_stack.ordinal_blocks)])
    m_hfile.close.assert_called_once()


def test_statistics():
    n_features = 2
    n_rows = 10
    n_blocks = 5
    stats = featurewrite._Statistics(n_features)
    data = [np.random.randn(n_rows, n_features) for i in range(n_blocks)]
    all_data = np.concatenate(data, axis=0)
    for d in data:
        stats.update(d)
    mean = stats.mean
    var = stats.variance
    true_mean = np.mean(all_data, axis=0)
    true_var = np.var(all_data, axis=0)
    assert np.allclose(mean, true_mean)
    assert np.allclose(var, true_var)


def test_statistics_masked():
    n_features = 2
    n_rows = 10
    n_blocks = 5
    stats = featurewrite._Statistics(n_features)
    data = [np.random.randn(n_rows, n_features) for i in range(n_blocks)]
    masks = [np.random.choice(2, size=(n_rows, n_features)).astype(bool)
             for i in range(n_blocks)]
    all_data = np.concatenate(data, axis=0)
    all_masks = np.concatenate(masks, axis=0)
    all_marray = np.ma.MaskedArray(data=all_data, mask=all_masks)
    m_data = [np.ma.MaskedArray(data=d, mask=m) for d, m in zip(data, masks)]
    for d in m_data:
        stats.update(d)
    mean = stats.mean
    var = stats.variance
    true_mean = np.ma.mean(all_marray, axis=0)
    true_var = np.ma.var(all_marray, axis=0)
    assert np.allclose(mean, true_mean)
    assert np.allclose(var, true_var)


def test_to_masked():
    cols = 4
    rows = 10
    data = np.arange(cols * rows).reshape(rows, cols)
    missing_values = [None, 5, 0, 11]
    result = featurewrite._to_masked(data, missing_values)
    answer = np.zeros_like(data, dtype=bool)
    answer[1, 1] = True
    answer[2, 3] = True
    assert np.all(answer == result.mask)


def test_write():
    rows = 10
    cols = 2
    data = np.arange(rows * cols).reshape((rows, cols))
    blocks = [data[0:3], data[3:6], data[6:9], data[9:10]]

    array = np.zeros_like(data)

    def block_f():
        return blocks

    featurewrite._write(array, block_f)
    assert np.all(array == data)


def setup_write(parent_mock):
    m_masked = parent_mock.patch("landshark.importers.featurewrite._to_masked")
    m_masked.side_effect = [0, 1, 2, 3]
    m_missing_values = parent_mock.Mock()
    m_missing_values.side_effect = [4, 5 ,6 ,7]
    rows = 10
    cols = 2
    data = np.arange(rows * cols).reshape((rows, cols))
    blocks = [data[0:3], data[3:6], data[6:9], data[9:10]]
    array = parent_mock.MagicMock()
    array.atom.shape = (data.shape[1],)

    def block_f():
        return blocks
    return array, block_f, m_missing_values, m_masked


def test_get_stats(mocker):
    call = mocker.mock_module.call
    m_stats = mocker.patch("landshark.importers.featurewrite._Statistics",
                           autospec=True)
    m_stats_obj = mocker.Mock()
    m_stats_obj.mean = 10
    m_stats_obj.variance = 2
    m_stats.return_value = m_stats_obj
    array, block_f, m_missing_values, m_masked = setup_write(mocker)
    mean, variance = featurewrite._get_stats(array, block_f, m_missing_values)

    blocks = block_f()
    for call, b in zip(m_masked.call_args_list, blocks):
        assert np.all(call[0][0] == b)
        assert call[0][1] == m_missing_values

    m_masked_calls = [call(k, m_missing_values) for k in blocks]
    # m_masked.assert_has_calls(m_masked_calls)
    m_stats.assert_called_once_with(array.atom.shape[0])
    # from the side effect of m_masked
    m_stats_obj.update.assert_has_calls([call(0), call(1), call(2), call(3)])

    assert mean == m_stats_obj.mean
    assert variance == m_stats_obj.variance


def test_standardise_write(mocker):
    m_masked = mocker.patch("landshark.importers.featurewrite._to_masked")
    m_missing_data = mocker.Mock()
    mean = 10.0
    variance = 2.0

    def fake_masked(b, missing_vals):
        return np.ma.MaskedArray(data=np.copy(b), mask=False)

    m_masked.side_effect = fake_masked

    rows = 10
    cols = 2
    data = np.arange(rows * cols).reshape((rows, cols)).astype(float)
    blocks = [data[0:3], data[3:6], data[6:9], data[9:10]]
    array = np.zeros_like(data)

    def block_f():
        return blocks

    featurewrite._standardise_write(array, block_f,
                                    m_missing_data, mean, variance)
    assert np.allclose(array, (data - mean) / np.sqrt(variance))

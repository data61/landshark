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
    assert m_ord_array.attrs.mean is None
    assert m_ord_array.attrs.variance is None

    if standardise:
        m_write.assert_has_calls([call(m_cat_array,
                                       image_stack.categorical_blocks)])
        m_std_write.assert_called_with(m_ord_array,
                                       image_stack.ordinal_blocks,
                                       image_stack.ordinal_missing)
    else:
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
    var = stats.var
    true_mean = np.mean(all_data, axis=0)
    true_var = np.var(all_data, axis=0)
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

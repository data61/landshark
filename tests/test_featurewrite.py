"""Tests for the feature writing importer module."""

import numpy as np
import tables

from landshark.importers.tifread import _block_shape
from landshark.importers import featurewrite

def test_write_datafile(mocker):
    """Checks that write_datafile calls all the right functions."""
    call = mocker.mock_module.call
    m_open = mocker.patch('landshark.importers.featurewrite.tables.open_file')
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
    featurewrite.write_datafile(image_stack, filename)

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

    cat_slices = [k[0][0] for k in m_cat_array.__setitem__.call_args_list]
    ord_slices = [k[0][0] for k in m_ord_array.__setitem__.call_args_list]

    assert cat_slices[0].start == 0
    assert cat_slices[0].stop == 2
    assert cat_slices[-1].start == 4
    assert cat_slices[-1].stop == 5

    assert ord_slices[0].start == 0
    assert ord_slices[0].stop == 2
    assert ord_slices[-1].start == 4
    assert ord_slices[-1].stop == 5

    m_hfile.close.assert_called_once()

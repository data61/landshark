"""Input/output routines for geo data types."""

import itertools
import logging
import datetime

import numpy as np
import shapefile
# for mypy type checking
from typing import List, Tuple, Iterator, Union

log = logging.getLogger(__name__)

# Typechecking aliases
ShpFieldsType = List[Tuple[str, str, int, int]]


def _shapefile_float_fields(fields: ShpFieldsType) \
        -> Iterator[Tuple[int, str]]:
    """Pull out the float fields from the shapefile field specification.

    Parameters
    ----------
    fields : ShpFieldsType
        The weird list-of-lists that shapefile uses to describe field types

    Returns
    -------
    result : Iterator[Tuple[int, str]]
        An iterator over (<index_number>, <name>) pairs of the float columns.

    """
    shapefields = [f[0] for f in fields[1:]]  # Skip DeletionFlag
    dtype_flags = [(f[1], f[2]) for f in fields[1:]]  # Skip DeletionFlag
    # http://www.dbase.com/Knowledgebase/INT/db7_file_fmt.htm
    # We're only going to support float types for now
    field_indices = [i for i, k in enumerate(dtype_flags) if k[0] == "N"]
    field_names = [shapefields[i] for i in field_indices]
    result = zip(field_indices, field_names)
    return result


def _extract_type(python_type, field_length):
    if python_type is float:
        return "f8"
    elif python_type is int:
        return "i8"
    elif python_type is str:
        return "a" + str(field_length)
    elif python_type is datetime.date:
        return "a10"


def _get_record_info(shp):
    field_list = shp.fields[1:]
    labels, type_strings, nbytes, decimals = zip(*field_list)
    record0 = shp.record(0)
    record0[3] = str(record0[3])
    types_from_data = [type(k) for k in record0]
    type_list = [_extract_type(t, l) for t, l in zip(types_from_data, nbytes)]
    return labels, type_list


def _get_dtype(labels, all_labels, all_dtypes):
    dtype_dict = dict(zip(all_labels, all_dtypes))
    dtype_set = {dtype_dict[l] for l in labels}
    if len(dtype_set) > 1:
        raise ValueError("Requested target labels have different types")
    dtype = dtype_set.pop()
    return dtype


def _to_array(record, indices, dtype):
    x_i = np.array([record[i] for i in indices], dtype=dtype)
    return x_i


def _get_indices(labels, all_labels):
    label_dict = dict(zip(all_labels, range(len(all_labels))))
    label_indices = [label_dict[k] for k in labels]
    return label_indices


class ShapefileTargets:
    """
    Targets for spatial inference backed by a shapefile.

    This class reads a shapefile with point information and provides
    iterators to the coordinates and the (float) records.

    Parameters
    ----------
    filename : str
        The shapefile (.shp)
    """
    def __init__(self, filename: str, labels: List[str],
                 batchsize: int=100) -> None:
        """Construct an instance of ShapefileTargets."""
        self._sf = shapefile.Reader(filename)
        self.all_fields, self.all_dtypes = _get_record_info(self._sf)
        self.labels = labels
        self._label_indices = _get_indices(self.labels, self.all_fields)
        self.dtype = _get_dtype(self.labels, self.all_fields, self.all_dtypes)
        self._ntotal = self._sf.numRecords
        self.n = self._ntotal
        self.batchsize = batchsize

    def _data(self) -> Iterator[np.ndarray]:
        """Create an iterator for the shapefile data.

        This will return a batch of coordinates,  in order
        from the shapefile.

        Yields
        ------

        """
        for sr in self._sf.iterShapeRecords():
            coords = _to_coords(sr.shape)
            array = _to_array(sr.record, self._label_indices, self.dtype)
            yield coords, array

    def batches(self) -> Iterator[np.ndarray]:
        batch_it = itertools.islice(self._data(), self.batchsize)
        for b in batch_it:
            import IPython; IPython.embed(); import sys; sys.exit()


        # batch = []
        # for sr in self._sf.iterShapeRecords():
        #     coords, record = _to_coords(sr.shape), sr.record
        #     array = _to_array(record, self._label_indices, self.dtype)
        #     batch.append((coords, array))
        #     if len(batch) == self.batchsize:
        #         full_result = _cast_batch(batch, self.dtypes, self.fields)
        #         result = _trim_to_labels(full_result, labels)

        #         yield result
        #         batch = []
        # if len(batch) > 0:
        #     full_result = _cast_batch(batch, self.dtypes, self.fields)
        #     result = _trim_to_labels(full_result, labels)
        #     yield result

def _to_coords(shape):
    result = shape.__geo_interface__["coordinates"]
    return result


def _trim_to_labels(batch, labels):
    (coords_x, coords_y), record_dict = batch
    records = [record_dict[k] for k in labels]
    dtype_set = {k.dtype for k in records}
    dtype = dtype_set.pop()
    import IPython; IPython.embed(); import sys; sys.exit()

def _cast_batch(batch_list, dtypes, labels):
    coord_list, record_list = zip(*batch_list)
    coords_x, coords_y = np.array(coord_list, dtype=float).T
    record_cols = zip(*record_list)
    record_arrays = [np.array(r, dtype=d) for r, d in zip(record_cols, dtypes)]
    record_dict = dict(zip(labels, record_arrays))
    return (coords_x, coords_y), record_dict

def _make_mask(npoints, indices):
    if indices is not None:
        mask = np.zeros(npoints, dtype=bool)
        mask[indices] = True
    else:
        mask = np.ones(npoints, dtype=bool)
    return mask


def train_test_targets(
        filename: str,
        target_labels: List[str],
        test_proportion: float,
        random_state: Union[int, None]=None
        ) -> Tuple[ShapefileTargets, ShapefileTargets]:
    """Create training and testing data shapefile readers.

    Parameters
    ----------
    filename : str
        The shapefile (.shp)
    test_proportion : float
        Fraction of data to make testing
    random_state : int, None
        Random state to use for splitting the data, None for no seed.

    Returns
    -------
    tr_read : ShapefileTargets
        the training data shape file reader
    ts_read : ShapefileTargets
        the testing data shape file reader

    """
    assert test_proportion > 0. and test_proportion < 1.0

    # Get length of data
    n = shapefile.Reader(filename).numRecords

    # Randomly choose training data
    rnd = np.random.RandomState(random_state)
    ts_ind = np.sort(rnd.choice(n, round(test_proportion * n), replace=False))

    # Create training indices
    tr_mask = np.ones(n, dtype=bool)
    tr_mask[ts_ind] = False
    tr_ind = np.where(tr_mask)[0]

    # Make readers
    ts_read = ShapefileTargets(filename, target_labels)

    return tr_read, ts_read

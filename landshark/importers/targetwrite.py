"""Module for writing targets to HDF5."""

import os.path
import logging

import tables

from landshark.importers.shpread import ShapefileTargets

log = logging.getLogger(__name__)


def write_targetfile(sf: ShapefileTargets, filename: str) -> None:
    """
    Write out a representation of target data to an HDF5 from a shapefile.

    Parameters
    ----------
    sf : ShapefileTargets
        The shapefile object to output.
    filename : str
        The output filename of the HDF5 file.

    """
    title = "Landshark Targets"
    log.info("Creating HDF5 target file")
    h5file = tables.open_file(filename, mode="w", title=title)

    n = sf.n
    # ncols_ord = len(sf.fields)
    ord_atom = tables.Float32Atom()
    filters = tables.Filters(complevel=1, complib="blosc:lz4")

    # log.info("Creating data arrays")
    # target_array = h5file.create_carray(h5file.root, name="targets",
    #                                     atom=ord_atom, shape=(n, ncols_ord),
    #                                     filters=filters)
    # target_array.attrs.labels = sf.fields

    coord_array = h5file.create_carray(h5file.root, name="coordinates",
                                       atom=ord_atom, shape=(n, 2),
                                       filters=filters)
    coord_array.attrs.labels = ["x", "y"]

    # log.info("Writing target data")
    # for i, r in enumerate(sf.ordinal_data()):
    #     target_array[i] = r

    log.info("Writing coordinate data")
    for i, c in enumerate(sf.coordinates()):
        coord_array[i] = c

    log.info("Closing file")
    h5file.close()
    file_size = os.path.getsize(filename) // (1024 ** 2)
    log.info("Written {}MB file to disk.".format(file_size))

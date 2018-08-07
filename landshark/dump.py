"""Dump training/query data to HDF5."""

import numpy as np
import tables

from landshark.basetypes import IdReader
from landshark.dataprocess import (QueryDataProcessor, SourceMetadata,
                                   TrainingDataProcessor)
from landshark.featurewrite import write_imagespec
from landshark.hread import H5Features
from landshark.image import indices_strip
from landshark.iteration import batch_slices, with_slices
from landshark.metadata import (CategoricalMetadata, QueryMetadata,
                                TrainingMetadata)
from landshark.multiproc import task_list


def dump_training(tinfo: SourceMetadata, metadata: TrainingMetadata,
                  fname: str, batchsize: int, nworkers: int) -> None:

    n_rows = metadata.targets.N
    has_ord = metadata.features.D_ordinal > 0
    has_cat = metadata.features.D_categorical > 0
    cat_targets = isinstance(metadata.targets, CategoricalMetadata)
    patchwidth = metadata.halfwidth * 2 + 1

    worker = TrainingDataProcessor(tinfo)
    tasks = list(batch_slices(batchsize, n_rows))
    out_it = task_list(tasks, tinfo.target_src, worker, nworkers)
    fold_it = tinfo.folds.iterator(batchsize)

    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    target_atom = tables.Int32Atom() if cat_targets else tables.Float32Atom()

    with tables.open_file(fname, mode="w", title=tinfo.name) as outfile:

        if has_ord:
            ord_shape = (n_rows, metadata.features.ordinal.D,
                         patchwidth, patchwidth)
            ord_array = outfile.create_carray(outfile.root, name="ordinal",
                                              atom=tables.Float32Atom(),
                                              shape=ord_shape, filters=filters)
            ord_array.attrs.missing = metadata.features.ordinal.missing
            ord_array.attrs.labels = metadata.features.ordinal.labels
            ord_array.attrs.means = metadata.features.ordinal.means
            ord_array.attrs.variances = metadata.features.ordinal.variances
        if has_cat:
            cat_shape = (n_rows, metadata.features.categorical.D,
                         patchwidth, patchwidth)
            cat_array = outfile.create_carray(outfile.root, name="categorical",
                                              atom=tables.Int32Atom(),
                                              shape=cat_shape, filters=filters)
            cat_array.attrs.missing = metadata.features.categorical.missing
            cat_array.attrs.labels = metadata.features.categorical.labels
            cat_array.attrs.mappings = metadata.features.categorical.mappings
            cat_array.attrs.counts = metadata.features.categorical.counts

        target_shape = (n_rows, metadata.targets.D)
        target_array = outfile.create_carray(outfile.root, name="targets",
                                             atom=target_atom,
                                             shape=target_shape,
                                             filters=filters)
        target_array.attrs.label = metadata.targets.labels
        if hasattr(metadata.targets, "mappings"):
            target_array.attrs.mappings = metadata.targets.mappings

        folds_array = outfile.create_carray(outfile.root, name="folds",
                                            atom=tables.Int32Atom(),
                                            shape=(n_rows,), filters=filters)
        target_array.attrs.labels = metadata.targets.labels
        target_array.attrs.missing = metadata.targets.missing
        if isinstance(metadata.targets, CategoricalMetadata):
            target_array.attrs.mappings = metadata.targets.mappings
            target_array.attrs.counts = metadata.targets.counts
        else:
            target_array.attrs.means = metadata.targets.means
            target_array.attrs.variances = metadata.targets.variances

        start = 0
        for o, c, t in out_it:
            stop = start + t.shape[0]
            if has_ord:
                if ord_array.attrs.missing is not None:
                    o.data[o.mask] = ord_array.attrs.missing
                ord_array[start:stop] = o.data
            if has_cat:
                if cat_array.attrs.missing is not None:
                    c.data[c.mask] = cat_array.attrs.missing
                cat_array[start:stop] = c.data
            target_array[start:stop] = t
            start = stop

        if has_ord:
            ord_array.flush()
        if has_cat:
            cat_array.flush()
        target_array.flush()

        for s, f in with_slices(fold_it):
            folds_array[s.start:s.stop] = f
        folds_array.flush()


def dump_query(feature_path: str, metadata: QueryMetadata, strip: int,
               totalstrips: int, batchsize: int, halfwidth: int,
               nworkers: int, name: str, fname: str) -> None:

    image_spec = metadata.features.image
    true_batchsize = batchsize * metadata.features.image.width
    reader_src = IdReader()
    it, n_total = indices_strip(image_spec, strip, totalstrips,
                                true_batchsize)

    # read stuff from features because we dont have a metadata object
    feature_source = H5Features(feature_path)
    has_ord = False
    has_cat = False
    nfeatures_ord, nfeatures_cat = 0, 0
    if feature_source.ordinal is not None:
        has_ord = True
        nfeatures_ord = feature_source.ordinal.atom.shape[0]
        missing_ord = feature_source.ordinal.attrs.missing
    if feature_source.categorical is not None:
        has_cat = True
        nfeatures_cat = feature_source.categorical.atom.shape[0]
        missing_cat = feature_source.categorical.attrs.missing
    del feature_source

    active_ord = np.ones(nfeatures_ord, dtype=bool)
    active_cat = np.ones(nfeatures_cat, dtype=bool)
    patchwidth = halfwidth * 2 + 1
    worker = QueryDataProcessor(image_spec, feature_path, halfwidth,
                                active_ord, active_cat)
    tasks = list(it)
    out_it = task_list(tasks, reader_src, worker, nworkers)

    filters = tables.Filters(complevel=1, complib="blosc:lz4")

    with tables.open_file(fname, mode="w", title=name) as outfile:
        if has_ord:
            ord_shape = (n_total, nfeatures_ord,
                         patchwidth, patchwidth)
            ord_array = outfile.create_carray(outfile.root, name="ordinal",
                                              atom=tables.Float32Atom(),
                                              shape=ord_shape, filters=filters)
            ord_array.attrs.missing = missing_ord
            ord_array.attrs.labels = metadata.features.ordinal.labels
        if has_cat:
            cat_shape = (n_total, nfeatures_cat,
                         patchwidth, patchwidth)
            cat_array = outfile.create_carray(outfile.root, name="categorical",
                                              atom=tables.Int32Atom(),
                                              shape=cat_shape, filters=filters)
            cat_array.attrs.missing = missing_cat
            cat_array.attrs.labels = metadata.features.categorical.labels
            cat_array.attrs.mappings = metadata.features.categorical.mappings

        start = 0
        for o, c in out_it:
            n = o.shape[0] if o is not None else c.shape[0]
            stop = start + n
            if has_ord:
                if ord_array.attrs.missing is not None:
                    o.data[o.mask] = ord_array.attrs.missing
                ord_array[start:stop] = o.data
            if has_cat:
                if cat_array.attrs.missing is not None:
                    c.data[c.mask] = cat_array.attrs.missing
                cat_array[start:stop] = c.data
            start = stop
        if has_ord:
            ord_array.flush()
        if has_cat:
            cat_array.flush()

        write_imagespec(image_spec, outfile)

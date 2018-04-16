import tables


from landshark.trainingdata import TrainingDataProcessor
from landshark.iteration import batch_slices, with_slices
from landshark.multiproc import task_list


def to_hdf5(tinfo, metadata, fname, batchsize, nworkers):

    # TODO what are missing values!!!!

    n_rows = len(tinfo.target_src)
    has_ord = metadata.nfeatures_ord is not None
    has_cat = metadata.nfeatures_cat is not None
    cat_targets = metadata.target_counts is not None
    patchwidth = metadata.halfwidth * 2 + 1

    worker = TrainingDataProcessor(tinfo.image_spec, tinfo.feature_path,
                                   tinfo.halfwidth)
    tasks = list(batch_slices(batchsize, n_rows))
    out_it = task_list(tasks, tinfo.target_src, worker, nworkers)
    fold_it = tinfo.folds.iterator(batchsize)

    filters = tables.Filters(complevel=1, complib="blosc:lz4")
    target_atom = tables.Int32Atom() if cat_targets else tables.Float32Atom()

    with tables.open_file(fname, mode="w", title=tinfo.name) as outfile:
        if has_ord:
            ord_shape = (n_rows, metadata.nfeatures_ord,
                         patchwidth, patchwidth)
            ord_array = outfile.create_carray(outfile.root, name="ordinal",
                                              atom=tables.Float32Atom(),
                                              shape=ord_shape, filters=filters)
            ord_array.attrs.missing = metadata.missing_ord
        if has_cat:
            cat_shape = (n_rows, metadata.nfeatures_cat,
                         patchwidth, patchwidth)
            cat_array = outfile.create_carray(outfile.root, name="categorical",
                                              atom=tables.Int32Atom(),
                                              shape=cat_shape, filters=filters)
            cat_array.attrs.missing = metadata.missing_cat

        target_shape = (n_rows, metadata.ntargets)
        target_array = outfile.create_carray(outfile.root, name="targets",
                                             atom=target_atom,
                                             shape=target_shape,
                                             filters=filters)
        folds_array = outfile.create_carray(outfile.root, name="folds",
                                            atom=tables.Int32Atom(),
                                            shape=(n_rows,), filters=filters)

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



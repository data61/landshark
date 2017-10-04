import numpy as np

from landshark import patch


def _read(data, patch_reads, mask_reads, n, patchwidth):
    patch_data = np.empty((n, patchwidth, patchwidth, data.nfeatures),
                          dtype=data.dtype)
    patch_mask = np.zeros_like(patch_data, dtype=bool)

    for r in patch_reads:
        patch_data[r.idx, r.yp, r.xp] = data(r.y, r.x)

    for r in mask_reads:
        patch_mask[r.idx, r.yp, r.xp] = True

    for i, v in enumerate(data.missing_values):
        if v is not None:
            patch_mask[..., i] |= (patch_data[..., i] == v)

    marray = np.ma.MaskedArray(data=patch_data, mask=patch_mask)
    return marray


def read_batch(indices_x, indices_y, features, halfwidth):

    patch_reads, mask_reads = patch.patches(indices_x, indices_y,
                                            halfwidth,
                                            features.image_spec.width,
                                            features.image_spec.height)
    n = indices_x.shape[0]
    patchwidth = 2 * halfwidth + 1

    ord_marray = _read(features.ord, patch_reads, mask_reads, n, patchwidth)
    cat_marray = _read(features.cat, patch_reads, mask_reads, n, patchwidth)
    return ord_marray, cat_marray

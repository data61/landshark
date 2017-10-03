import numpy as np
from . import image

def _extract_patches(data, mask, cache, patch_reads, mask_reads, missing_values):

    for r in patch_reads:
        data[r.idx, r.yp, r.xp] = cache(r.y, r.x)

    for r in mask_reads:
        mask[r.idx, r.yp, r.xp] = True

    for i, v in enumerate(missing_values):
        if v is not None:
            mask[..., i] |= (data[..., i] == v)


def read_batch(coords_x, coords_y, hfile, ord_cache, cat_cache):
    image_height = hfile.root._v_attrs.height
    image_width = hfile.root._v_attrs.width
    ord_data = hfile.root.ordinal_data
    cat_data = hfile.root.categorical_data
    n = coords_x.shape[0]
    n_feats_ord = ord_data.atom.shape[0]
    n_feats_cat = cat_data.atom.shape[0]

    ord_patch_data = np.empty((n, patchwidth, patchwidth, n_feats_ord),
                              dtype=np.float32)
    cat_patch_data = np.empty((n, patchwidth, patchwidth, n_feats_cat),
                              dtype=np.int32)
    ord_patch_mask = np.zeros_like(ord_patch_data, dtype=bool)
    cat_patch_mask = np.zeros_like(cat_patch_data, dtype=bool)

    patch_reads = list(ls.patch.patches(coords_x, coords_y, halfwidth, image_width,
                                   image_height))
    mask_reads = list(ls.patch.mask_patches(coords_x, coords_y, halfwidth,
                                       image_width, image_height))

    ord_missing = xfile.root.ordinal_data.attrs.missing_values
    _extract_patches(ord_patch_data, ord_patch_mask, ord_cache, patch_reads,
                   mask_reads, ord_missing)

    cat_missing = xfile.root.categorical_data.attrs.missing_values
    _extract_patches(cat_patch_data, cat_patch_mask, cat_cache, patch_reads,
                   mask_reads, cat_missing)

    ord_marray = np.ma.MaskedArray(data=ord_patch_data, mask=ord_patch_mask)
    cat_marray = np.ma.MaskedArray(data=cat_patch_data, mask=cat_patch_mask)

    return ord_marray, cat_marray


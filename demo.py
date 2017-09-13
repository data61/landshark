import numpy as np
import landshark as ls
#  TODO do this in the __init__
import landshark.image
import landshark.patch
import tables

halfwidth = 2
patch_pixels = ((2 * halfwidth) + 1) ** 2
target_label = 'Fe_ppm_imp'

xfile = tables.open_file("lbalpha.hdf5")
yfile = tables.open_file("geochem_sites.hdf5")

#  TODO iterate these properly in batches
#  TODO save these in the opposite order so dont transpose
#  TODO sort the coordinates in the Y
coords_x, coords_y = yfile.root.coordinates.read().transpose()
targets = yfile.root.targets.read()
labels = yfile.root.targets.attrs.labels
Y = targets[:, labels.index(target_label)]
n = targets.shape[0]

x_pixel_array = xfile.root.x_coordinates.read()
y_pixel_array = xfile.root.y_coordinates.read()
bounds = ls.image.bounds(x_pixel_array, y_pixel_array)
image_height = xfile.root._v_attrs.height
image_width = xfile.root._v_attrs.width

data_in_bounds = ls.image.in_bounds(coords_x, coords_y, bounds)
assert np.all(data_in_bounds)

coords_x_image = ls.image.world_to_image(coords_x, x_pixel_array)
coords_y_image = ls.image.world_to_image(coords_y, y_pixel_array)

ord_data = xfile.root.ordinal_data
cat_data = xfile.root.categorical_data


# TODO could make these at batch time if there are billions of points
patches = [ls.patch.Patch(x, y, halfwidth, image_width, image_height)
           for x, y in zip(coords_x_image, coords_y_image)]


ord_patch_data = np.empty((n, patch_pixels, ord_data.shape[1]),
                          dtype=np.float32)
cat_patch_data = np.empty((n, patch_pixels, cat_data.shape[1]),
                          dtype=np.int32)

for i, p in enumerate(patches):
    #  iterating over contiguous reads for a patch
    for rp, r in zip(p.patch_flat, p.flat):
        ord_patch_data[i, rp] = ord_data[r]
        cat_patch_data[i, rp] = cat_data[r]

# TODO missing data harder if everything flat
cat_missing = cat_data.attrs.missing_values
ord_missing = ord_data.attrs.missing_values

ord_mask = np.zeros_like(ord_patch_data, dtype=bool)
cat_mask = np.zeros_like(cat_patch_data, dtype=bool)

for i, v in enumerate(cat_missing):
    if v is not None:
        cat_mask[:, :, i] = cat_patch_data[:, :, i] == v

for i, v in enumerate(ord_missing):
    if v is not None:
        ord_mask[:, :, i] = ord_patch_data[:, :, i] == v

ord_marray = np.ma.MaskedArray(data=ord_patch_data, mask=ord_mask)
cat_marray = np.ma.MaskedArray(data=cat_patch_data, mask=cat_mask)

# ...

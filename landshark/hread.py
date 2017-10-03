from image import coords_query, coords_training
from feed import read_batch


def read_targets(xfile, yfile, target_label, batchsize):
    x_pixel_array = xfile.root.x_coordinates.read()
    y_pixel_array = xfile.root.y_coordinates.read()
    coords_it = coords_training(yfile.root.coordinates, x_pixel_array,
                                y_pixel_array, batchsize)
    labels = yfile.root.targets.attrs.labels
    targets = yfile.root.targets.read()
    Y = targets[:, labels.index(target_label)]
    return coords_it, Y


def read_features(xfile, ord_cache, cat_cache, batchsize, coords_it=None):
    image_height = xfile.root._v_attrs.height
    image_width = xfile.root._v_attrs.width
    if coords_it is None:
        coords_it = coords_query(image_width, image_height, batchsize)
    data_batches = (read_batch(cx, cy, xfile, ord_cache, cat_cache)
                    for cx, cy in coords_it)
    return data_batches

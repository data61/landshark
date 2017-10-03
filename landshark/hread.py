

def read_targets(xfile, yfile):
    x_pixel_array = xfile.root.x_coordinates.read()
    y_pixel_array = xfile.root.y_coordinates.read()
    coords_it = get_coords_training(yfile.root.coordinates, x_pixel_array,
                                    y_pixel_array)
    labels = yfile.root.targets.attrs.labels
    targets = yfile.root.targets.read()
    Y = targets[:, labels.index(target_label)]
    return coords_it, Y


def read_features(xfile, ord_cache, cat_cache, coords_it=None):
    image_height = xfile.root._v_attrs.height
    image_width = xfile.root._v_attrs.width
    if coords_it is None:
        coords_it = get_coords_query(image_width, image_height)
        # coords_it = get_coords_query(100, 100)
    data_batches = (read_batch(cx, cy, xfile, ord_cache, cat_cache)
                    for cx, cy in coords_it)
    return data_batches


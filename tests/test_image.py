"""Tests for the image module."""
from itertools import product

import numpy as np

from landshark import image

SEED = 666


def test_pixel_coordinates(random_image_transform):
    """Test that pixel coordinates are valid for random examples."""
    data = random_image_transform
    coords_x, coords_y = image.pixel_coordinates(data.width, data.height,
                                                 data.affine)
    # outer corners of last pixel
    pix_x = np.arange(data.width + 1, dtype=np.float64)
    pix_y = np.arange(data.height + 1, dtype=np.float64)    # ditto
    true_coords_x = (pix_x * data.pixel_width) + data.origin_x
    true_coords_y = (pix_y * (-1.0 * data.pixel_height)) + data.origin_y
    assert np.allclose(true_coords_x, coords_x)
    assert np.allclose(true_coords_y, coords_y)


def test_image_to_world(random_image_transform):
    """Test that pixel coordinates are valid for random examples."""
    data = random_image_transform
    coords_x, coords_y = image.pixel_coordinates(data.width, data.height,
                                                 data.affine)
    w = np.arange(data.width, dtype=int)
    h = np.arange(data.height, dtype=int)
    true_coords_x = (w * data.pixel_width) + data.origin_x
    true_coords_y = (h * (-1.0 * data.pixel_height)) + data.origin_y
    result_x = image.image_to_world(w, coords_x)
    result_y = image.image_to_world(h, coords_y)
    assert np.all(true_coords_x == result_x)
    assert np.all(true_coords_y == result_y)


def test_world_to_image_edges(random_image_transform):
    """Checks that pixel edges are correctly mapped to indices."""
    data = random_image_transform
    pixel_coords_x, pixel_coords_y = image.pixel_coordinates(data.width,
                                                             data.height,
                                                             data.affine)
    w = np.arange(data.width + 1, dtype=int)
    h = np.arange(data.height + 1, dtype=int)
    coords_x = (w * data.pixel_width) + data.origin_x
    coords_y = (h * (-1.0 * data.pixel_height)) + data.origin_y
    idx_x = image.world_to_image(coords_x, pixel_coords_x)
    idx_y = image.world_to_image(coords_y, pixel_coords_y)

    true_idx_x = np.array(list(range(data.width)) +
                          [data.width - 1], dtype=int)
    true_idx_y = np.array(list(range(data.height)) +
                          [data.height - 1], dtype=int)

    assert np.all(true_idx_x == idx_x)
    assert np.all(true_idx_y == idx_y)


def test_world_to_image_centers(random_image_transform):
    """Checks that pixel centres are correctly mapped to indices."""
    data = random_image_transform
    pixel_coords_x, pixel_coords_y = image.pixel_coordinates(data.width,
                                                             data.height,
                                                             data.affine)
    w = np.arange(data.width, dtype=int)
    h = np.arange(data.height, dtype=int)
    coords_x = ((w.astype(float) + 0.5) * data.pixel_width) + data.origin_x
    coords_y = ((h.astype(float) + 0.5) * (-1.0 * data.pixel_height)) \
        + data.origin_y
    idx_x = image.world_to_image(coords_x, pixel_coords_x)
    idx_y = image.world_to_image(coords_y, pixel_coords_y)

    true_idx_x = np.arange(data.width, dtype=int)
    true_idx_y = np.arange(data.height, dtype=int)

    assert np.all(true_idx_x == idx_x)
    assert np.all(true_idx_y == idx_y)


def test_bounding_box():

    x_coords = np.arange(10)
    y_coords = np.arange(5)

    b = image.bounds(x_coords, y_coords)
    assert b.x0 == x_coords[0]
    assert b.xn == x_coords[-1]
    assert b.y0 == y_coords[0]
    assert b.yn == y_coords[-1]


def test_coords_training():
    """Check we get consistent target coodinates in the batch generator."""
    batchsize = 10
    width = 100
    height = 50

    # Fake up some image coord data, make coords equal indices
    x = np.arange(width)
    y = np.arange(height)

    # fake some labels coords
    rnd = np.random.RandomState(SEED)
    label_x = rnd.choice(width, 30, replace=False)
    label_y = rnd.choice(height, 30, replace=False)
    label_coords = np.stack((label_x, label_y)).T

    # Make the generator
    coord_gen = image.coords_training(label_coords, x, y, batchsize)

    label_accum = []
    for cx, cy in coord_gen:

        # Test sensible batch sizes
        assert len(cx) <= batchsize
        assert len(cy) <= batchsize

        label_accum.append((cx, cy))

    # Test we can reconstruct the labels array
    label_accum = np.concatenate(label_accum, axis=-1).T
    assert np.all(label_accum == label_coords)


def test_coords_query():
    """Check we get consistent image coodinates in the batch generator."""
    batchsize = 10
    width = 20
    height = 10
    # Fake up some image coord data, make coords equal indices
    x = np.arange(width)
    y = np.arange(height)
    xy = np.array(list(product(y, x)))[..., ::-1]

    # Make the generator
    coord_gen = image.coords_query(width, height, batchsize)

    coord_accum = []
    for cx, cy in coord_gen:

        # Test sensible batch sizes
        assert len(cx) <= batchsize
        assert len(cy) <= batchsize

        coord_accum.append((cx, cy))

    # Test we can reconstruct the labels array
    coord_accum = np.concatenate(coord_accum, axis=-1).T
    assert np.all(coord_accum == xy)

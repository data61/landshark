"""Tests for the image module."""
import numpy as np

from landshark import image


def test_pixel_coordinates(random_image_transform):
    """Test that pixel coordinates are valid for random examples."""
    data = random_image_transform
    coords_x, coords_y = image.pixel_coordinates(data.width, data.height,
                                                 data.affine)
    # outer corners of last pixel
    pix_x = np.arange(data.width + 1, dtype=np.float64)
    pix_y = np.arange(data.height + 1, dtype=np.float64)    # ditto
    true_coords_x = (pix_x * data.pixel_width) + data.origin_x
    true_coords_y = (pix_y * data.pixel_height) + data.origin_y
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
    true_coords_y = (h * data.pixel_height) + data.origin_y
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
    coords_y = (h * data.pixel_height) + data.origin_y
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
    coords_y = ((h.astype(float) + 0.5) * data.pixel_height) + data.origin_y
    idx_x = image.world_to_image(coords_x, pixel_coords_x)
    idx_y = image.world_to_image(coords_y, pixel_coords_y)

    true_idx_x = np.arange(data.width, dtype=int)
    true_idx_y = np.arange(data.height, dtype=int)

    assert np.all(true_idx_x == idx_x)
    assert np.all(true_idx_y == idx_y)


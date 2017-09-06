import numpy as np
from random import random, choice, randint
import rasterio.transform

from landshark import image


def test_pixel_coordinates():
    """Test that pixel coordinates are valid for random examples."""
    for i in range(100):
        pixel_width = random() * choice([-1, 1])
        pixel_height = random() * choice([-1, 1])
        origin_x = random() * choice([-1, 1])
        origin_y = random() * choice([-1, 1])
        width = randint(1,100)
        height = randint(1, 200)

        affine = rasterio.transform.from_origin(origin_x,
                                                origin_y,
                                                pixel_width,
                                                pixel_height)

        coords_x, coords_y = image.pixel_coordinates(width, height, affine)
        # outer corners of last pixel
        pix_x = np.arange(width + 1, dtype=np.float64)
        pix_y = np.arange(height + 1, dtype=np.float64)    # ditto
        true_coords_x = (pix_x * pixel_width) + origin_x
        true_coords_y = (pix_y * pixel_height) + origin_y
        assert np.allclose(true_coords_x, coords_x)
        assert np.allclose(true_coords_y, coords_y)


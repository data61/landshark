"""Tests for the image module."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import product

import numpy as np
import pytest

from landshark import image
from landshark.basetypes import IndexType

SEED = 666


def test_bounding_box():

    x_coords = np.arange(10)
    y_coords = np.arange(5)

    b = image.BoundingBox(x_coords, y_coords)
    assert b.xmin == x_coords[0]
    assert b.xmax == x_coords[-1]
    assert b.ymin == y_coords[0]
    assert b.ymax == y_coords[-1]

    b = image.BoundingBox(x_coords[::-1], y_coords[::-1])
    assert b.xmin == x_coords[0]
    assert b.xmax == x_coords[-1]
    assert b.ymin == y_coords[0]
    assert b.ymax == y_coords[-1]


def test_bounding_box_contains():
    x_coords = np.arange(10)
    y_coords = np.arange(5)

    b = image.BoundingBox(x_coords, y_coords)

    tests = np.array([[1, 1], [-1, 3], [3, -1], [8, 7], [0, 0], [9, 4], [10, 10]])
    result = b.contains(tests)
    ans = np.array([True, False, False, False, True, True, False])
    assert np.all(result == ans)


def test_image_spec(mocker):
    p_bbox = mocker.patch("landshark.image.BoundingBox")
    x_coords = np.arange(10)
    y_coords = np.arange(5)
    crs = {"init": "egs123"}
    spec = image.ImageSpec(x_coords, y_coords, crs)
    assert spec.width == 9
    assert spec.height == 4
    assert spec.bbox == p_bbox.return_value
    assert spec.crs == crs


def test_pixel_coordinates(random_image_transform):
    """Test that pixel coordinates are valid for random examples."""
    data = random_image_transform
    coords_x, coords_y = image.pixel_coordinates(data.width, data.height, data.affine)
    # outer corners of last pixel
    pix_x = np.arange(data.width + 1, dtype=np.float64)
    pix_y = np.arange(data.height + 1, dtype=np.float64)  # ditto
    true_coords_x = (pix_x * data.pixel_width) + data.origin_x
    true_coords_y = (pix_y * (-1.0 * data.pixel_height)) + data.origin_y
    assert np.allclose(true_coords_x, coords_x)
    assert np.allclose(true_coords_y, coords_y)


def test_image_to_world(random_image_transform):
    """Test that pixel coordinates are valid for random examples."""
    data = random_image_transform
    coords_x, coords_y = image.pixel_coordinates(data.width, data.height, data.affine)
    w = np.arange(data.width, dtype=IndexType)
    h = np.arange(data.height, dtype=IndexType)
    true_coords_x = (w * data.pixel_width) + data.origin_x
    true_coords_y = (h * (-1.0 * data.pixel_height)) + data.origin_y
    result_x = image.image_to_world(w, coords_x)
    result_y = image.image_to_world(h, coords_y)
    assert np.all(true_coords_x == result_x)
    assert np.all(true_coords_y == result_y)


def test_world_to_image_edges(random_image_transform):
    """Checks that pixel edges are correctly mapped to indices."""
    data = random_image_transform
    pixel_coords_x, pixel_coords_y = image.pixel_coordinates(
        data.width, data.height, data.affine
    )
    w = np.arange(data.width + 1, dtype=IndexType)
    h = np.arange(data.height + 1, dtype=IndexType)
    coords_x = (w * data.pixel_width) + data.origin_x
    coords_y = (h * (-1.0 * data.pixel_height)) + data.origin_y
    idx_x = image.world_to_image(coords_x, pixel_coords_x)
    idx_y = image.world_to_image(coords_y, pixel_coords_y)

    true_idx_x = np.array(list(range(data.width)) + [data.width - 1], dtype=IndexType)
    true_idx_y = np.array(list(range(data.height)) + [data.height - 1], dtype=IndexType)

    assert np.all(true_idx_x == idx_x)
    assert np.all(true_idx_y == idx_y)


def test_world_to_image_centers(random_image_transform):
    """Checks that pixel centres are correctly mapped to indices."""
    data = random_image_transform
    pixel_coords_x, pixel_coords_y = image.pixel_coordinates(
        data.width, data.height, data.affine
    )
    w = np.arange(data.width, dtype=IndexType)
    h = np.arange(data.height, dtype=IndexType)
    coords_x = ((w.astype(float) + 0.5) * data.pixel_width) + data.origin_x
    coords_y = ((h.astype(float) + 0.5) * (-1.0 * data.pixel_height)) + data.origin_y
    idx_x = image.world_to_image(coords_x, pixel_coords_x)
    idx_y = image.world_to_image(coords_y, pixel_coords_y)

    true_idx_x = np.arange(data.width, dtype=IndexType)
    true_idx_y = np.arange(data.height, dtype=IndexType)

    assert np.all(true_idx_x == idx_x)
    assert np.all(true_idx_y == idx_y)


@pytest.mark.parametrize("nstrips,rows,cols", [(1, 10, 3), (3, 3, 10), (4, 101, 102)])
def test_strip_image_spec(nstrips, rows, cols):
    # coords are 1 past last pixel
    x_coords = np.arange(cols + 1)
    y_coords = np.arange(rows + 1)
    crs = {"init": "egs123"}
    spec = image.ImageSpec(x_coords, y_coords, crs)
    specs = [image.strip_image_spec(i + 1, nstrips, spec) for i in range(nstrips)]
    for s in specs:
        assert s.crs == crs
        assert np.all(x_coords == s.x_coordinates)

    y_coords_new = np.concatenate(
        [a.y_coordinates[:-1] for a in specs[:-1]] + [specs[-1].y_coordinates], axis=0
    )
    assert np.all(y_coords == y_coords_new)


@pytest.mark.parametrize("nstrips,rows,cols", [(1, 10, 3), (3, 3, 10), (4, 101, 102)])
def test_indices_strip(nstrips, rows, cols):
    # coords are 1 past last pixel
    x_coords = np.arange(cols + 1)
    y_coords = np.arange(rows + 1)
    crs = {"init": "egs123"}
    spec = image.ImageSpec(x_coords, y_coords, crs)
    batchsize = 10
    xy_inds = []
    n = 0
    for i in range(nstrips):
        it_i, n_i = image.indices_strip(spec, i + 1, nstrips, batchsize)
        n += n_i
        for xy in it_i:
            assert xy.shape[0] <= batchsize
            assert xy.shape[1] == 2
            x, y = xy.T
            xy_inds.append(xy)
    xy_inds = np.concatenate(xy_inds, axis=0)
    assert n == rows * cols
    ans = np.fliplr(np.array(list(product(range(rows), range(cols)))))
    assert np.all(xy_inds == ans)


@pytest.mark.parametrize("total_size, nstrips", [(100, 4), (10, 10), (7, 2), (8, 1)])
def test_strip_slices(total_size, nstrips):
    slice_list = image._strip_slices(total_size, nstrips)
    assert len(slice_list) == nstrips
    assert slice_list[0].start == 0
    for i in range(1, len(slice_list)):
        assert slice_list[i].start == slice_list[i - 1].stop
    assert slice_list[-1].stop == total_size


def test_array_pair_it():
    """Test the convenience function for manipulating the batch iterators."""
    x = np.random.randn(10, 2)
    x_list = x.tolist()
    out = image._array_pair_it(x_list)
    assert np.all(out == np.fliplr(x))


def test_indices_query():
    """Check we get consistent image coodinates in the batch generator."""
    batchsize = 10
    width = 20
    height = 10
    # Fake up some image coord data, make coords equal indices
    x = np.arange(width)
    y = np.arange(height)
    xy = np.array(list(product(y, x)))[..., ::-1]

    # Make the generator
    coord_gen = image._indices_query(width, height, batchsize)

    coord_accum = []
    for c in coord_gen:

        # Test sensible batch sizes
        assert c.shape[0] <= batchsize
        assert c.shape[1] == 2

        coord_accum.append(c)

    # Test we can reconstruct the labels array
    coord_accum = np.concatenate(coord_accum)
    assert np.all(coord_accum == xy)


def test_random_indices():
    """Test selecting random indices into a image."""
    cols = 100
    rows = 200
    npoints = 10
    x_coords = np.arange(cols + 1)
    y_coords = np.arange(rows + 1)
    crs = {"init": "egs123"}
    spec = image.ImageSpec(x_coords, y_coords, crs)
    image.random_indices(spec, npoints, batchsize=1000, random_seed=220)

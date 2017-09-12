"""Tests for the data module."""

import numpy as np

from landshark import patch


def test_patches():
    """Check that patches are correctly created from points."""
    halfwidth = 1
    im_width = 5
    im_height = 5

    image = np.arange((im_height * im_width)).reshape((im_height, im_width))
    n = 2 * halfwidth + 1

    #  0,0 corner
    p_data = np.zeros((n, n), dtype=int) - 1
    x = 0
    y = 0
    p = patch.Patch(x, y, halfwidth, im_width, im_height)
    for yi, patch_yi in zip(p.y_indices, p.patch_y_indices):
        p_data[patch_yi, p.patch_x] = image[yi, p.x]
    true_answer = np.array([[-1, -1, -1],
                            [-1, 0, 1],
                            [-1, 5, 6]], dtype=int)
    true_mask = true_answer == -1
    assert np.all(true_answer == p_data)
    assert np.all(true_mask == p.mask)

    # 4,4 corner
    p_data = np.zeros((n, n), dtype=int) - 1
    x = 4
    y = 4
    p = patch.Patch(x, y, halfwidth, im_width, im_height)
    for yi, patch_yi in zip(p.y_indices, p.patch_y_indices):
        p_data[patch_yi, p.patch_x] = image[yi, p.x]
    true_answer = np.array([[18, 19, -1],
                            [23, 24, -1],
                            [-1, -1, -1]], dtype=int)
    true_mask = true_answer == -1
    assert np.all(true_answer == p_data)
    assert np.all(true_mask == p.mask)

    # 0,2 edge
    p_data = np.zeros((n, n), dtype=int) - 1
    x = 0
    y = 2
    p = patch.Patch(x, y, halfwidth, im_width, im_height)
    for yi, patch_yi in zip(p.y_indices, p.patch_y_indices):
        p_data[patch_yi, p.patch_x] = image[yi, p.x]
    true_answer = np.array([[-1, 5, 6],
                            [-1, 10, 11],
                            [-1, 15, 16]], dtype=int)
    true_mask = true_answer == -1
    assert np.all(true_answer == p_data)
    assert np.all(true_mask == p.mask)


    # 2,0 edge
    p_data = np.zeros((n, n), dtype=int) - 1
    x = 2
    y = 0
    p = patch.Patch(x, y, halfwidth, im_width, im_height)
    for yi, patch_yi in zip(p.y_indices, p.patch_y_indices):
        p_data[patch_yi, p.patch_x] = image[yi, p.x]
    true_answer = np.array([[-1, -1, -1],
                            [1, 2, 3],
                            [6, 7, 8]], dtype=int)
    true_mask = true_answer == -1
    assert np.all(true_answer == p_data)
    assert np.all(true_mask == p.mask)


"""Tests for the data module."""

import numpy as np

from landshark import patch


def test_patch_00():
    """Check that patches are correctly created from points in 00 corner."""
    halfwidth = 1
    im_width = 5
    im_height = 5

    image = np.arange((im_height * im_width)).reshape((im_height, im_width))
    n = 2 * halfwidth + 1

    #  0,0 corner
    p_data = np.zeros((n, n), dtype=int) - 1
    x = np.array([0])
    y = np.array([0])
    patch_rws, mask_ws = patch.patches(x, y, halfwidth, im_width, im_height)
    for r in patch_rws:
        p_data[r.yp, r.xp] = image[r.y, r.x]

    true_answer = np.array([[-1, -1, -1],
                            [-1, 0, 1],
                            [-1, 5, 6]], dtype=int)
    assert np.all(true_answer == p_data)


def test_patch_00_mask():
    """Check that patches are correctly created from points in 00 corner."""
    halfwidth = 1
    im_width = 5
    im_height = 5
    n = 2 * halfwidth + 1

    #  0,0 corner
    p_mask = np.zeros((n, n), dtype=bool)
    x = np.array([0])
    y = np.array([0])
    patch_rws, mask_ws = patch.patches(x, y, halfwidth, im_width, im_height)
    for r in mask_ws:
        p_mask[r.yp, r.xp] = True

    true_answer = np.array([[True, True, True],
                            [True, False, False],
                            [True, False, False]], dtype=bool)
    assert np.all(true_answer == p_mask)


def test_patch_44():
    """Check patch code in outer corner."""
    halfwidth = 1
    im_width = 5
    im_height = 5

    image = np.arange((im_height * im_width)).reshape((im_height, im_width))
    n = 2 * halfwidth + 1

    # 4,4 corner
    p_data = np.zeros((n, n), dtype=int) - 1
    x = np.array([4])
    y = np.array([4])
    patch_rws, mask_ws = patch.patches(x, y, halfwidth, im_width, im_height)
    for r in patch_rws:
        p_data[r.yp, r.xp] = image[r.y, r.x]
    true_answer = np.array([[18, 19, -1],
                            [23, 24, -1],
                            [-1, -1, -1]], dtype=int)
    assert np.all(true_answer == p_data)


def test_patch_44_mask():
    """Check that patches are correctly created from points in 00 corner."""
    halfwidth = 1
    im_width = 5
    im_height = 5
    n = 2 * halfwidth + 1

    #  0,0 corner
    p_mask = np.zeros((n, n), dtype=bool)
    x = np.array([4])
    y = np.array([4])
    patch_rws, mask_ws = patch.patches(x, y, halfwidth, im_width, im_height)
    for r in mask_ws:
        p_mask[r.yp, r.xp] = True

    true_answer = np.array([[False, False, True],
                            [False, False, True],
                            [True, True, True]], dtype=bool)
    assert np.all(true_answer == p_mask)


def test_patch_02():
    """Check patch code in x edge."""
    halfwidth = 1
    im_width = 5
    im_height = 5

    image = np.arange((im_height * im_width)).reshape((im_height, im_width))
    n = 2 * halfwidth + 1

    # 0,2 edge
    p_data = np.zeros((n, n), dtype=int) - 1
    x = np.array([0])
    y = np.array([2])

    patch_rws, mask_ws = patch.patches(x, y, halfwidth, im_width, im_height)
    for r in patch_rws:
        p_data[r.yp, r.xp] = image[r.y, r.x]
    true_answer = np.array([[-1, 5, 6],
                            [-1, 10, 11],
                            [-1, 15, 16]], dtype=int)
    assert np.all(true_answer == p_data)


def test_patch_02_mask():
    """Check that patches are correctly created from points in 00 corner."""
    halfwidth = 1
    im_width = 5
    im_height = 5
    n = 2 * halfwidth + 1

    #  0,0 corner
    p_mask = np.zeros((n, n), dtype=bool)
    x = np.array([0])
    y = np.array([2])
    patch_rws, mask_ws = patch.patches(x, y, halfwidth, im_width, im_height)
    for r in mask_ws:
        p_mask[r.yp, r.xp] = True

    true_answer = np.array([[True, False, False],
                            [True, False, False],
                            [True, False, False]], dtype=bool)

    assert np.all(true_answer == p_mask)


def test_patch_20():
    """Check patch code in y edge."""
    halfwidth = 1
    im_width = 5
    im_height = 5

    image = np.arange((im_height * im_width)).reshape((im_height, im_width))
    n = 2 * halfwidth + 1

    # 2,0 edge
    p_data = np.zeros((n, n), dtype=int) - 1
    x = np.array([2])
    y = np.array([0])
    patch_rws, mask_ws = patch.patches(x, y, halfwidth, im_width, im_height)
    for r in patch_rws:
        p_data[r.yp, r.xp] = image[r.y, r.x]
    true_answer = np.array([[-1, -1, -1],
                            [1, 2, 3],
                            [6, 7, 8]], dtype=int)
    assert np.all(true_answer == p_data)


def test_patch_20_mask():
    """Check that patches are correctly created from points in 00 corner."""
    halfwidth = 1
    im_width = 5
    im_height = 5
    n = 2 * halfwidth + 1

    #  0,0 corner
    p_mask = np.zeros((n, n), dtype=bool)
    x = np.array([2])
    y = np.array([0])
    patch_rws, mask_ws = patch.patches(x, y, halfwidth, im_width, im_height)
    for r in mask_ws:
        p_mask[r.yp, r.xp] = True

    true_answer = np.array([[True, True, True],
                            [False, False, False],
                            [False, False, False]], dtype=bool)

    assert np.all(true_answer == p_mask)

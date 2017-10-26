"""Tests for the category object in the importer code."""

import numpy as np
from landshark.importers import category

# TODO write lower level tests

def test_category_obj_missing():
    missing_values = [None, 11]
    cat = category._Categories(missing_values)
    assert cat.missing_values == [None, 0]

def test_category_obj():

    missing_values = [None, 11]
    data = np.array([[0, 10],
                     [1, 11],
                     [0, 10],
                     [4, 14],
                     [5, 15],
                     [1, 11],
                     [6, 16]])

    data_list = [data[0:2], data[2:5], data[5:]]
    new_list = []
    cat = category._Categories(missing_values)
    for d in data_list:
        new_list.append(cat.update(d))
    new_data = np.concatenate(new_list, axis=0)
    answer = np.array([[0, 1],
                       [1, 0],
                       [0, 1],
                       [2, 2],
                       [3, 3],
                       [1, 0],
                       [4, 4]])
    assert np.all(new_data == answer)

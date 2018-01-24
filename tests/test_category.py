"""Tests for the category object in the importer code."""

import numpy as np
from landshark import category
from landshark.basetypes import CategoricalValues


def test_category_preprocessor():
    ncols = 3
    p = category._CategoryPreprocessor(ncols)
    in_data = np.array([[1, 2, 2], [1, 2, 3],
                        [1, 1, 2], [1, 1, 1]])
    x = CategoricalValues(in_data)
    unique_vals, counts = p(x)
    true_vals = [np.array([1]), np.array([1, 2]), np.array([1, 2, 3])]
    true_counts = [np.array([4]), np.array([2, 2]), np.array([1, 2, 1])]

    for v, w in zip(unique_vals, true_vals):
        assert np.all(v == w)
    for v, w in zip(counts, true_counts):
        assert np.all(v == w)


def test_category_accumulator():
    pass



    # in_data_2 = np.array([[2, 3, 4], [1, 5, 6],
    #                       [1, 1, 1], [1, 8, 8]])
    # y = CategoricalValues(in_data_2)
    # new_unique, new_counts = p(y)
    # import IPython; IPython.embed(); import sys; sys.exit()


# def test_category_obj_missing():
#     missing_values = [None, 11]
#     cat = category._Categories(missing_values)
#     assert cat.missing_values == [None, 0]

# def test_category_obj():

#     missing_values = [None, 11]
#     data = np.array([[0, 10],
#                      [1, 11],
#                      [0, 10],
#                      [4, 14],
#                      [5, 15],
#                      [1, 11],
#                      [6, 16]])

#     data_list = [data[0:2], data[2:5], data[5:]]
#     new_list = []
#     cat = category._Categories(missing_values)
#     for d in data_list:
#         new_list.append(cat.update(d))
#     new_data = np.concatenate(new_list, axis=0)
#     answer = np.array([[0, 1],
#                        [1, 0],
#                        [0, 1],
#                        [2, 2],
#                        [3, 3],
#                        [1, 0],
#                        [4, 4]])
#     assert np.all(new_data == answer)

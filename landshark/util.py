import numpy as np

from typing import List, Union

from landshark.basetypes import CategoricalType, OrdinalType

MissingValueList = List[Union[OrdinalType, CategoricalType, None]]

def to_masked(array: np.ndarray, missing_values: MissingValueList) \
        -> np.ma.MaskedArray:
    """Create a masked array from array plus list of missing."""
    assert len(missing_values) == array.shape[-1]
    mask = np.zeros_like(array, dtype=bool)
    for i, m in enumerate(missing_values):
        if m:
            mask[..., i] = array[..., i] == m
    marray = np.ma.MaskedArray(data=array, mask=mask)
    return marray


import numpy as np

from landshark.basetypes import MissingType


def to_masked(array: np.ndarray, missing_value: MissingType) \
        -> np.ma.MaskedArray:
    """Create a masked array from array plus list of missing."""
    if missing_value is None:
        marray = np.ma.MaskedArray(data=array, mask=np.ma.nomask)
    else:
        mask = array == missing_value
        marray = np.ma.MaskedArray(data=array, mask=mask)
    return marray


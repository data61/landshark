"""Configuration for test suite."""

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

from collections import namedtuple

import numpy as np
import pytest
import rasterio.transform

TestImageData = namedtuple(
    "TestImageData",
    [
        "pixel_width",
        "pixel_height",
        "origin_x",
        "origin_y",
        "width",
        "height",
        "affine",
    ],
)


@pytest.fixture(params=list(range(100)))
def random_image_transform(request):
    """Make a bunch of random image transforms."""
    r = np.random.RandomState(request.param)
    pixel_width = r.uniform(-5, 5)
    pixel_height = r.uniform(-5, 5)
    origin_x = r.uniform(-10, 10)
    origin_y = r.uniform(-10, 10)
    width = r.randint(1, 100)
    height = r.randint(1, 200)
    affine = rasterio.transform.from_origin(
        origin_x, origin_y, pixel_width, pixel_height
    )

    data = TestImageData(
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        origin_x=origin_x,
        origin_y=origin_y,
        width=width,
        height=height,
        affine=affine,
    )
    return data

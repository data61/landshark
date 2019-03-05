"""Test fileio modeule."""

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

import pytest

from landshark.fileio import tifnames

tifs = [
    "./integration/data/categorical/PM_Lithology_Unit_Type_missing.tif",
    "./integration/data/categorical/Soil_ASC_missing.tif",
    "./integration/data/continuous/SirSamualAusGravity2011_ll_missing.tif",
    "./integration/data/continuous/SirSamualAusMagV6_rtp_ll_missing.tif",
    "./integration/data/nonmissing/PM_Lithology_Unit_Type.tif",
    "./integration/data/nonmissing/SirSamualAusGravity2011_ll.tif",
    "./integration/data/nonmissing/SirSamualAusMagV6_rtp_ll.tif",
    "./integration/data/nonmissing/Soil_ASC.tif",
]

dirs_tifs = [
    (["./integration/data"], tifs),
    (["./integration/data/categorical/"], tifs[:2]),
    (["./integration/data/continuous/",
      "./integration/data/categorical/"], tifs[:4]),
]


@pytest.mark.parametrize("dirs,tifs", dirs_tifs)
def test_tifnames(dirs, tifs):
    assert set(tifnames(dirs)) == set(tifs)

# """Tests for the shapefile reading importer module."""

# import numpy as np

# from landshark.importers import shpread


# def test_shapefile_float_fields():
#     """Checks that it ignores the first field and returns float fields."""

#     fields = [("1", "N", "10"),
#               ("2", "N", "11"),
#               ("3", "C", "12"),
#               ("4", "N", "13")]
#     result = shpread._shapefile_float_fields(fields)
#     assert list(result) == [(0, '2'), (2, '4')]


# def test_shapefiletargets(mocker):
#     """Checks that the constructor calls the right functions."""

#     filename = "path"
#     m_reader = mocker.patch("landshark.importers.shpread.shapefile.Reader")
#     m_sf = mocker.MagicMock()
#     m_reader.return_value = m_sf
#     m_sf.fields = [("1", "N", "10"),
#                    ("2", "N", "11"),
#                    ("3", "C", "12"),
#                    ("4", "N", "13")]

#     m_sf.iterShapes = mocker.MagicMock()
#     m_sf.iterRecords = mocker.MagicMock()

#     m_sf.iterShapes.return_value = [mocker.MagicMock(), mocker.MagicMock()]
#     m_sf.iterRecords.return_value = [range(5), range(5)]
#     m_sf.record.side_effect = [range(5, 10)]

#     m_sf.iterShapes.return_value[0].__geo_interface__ = {'coordinates': 666}
#     m_sf.iterShapes.return_value[1].__geo_interface__ = {'coordinates': 667}

#     sft = shpread.ShapefileTargets(filename)
#     assert sft.n == m_sf.numRecords
#     assert sft._field_indices == [0, 2]
#     assert sft.fields == ["2", "4"]
#     assert sft.dtype == np.float32

#     # Read the whole shape file
#     coords = np.array(list(sft.coordinates()))
#     assert np.all(coords == np.array([666.0, 667.0], dtype=np.float64))
#     ord_data = list(sft.ordinal_data())
#     assert np.all(ord_data[0] == np.array([0., 2.], dtype=np.float32))
#     assert np.all(ord_data[1] == np.array([0., 2.], dtype=np.float32))

#     # Read indices of the shape file
#     sft = shpread.ShapefileTargets(filename, np.array([0]))
#     assert sft.n == 1
#     ord_data = list(sft.ordinal_data())
#     assert len(ord_data) == 1
#     assert np.all(ord_data[0] == np.array([5., 7.], dtype=np.float32))


# def test_train_test_targets(mocker):
#     """Checks that we get the right sized splits."""

#     filename = "path"
#     m_reader = mocker.patch("landshark.importers.shpread.shapefile.Reader")
#     m_sf = mocker.MagicMock()
#     m_reader.return_value = m_sf
#     m_sf.fields = [("1", "N", "10"),
#                    ("2", "N", "11"),
#                    ("3", "C", "12"),
#                    ("4", "N", "13")]
#     m_sf.numRecords = 4

#     sf_tr, sf_ts = shpread.train_test_targets(filename, 1. / 4, 1)
#     assert sf_tr.n == 3
#     assert sf_ts.n == 1
#     assert len(list(sf_tr.ordinal_data())) == 3
#     assert len(list(sf_ts.ordinal_data())) == 1

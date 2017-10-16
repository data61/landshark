"""Tests for target writing importer module."""
from landshark.importers import targetwrite


def test_write_targetfile(mocker):
    """Checks that the writer calls the correct functions."""
    call = mocker.mock_module.call
    m_open = mocker.patch('landshark.importers.targetwrite.tables.open_file')
    m_hfile = mocker.MagicMock()
    m_open.return_value = m_hfile
    filename = 'path'
    m_sf = mocker.MagicMock()
    m_size = mocker.patch('landshark.importers.targetwrite.os.path.getsize')

    targetwrite.write_targetfile(m_sf, filename)
    m_open.assert_called_once_with(filename, mode='w',
                                   title='Landshark Targets')
    # TODO this code is going to be deleted soon

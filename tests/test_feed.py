import numpy as np
from landshark import feed
from landshark.patch import PatchRowRW, PatchMaskRowRW


class FakeFeatures:

    def __init__(self, width, height, missing_vals):
        self.missing_values = missing_vals
        k = len(missing_vals)
        self.nfeatures = k
        self.dtype = np.float32
        self._data = np.arange(width * height * k).reshape((height, width, k))

    def __call__(self, y, x_slice):
        return self._data[y, x_slice]


def test_read():
    """Check that read extracts data correctly into the patches."""
    width = 10
    height = 8
    npatches = 3
    patchwidth = 5
    missing_values = [3, 10, 5]
    nfeatures = len(missing_values)
    data = FakeFeatures(width, height, missing_values)

    patch_ops = [PatchRowRW(idx=0, x=slice(0, 2), y=0, xp=slice(0, 2), yp=2),
                 PatchRowRW(idx=1, x=slice(0, 1), y=0, xp=slice(1, 2), yp=3)]
    mask_ops = [PatchMaskRowRW(idx=0, xp=slice(1, 3), yp=2),
                PatchMaskRowRW(idx=2, xp=slice(0, 3), yp=1)]

    result = feed._read(data, patch_ops, mask_ops, npatches, patchwidth, True)
    assert result.shape == (npatches, patchwidth, patchwidth, nfeatures)
    for p in patch_ops:
        rd = result[p.idx, p.yp, p.xp].data
        rm = result[p.idx, p.yp, p.xp].mask
        assert np.all(rd == data(p.y, p.x))
        for i, mv in enumerate(missing_values):
            assert not np.any(np.logical_and(rm[:, i] == False,
                                             (rd[:, i] == mv)))
    for m in mask_ops:
        assert np.all(result[m.idx, m.yp, m.xp].mask)


def test_read_batch(mocker):
    indices_x = np.arange(10)
    indices_y = np.arange(10)
    features = mocker.Mock()
    features.ord = mocker.Mock()
    features.cat = mocker.Mock()
    halfwidth = 2
    patchwidth = 2 * halfwidth + 1
    m_patches = mocker.patch('landshark.feed.patch.patches')
    patch_specs = (mocker.Mock(), mocker.Mock())
    m_patches.return_value = patch_specs
    m_read = mocker.patch('landshark.feed._read')
    m_read.side_effect = [1, 2]
    result = feed._read_batch(indices_x, indices_y, features, halfwidth)
    m_patches.assert_has_calls([
    mocker.call(indices_x, indices_y, halfwidth, features.image_spec.width,
                features.image_spec.height)
    ])
    m_read.assert_has_calls([
        mocker.call(features.ord, patch_specs[0],
                    patch_specs[1], 10, patchwidth),
        mocker.call(features.cat, patch_specs[0],
                    patch_specs[1], 10, patchwidth)
    ])
    assert result == (1, 2)


def test_training_data(mocker):
    feats = mocker.Mock()
    targets = mocker.Mock()
    m_training = mocker.MagicMock(return_value=[(0, 0, 666), (1, 1, 666)])
    targets.training = m_training
    m_read_batch = mocker.patch('landshark.feed._read_batch')
    m_read_batch.return_value = mocker.Mock(), mocker.Mock()
    batchsize = 10
    halfwidth = 2
    result = list(feed.training_data(feats, targets, batchsize, halfwidth))
    m_training.assert_has_calls([mocker.call(feats.image_spec, batchsize)])
    m_read_batch.assert_has_calls([
        mocker.call(0, 0, feats, halfwidth),
        mocker.call(1, 1, feats, halfwidth)])
    assert len(result) == 2
    for r in result:
        assert r == feed.TrainingBatch(x_ord=m_read_batch.return_value[0],
                                       x_cat=m_read_batch.return_value[1],
                                       y=666)

def test_query_data(mocker):
    feats = mocker.Mock()
    m_indices = mocker.MagicMock(return_value=[(0, 0), (1, 1)])
    feats.pixel_indices = m_indices
    m_read_batch = mocker.patch('landshark.feed._read_batch')
    m_read_batch.return_value = mocker.Mock(), mocker.Mock()
    batchsize = 10
    halfwidth = 2
    result = list(feed.query_data(feats, batchsize, halfwidth))
    m_indices.assert_has_calls([mocker.call(batchsize)])
    m_read_batch.assert_has_calls([
        mocker.call(0, 0, feats, halfwidth),
        mocker.call(1, 1, feats, halfwidth)])
    assert len(result) == 2
    for r in result:
        assert r == feed.QueryBatch(x_ord=m_read_batch.return_value[0],
                                    x_cat=m_read_batch.return_value[1])

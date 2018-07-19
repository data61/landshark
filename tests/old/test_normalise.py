# """Tests for landshark/importers/normalise.py."""

# import numpy as np

# from landshark.importers import normalise


# def test_statistics():
#     n_features = 2
#     n_rows = 10
#     n_blocks = 5
#     stats = normalise._Statistics(n_features)
#     data = [np.random.randn(n_rows, n_features) for i in range(n_blocks)]
#     all_data = np.concatenate(data, axis=0)
#     for d in data:
#         stats.update(d)
#     mean = stats.mean
#     var = stats.variance
#     true_mean = np.mean(all_data, axis=0)
#     true_var = np.var(all_data, axis=0)
#     assert np.allclose(mean, true_mean)
#     assert np.allclose(var, true_var)


# def test_statistics_masked():
#     n_features = 2
#     n_rows = 10
#     n_blocks = 5
#     stats = normalise._Statistics(n_features)
#     data = [np.random.randn(n_rows, n_features) for i in range(n_blocks)]
#     masks = [np.random.choice(2, size=(n_rows, n_features)).astype(bool)
#              for i in range(n_blocks)]
#     all_data = np.concatenate(data, axis=0)
#     all_masks = np.concatenate(masks, axis=0)
#     all_marray = np.ma.MaskedArray(data=all_data, mask=all_masks)
#     m_data = [np.ma.MaskedArray(data=d, mask=m) for d, m in zip(data, masks)]
#     for d in m_data:
#         stats.update(d)
#     mean = stats.mean
#     var = stats.variance
#     true_mean = np.ma.mean(all_marray, axis=0)
#     true_var = np.ma.var(all_marray, axis=0)
#     assert np.allclose(mean, true_mean)
#     assert np.allclose(var, true_var)

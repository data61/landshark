"""Test kfold module."""

import pytest

from landshark.kfold import KFolds

fold_params = [
    (10, 2, 5),
    (123456, 10, 99)
]


@pytest.mark.parametrize("N,K,B", fold_params)
def test_kfolds(N, K, B):
    folds = KFolds(N, K)
    ixs = list(folds.iterator(B))
    bs = [len(b) for b in ixs]
    assert bs == [B] * (N // B) + [] if N % B == 0 else [N % B]
    ixs_flat = [i for b in ixs for i in b]
    assert len(set(ixs_flat)) == K
    assert min(ixs_flat) > 0
    assert max(ixs_flat) <= K
    assert set(folds.counts.keys()) == set(range(1, K + 1))
    assert sum(folds.counts.values()) == N

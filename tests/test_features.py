import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deeplob.features import book_features  # noqa: E402


def test_features_use_only_the_past():
    rng = np.random.default_rng(0)
    lob = rng.uniform(1, 2, (300, 40))
    X1, names = book_features(lob, np.array([150]))
    lob2 = lob.copy()
    lob2[151:] = rng.uniform(1, 2, (149, 40))   # change the future only
    X2, _ = book_features(lob2, np.array([150]))
    np.testing.assert_array_equal(X1, X2)
    assert len(names) == X1.shape[1] == 8 + 1 + 4 + 40
    imb = X1[0, [names.index(f"imb_{n}") for n in (1, 3, 5, 10)]]
    assert np.all(np.abs(imb) <= 1)

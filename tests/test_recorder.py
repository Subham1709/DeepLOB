import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deeplob.recorder import LEVELS, Buffer, load_chunk  # noqa: E402


def test_chunk_roundtrip_is_exact(tmp_path):
    rng = np.random.default_rng(0)
    buf = Buffer("BTCUSDT", tick=0.01, step=0.00001)
    snaps = []
    for k in range(50):
        mid = 81798.0 + 0.01 * rng.integers(-20, 20)
        asks = [[f"{mid + 0.01 * (i + 1):.2f}", f"{rng.integers(1, 10**6) * 1e-5:.5f}"] for i in range(20)]
        bids = [[f"{mid - 0.01 * i:.2f}", f"{rng.integers(1, 10**6) * 1e-5:.5f}"] for i in range(20)]
        buf.add(1_700_000_000_000 + 100 * k, {"lastUpdateId": 10**11 + 7 * k, "asks": asks, "bids": bids})
        snaps.append((asks, bids))
    z = load_chunk(buf.flush(tmp_path))
    assert z["ask_p"].shape == (50, LEVELS)
    assert np.all(np.diff(z["t"]) == 100)
    for k, (asks, bids) in enumerate(snaps):
        np.testing.assert_allclose(z["ask_p"][k], [float(p) for p, _ in asks[:LEVELS]], atol=1e-9)
        np.testing.assert_allclose(z["bid_q"][k], [float(q) for _, q in bids[:LEVELS]], atol=1e-12)

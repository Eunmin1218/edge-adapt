"""pack / pack16 / pack4 부호화의 왕복(encode → decode) 시험: 채널별 배율·이동으로 접은 어댑터가 복원되고, 비트 수가 낮을수록 바이트가 줄며 오차는 허용 범위 안이다."""
import numpy as np
from src.delta_codec import encode, decode


def _toy(seed=0):
    rng = np.random.default_rng(seed); base, ad = {}, {}
    for n, (c, k) in enumerate([(8, 3), (16, 3), (12, 1)]):
        w = rng.normal(0, 0.2, (c, 4, k, k)).astype(np.float32); b = rng.normal(0, 0.1, c).astype(np.float32)
        base[f"m{n}.weight"], base[f"m{n}.bias"] = w, b
        s = 1 + rng.normal(0, 0.1, c).astype(np.float32); t = rng.normal(0, 0.05, c).astype(np.float32)
        ad[f"m{n}.weight"], ad[f"m{n}.bias"] = w * s.reshape(-1, 1, 1, 1), s * b + t
    return base, ad


def test_pack_roundtrip_error_shrinks_with_bits_and_bytes_shrink():
    base, ad = _toy(); sizes, errs = {}, {}
    for mode in ("pack16", "pack", "pack4"):
        raw = encode(ad, base, mode); out = decode(raw, base); sizes[mode] = len(raw)
        errs[mode] = max(float(np.abs(out[k] - ad[k]).max()) for k in ad)
    assert errs["pack16"] < 5e-3 and errs["pack"] < 2e-2 and errs["pack4"] < 0.2     # 허용 오차: 비트가 줄수록 커진다
    assert errs["pack16"] <= errs["pack"] <= errs["pack4"] + 1e-9
    assert sizes["pack4"] <= sizes["pack"] <= sizes["pack16"] + 8


def test_pack_keys_preserved():
    base, ad = _toy(1)
    for mode in ("pack16", "pack", "pack4"):
        assert set(decode(encode(ad, base, mode), base)) == set(ad)

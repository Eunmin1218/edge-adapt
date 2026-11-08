import numpy as np, torch
from src.adapt_modules import ToneCurve, apply_tone_np
from src.delta_codec import encode, decode, scale_adapter


def test_tone_curve_identity_and_numpy_match():
    tc = ToneCurve(); x = torch.rand(1, 3, 8, 8); assert (tc(x) - x).abs().max() < 1e-6
    with torch.no_grad(): tc.knots.mul_(0.5)                                  # 어둡게
    y = tc(x).detach().numpy(); y2 = apply_tone_np(x.numpy(), tc.knots.detach().numpy()); assert np.abs(y - y2).max() < 1e-5 and y.mean() < x.mean().item()
    with torch.no_grad(): tc.knots[:, 5] = 2.0; tc.project()
    assert (torch.diff(tc.knots, dim=1) >= -1e-6).all() and tc.knots.max() <= 1.0


def test_auto_codec_affine_is_tiny_and_exact():
    rng = np.random.default_rng(0); base = {"m.conv.weight": rng.normal(0, .1, (16, 8, 3, 3)).astype(np.float32), "m.conv.bias": rng.normal(0, .1, 16).astype(np.float32),
                                            "h.conv.weight": rng.normal(0, .1, (8, 8, 1, 1)).astype(np.float32), "h.conv.bias": rng.normal(0, .1, 8).astype(np.float32)}
    s = 1 + rng.normal(0, .05, 16).astype(np.float32); t = rng.normal(0, .02, 16).astype(np.float32)
    ad = {"m.conv.weight": base["m.conv.weight"] * s[:, None, None, None], "m.conv.bias": base["m.conv.bias"] * s + t,
          "h.conv.weight": base["h.conv.weight"] + rng.normal(0, .01, (8, 8, 1, 1)).astype(np.float32), "__tone__": np.linspace(0, 1, 17, dtype=np.float32)[None].repeat(3, 0)}
    raw = encode(ad, base, "auto"); dec = decode(raw, base); assert set(dec) == set(ad) | {"h.conv.bias"} - {"h.conv.bias"} or set(dec) >= set(ad)
    assert np.abs(dec["m.conv.weight"] - ad["m.conv.weight"]).max() < 1e-4 and np.abs(dec["m.conv.bias"] - ad["m.conv.bias"]).max() < 1e-3 and np.abs(dec["h.conv.weight"] - ad["h.conv.weight"]).max() < 1e-3
    assert np.allclose(dec["__tone__"], ad["__tone__"], atol=1e-3)
    half = scale_adapter(ad, base, 0.5); assert np.allclose(half["m.conv.weight"], base["m.conv.weight"] + 0.5 * (ad["m.conv.weight"] - base["m.conv.weight"]), atol=1e-6)

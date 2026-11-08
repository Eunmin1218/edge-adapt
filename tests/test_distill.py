import numpy as np, torch
from src.distill import distill_loss, strong_aug, _net_labels
from src.delta_codec import encode, decode


def test_distill_loss_zero_when_equal_and_positive_otherwise():
    torch.manual_seed(0); f = [torch.randn(2, 70, h, w) for h, w in ((8, 12), (4, 6), (2, 3))]
    l0, _, _ = distill_loss(f, [x.clone() for x in f], 64); l1, _, _ = distill_loss(f, [torch.randn_like(x) for x in f], 64)
    assert l1 > l0 > 0 and not torch.isnan(l1)          # BCE 소프트 타깃은 일치해도 0 이 아님(엔트로피) — 다를 때 더 크다
    f2 = [x.clone().requires_grad_(True) for x in f]; distill_loss(f2, f, 64)[0].backward(); assert all(x.grad is not None for x in f2)


def test_aug_keeps_shape_and_labels_flip():
    im = np.random.default_rng(0).integers(0, 255, (320, 640, 3), dtype=np.uint8); assert strong_aug(im, np.random.default_rng(1)).shape == im.shape
    lb = np.array([[1, 0.25, 0.5, 0.1, 0.2]]); a, b = _net_labels(lb, False), _net_labels(lb, True); assert abs(a[0, 1] + b[0, 1] - 1) < 1e-6 and a[0, 2] == b[0, 2]


def test_codec_roundtrip():
    rng = np.random.default_rng(0); base = {"a": rng.normal(0, .1, (8, 4, 3, 3)).astype(np.float32)}; ad = {"a": base["a"] + rng.normal(0, .01, (8, 4, 3, 3)).astype(np.float32)}
    for m in ("fp16", "d8", "sparse"):
        d = decode(encode(ad, base, m, keep=0.5), base)["a"]; assert d.shape == ad["a"].shape
    assert np.abs(decode(encode(ad, base, "d8"), base)["a"] - ad["a"]).max() < 2e-4

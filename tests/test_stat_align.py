import numpy as np, torch, tempfile
from pathlib import Path
from ultralytics import YOLO
from src.common import IMG_H, IMG_W
from src.server_core import save_pt
from src.stat_align import StatAligner
from src.onnx_detector import preprocess
from src.adapt_modules import _targets
from ultralytics.nn.modules import Conv


def _setup():
    d = Path(tempfile.mkdtemp()); y = YOLO("yolo11n.yaml"); save_pt(y, d / "m.pt"); rng = np.random.default_rng(0)
    src = [rng.integers(60, 200, (IMG_H, IMG_W, 3)).astype(np.uint8) for _ in range(6)]; return StatAligner(d / "m.pt", src), src


def test_identity_on_source_frames():
    sa, src = _setup(); st, sig = sa.fit(src)
    s = torch.cat([a.flatten() for a, b in st]); t = torch.cat([b.flatten() for a, b in st])
    assert (s - 1).abs().median() < 0.05 and t.abs().median() < 0.05            # 같은 프레임이면 거의 항등


def test_darkening_is_moved_toward_source_stats_and_fold_matches_hook():
    sa, src = _setup(); dark = [(f * 0.3).astype(np.uint8) for f in src]; st, sig = sa.fit(dark)
    assert float(sig[0]) > 0.05                                                  # 첫 층 로그 배율 평균이 양수: 어두워서 키워야 함
    x = torch.from_numpy(np.concatenate([preprocess(f) for f in dark])); y = YOLO(sa.pt); y.model.float().eval().fuse(verbose=False)
    lay = [(o, c) for o, c, a in _targets(y.model) if isinstance(o, Conv)]
    with torch.no_grad():
        for (o, c), (s, t) in zip(lay, st): c.weight.mul_(s.view(-1, 1, 1, 1)); c.bias.copy_(c.bias * s + t)
        first = []; h = lay[0][1].register_forward_hook(lambda m, i, out: first.append(out.mean().item())); y.model(x); h.remove()
        raw = []; h = sa.layers[0][1].register_forward_hook(lambda m, i, out: raw.append(out.mean().item())); sa.model(x); h.remove()
    mu_s = float(sa.src[0][0].mean()); assert abs(first[0] - mu_s) < abs(raw[0] - mu_s)   # 접은 모델의 첫 층 평균이 소스 평균에 더 가깝다

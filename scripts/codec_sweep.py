"""저장된 서버 어댑터(ad_<env>.pkl)로 압축 곡선을 잰다: 비트 수 b ∈ {8,6,4,3} × 남기는 비율 r ∈ {1, .5, .25, .1}.
affine 텐서는 (s-1, t) 벡터를, 그 밖의 텐서는 base 대비 변화량을 텐서 단위 대칭 b비트 양자화 + |값| 상위 r 비율만 남기고(나머지 0) 정수열을 zlib 로 묶어 바이트를 센다.
복원한 어댑터로 mAP50 을 재서 (바이트, mAP) 곡선을 만든다. ES 에도 그대로 적용 가능한 공통 압축이므로 우리 고유 이득이 아니다 — 가능한 한계를 알려고 재는 것."""
import sys, zlib, pickle, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT
from src.server_core import onnx_inits
from src.onnx_detector import OnnxDetector
from src.delta_codec import _affine_fit
from src.mapeval import evaluate

OUT = ROOT / "results" / "dev"; OUT.mkdir(parents=True, exist_ok=True); ENV = sys.argv[1] if len(sys.argv) > 1 else "night"; W = ROOT / "runs/toy3"; base = onnx_inits(W / "student.onnx")
D = pickle.load(open(OUT / f"ad_{ENV}.pkl", "rb")); ad, test = D["ad_srv"], D["test"]


def mAP(a):
    det = OnnxDetector(W / "student.onnx"); det.apply_adapter(a) if a else None; P = []
    for img, gt in test:
        d = det.infer(img); P.append((d["boxes"], d["scores"], d["cls"]))
    return evaluate(P, [g for _, g in test], 6)["map50"]


# 벡터화: 각 항목 = (키, 종류, 벡터)
items = []
for k, v in ad.items():
    v = np.asarray(v, np.float32)
    if k.endswith(".weight") and v.ndim == 4 and k in base:
        s = _affine_fit(v, np.asarray(base[k], np.float32)); bn = k[:-6] + "bias"
        if s is not None and bn in base:
            bb = np.asarray(base[bn], np.float32); items.append((k, "aff", np.r_[s - 1, np.asarray(ad.get(bn, bb), np.float32) - s * bb])); continue
    if k.endswith(".bias") and k[:-4] + "weight" in ad and k[:-4] + "weight" in base and np.asarray(ad[k[:-4] + "weight"]).ndim == 4 and _affine_fit(np.asarray(ad[k[:-4] + "weight"], np.float32), np.asarray(base[k[:-4] + "weight"], np.float32)) is not None: continue
    items.append((k, "dense", (v - np.asarray(base[k], np.float32)).ravel()))
nb_aff = sum(len(v) for _, t, v in items if t == "aff"); nb_den = sum(len(v) for _, t, v in items if t == "dense")
print(f"원소 수: affine 벡터 {nb_aff}  dense 변화량 {nb_den}  (dense 쪽이 {nb_den / max(nb_aff + nb_den, 1):.0%})")
print("dense 상위:", sorted([(len(v), k) for k, t, v in items if t == "dense"], reverse=True)[:5])


def quant(x, bits, keep):
    """대칭 bits 비트, |x| 상위 keep 비율만. 반환 (복원값, 바이트추정용 정수열)"""
    n = len(x); m = max(1, int(round(keep * n))); idx = np.argsort(-np.abs(x))[:m] if m < n else np.arange(n)
    q = np.zeros(n, np.int16); mx = np.abs(x[idx]).max() if m else 0; L = 2 ** (bits - 1) - 1; sc = mx / L if mx > 0 else 1.0
    q[idx] = np.clip(np.rint(x[idx] / sc), -L, L).astype(np.int16); return q.astype(np.float32) * np.float32(sc), q, np.float16(sc)


def build(bits, keep):
    parts, rec = [], {}
    for k, t, v in items:
        r, q, sc = quant(v, bits, keep); rec[k] = r
        if keep < 1: nz = np.flatnonzero(q); gaps = np.diff(nz, prepend=-1).astype(np.uint16); parts.append(sc.tobytes() + gaps.tobytes() + q[nz].astype(np.int8).tobytes())
        else: parts.append(sc.tobytes() + q.astype(np.int8).tobytes())
    nbytes = len(zlib.compress(b"".join(parts), 9)) + 2 * len(items)
    out = {}
    for k, t, v in items:
        r = rec[k]
        if t == "aff":
            C = len(r) // 2; s = 1 + r[:C]; bn = k[:-6] + "bias"; bb = np.asarray(base[bn], np.float32); out[k] = np.asarray(base[k], np.float32) * s.reshape(-1, 1, 1, 1); out[bn] = s * bb + r[C:]
        else: out[k] = (np.asarray(base[k], np.float32) + r.reshape(np.asarray(base[k]).shape)).astype(np.float32)
    return nbytes, out


rows = [("원본(무손실)", None, mAP(ad))]
for bits in (8, 6, 4, 3):
    for keep in (1.0, 0.5, 0.25, 0.1):
        nb, a = build(bits, keep); rows.append((f"{bits}비트 keep={keep}", nb, mAP(a)))
print(f"\n환경 {ENV}\n{'설정':20s} {'바이트':>8s} {'mAP50':>7s}")
for n, b, m in rows: print(f"{n:20s} {str(b):>8s} {m:7.3f}")
json.dump(rows, open(OUT / f"sweep_{ENV}.json", "w"), ensure_ascii=False)

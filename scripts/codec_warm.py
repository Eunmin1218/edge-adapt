"""'워밍업 부분 어댑터' 오프라인 시험 (스트림 없음).
서버: 받은 프레임으로 엣지와 같은 통계 정렬 어댑터를 먼저 접어 넣은 Student 에서 출발해, 레이어 번호 >= K 인 합성곱의 채널별 (ds, t) 만 증류로 학습한다.
엣지: 같은 프레임으로 통계 정렬 어댑터를 직접 만들 수 있으므로(전송 0) 보낼 것은 K 이상 레이어의 잔차 (ds, t) 뿐이다.
비교: 전체 어댑터(현재, alpha=1 로 맞춰 비교) 대비 바이트, 복원 후 mAP50. 합성 데이터 측정이다."""
import sys, zlib, time, json, copy
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT
from src.toyworld import make_img
from src.server_core import AdapterServer, onnx_inits, load_adapter_cfg, save_pt, export_onnx
from src.adapter_cfg import server_kwargs
from src.onnx_detector import OnnxDetector
from src.stat_align import StatAligner
from src.mapeval import evaluate
import src.adapt_modules as AM
from src.adapt_modules import _targets
from src.distill import distill_train
from ultralytics import YOLO
from ultralytics.nn.modules import Conv

W = ROOT / "runs/toy3"; S_PT, T_PT = W / "student.pt", W / "teacher.pt"
ENV = sys.argv[1] if len(sys.argv) > 1 else "night"; KS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "10,16,23").split(",")]
CFG = load_adapter_cfg(W / "adapter.json"); CFG["alpha"] = 1.0; base = onnx_inits(W / "student.onnx")
rg = np.random.default_rng(5); sa = StatAligner(S_PT, [make_img(rg, "clear")[0] for _ in range(60)])
import os, cv2
NTR, QJ = int(os.environ.get("NTR", 8)), int(os.environ.get("QJ", 0))
rt = np.random.default_rng(777); train = [make_img(rt, ENV)[0] for _ in range(8)]; test = [make_img(rt, ENV) for _ in range(40)]
train = train[:NTR]; JB = 0
if QJ:
    _t = []
    for f in train:
        ok, b = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, QJ]); JB += len(b); _t.append(cv2.imdecode(b, cv2.IMREAD_COLOR))
    train = _t
print(f"학습 프레임 {len(train)}장, JPEG q{QJ or '-'} 업링크 {JB} B", flush=True)
srv = AdapterServer(S_PT, W / "student.onnx", T_PT, W / "server" / "warm", device="cpu", imgsz=640, **server_kwargs(CFG))


def mAP(ad):
    det = OnnxDetector(W / "student.onnx"); det.apply_adapter(ad) if ad else None; P = []
    for img, gt in test:
        d = det.infer(img); P.append((d["boxes"], d["scores"], d["cls"]))
    return evaluate(P, [g for _, g in test], 6)["map50"]


def layer_of(model):
    m = {}
    for i, mod in enumerate(model.model):
        for s in mod.modules(): m[id(s)] = i
    return m


def to_dict(model, tag):
    d = W / "server" / "warm" / tag; d.mkdir(parents=True, exist_ok=True); save_pt(model, d / "m.pt"); new = onnx_inits(export_onnx(d / "m.pt", d / "m.onnx"))
    return {k: v for k, v in new.items() if k in base and v.shape == base[k].shape and not np.array_equal(v, base[k])}


def q_bytes(vecs, bits):
    """텐서별 대칭 bits 비트 양자화 + zlib (codec_sweep 과 같은 방식). 반환 (바이트, 복원 벡터들)"""
    parts, rec = [], []
    for x in vecs:
        L = 2 ** (bits - 1) - 1; mx = float(np.abs(x).max()); sc = np.float16(mx / L if mx > 0 else 1.0); q = np.clip(np.rint(x / np.float32(sc)), -L, L).astype(np.int8)
        parts.append(sc.tobytes() + q.tobytes()); rec.append(q.astype(np.float32) * np.float32(sc))
    return len(zlib.compress(b"".join(parts), 9)) + 2 * len(vecs), rec


def run(K):
    """통계 정렬을 접은 Student 에서 출발, 레이어 >= K 만 학습."""
    y = YOLO(str(S_PT)); y.model.float().eval().fuse(verbose=False); model = y.model
    st, _ = sa.fit(train); lay = [(o, c) for o, c, a in _targets(model) if isinstance(o, Conv)]
    with torch.no_grad():
        for (o, c), (s, t) in zip(lay, st): c.weight.mul_(s.view(-1, 1, 1, 1)); c.bias.copy_(c.bias * s + t)
    ad_stat = to_dict(y, "stat")
    lm = layer_of(model); rec = {}
    orig_att, orig_fold = AM.attach_affine, AM.fold_affine

    def att(m):
        params, owners = [], []
        for owner, conv, act in _targets(m):
            if lm.get(id(owner), 0) < K: continue
            c = conv.out_channels; ds = torch.nn.Parameter(torch.zeros(c)); t = torch.nn.Parameter(torch.zeros(c)); owner._ds, owner._t = ds, t; params += [ds, t]; old = owner.forward

            def fwd(x, owner=owner, conv=conv, act=act, old=old):
                yv = (old(x) if owner is conv else conv(x)) * (1 + owner._ds).view(1, -1, 1, 1) + owner._t.view(1, -1, 1, 1)
                return act(yv) if act is not None else yv
            owner._old_forward = old; owner.forward = fwd; owners.append(owner)
        m._affine_owners = owners; return params

    def fold(m):
        for owner in m._affine_owners: rec[id(owner)] = (owner._ds.detach().numpy().copy(), owner._t.detach().numpy().copy())
        orig_fold(m)
    AM.attach_affine, AM.fold_affine = att, fold
    try:
        t0 = time.time(); teacher = copy.deepcopy(srv.teacher.model)
        distill_train(model, teacher, train, srv.first_train, epochs=srv.epochs, lr=srv.lr, lrf=srv.lrf, batch=srv.batch or max(1, -(-len(train) // 2)), mode="affine", tone=False, lr_aff=srv.lr_aff, aug=srv.aug)
        sec = time.time() - t0
    finally: AM.attach_affine, AM.fold_affine = orig_att, orig_fold
    ad = to_dict(y, f"k{K}"); vec = [np.r_[ds, t] for ds, t in rec.values()]; res = {}
    out = {"K": K, "sec": round(sec), "mAP_fp": mAP(ad), "n_ch": int(sum(len(v) // 2 for v in vec))}
    for b in (8, 4, 3):
        nb, rv = q_bytes(vec, b); out[f"bytes{b}"] = nb
        y2 = YOLO(str(S_PT)); y2.model.float().eval().fuse(verbose=False); m2 = y2.model; lay2 = [(o, c) for o, c, a in _targets(m2) if isinstance(o, Conv)]
        with torch.no_grad():
            for (o, c), (s, t) in zip(lay2, st): c.weight.mul_(s.view(-1, 1, 1, 1)); c.bias.copy_(c.bias * s + t)
            lm2 = layer_of(m2); j = 0
            for owner, conv, act in _targets(m2):
                if lm2.get(id(owner), 0) < K: continue
                v = rv[j]; j += 1; C = len(v) // 2; sq = 1 + torch.from_numpy(v[:C]); tq = torch.from_numpy(v[C:])
                conv.weight.mul_(sq.view(-1, 1, 1, 1)); conv.bias.copy_(conv.bias * sq + tq)
        out[f"mAP{b}"] = mAP(to_dict(y2, f"k{K}b{b}"))
    return out, ad_stat


rows = []; ad_stat = None
for K in KS:
    r, ad_stat = run(K); print(r, flush=True); rows.append(r)
print(f"\n환경 {ENV}: 기준 mAP 통계 정렬만(전송 0) {mAP(ad_stat):.3f}")
print("전체 채널 7898 기준 (현재 방식 8비트 약 14.8KB, 4비트 8.0KB, 3비트 5.5KB)")
for r in rows: print(f"K={r['K']:2d} 채널 {r['n_ch']:5d} ({r['n_ch'] / 7898:.0%})  mAP {r['mAP_fp']:.3f}  8비트 {r['bytes8']:6d}B/{r['mAP8']:.3f}  4비트 {r['bytes4']:6d}B/{r['mAP4']:.3f}  3비트 {r['bytes3']:6d}B/{r['mAP3']:.3f}  (학습 {r['sec']}s)")
(ROOT / "results" / "dev").mkdir(parents=True, exist_ok=True); json.dump(rows, open(ROOT / "results" / "dev" / f"warm_{ENV}_n{NTR}_q{QJ}.json", "w"))

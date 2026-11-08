"""서버 어댑터 다운링크 부호화 측정 (스트림 시뮬레이션 없음).
같은 8프레임으로 (1) 서버 증류 어댑터 (2) 엣지 통계 정렬 어댑터를 만든 뒤, 서버 어댑터를
  A. base 대비 FP16 (pack16, 현재)       B. base 대비 int8 (pack)
  C. 통계 정렬 어댑터 대비 잔차 FP16      D. 통계 정렬 어댑터 대비 잔차 int8
로 부호화했을 때 (바이트, 복원 후 mAP50) 을 잰다. 잔차 기준(통계 정렬 어댑터)은 엣지가 같은 8프레임으로 직접 만들 수 있으므로 전송이 없다.
합성 데이터 측정이다."""
import sys, json, zlib, struct, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT
from src.toyworld import make_img
from src.server_core import AdapterServer, onnx_inits, load_adapter_cfg
from src.adapter_cfg import server_kwargs
from src.onnx_detector import OnnxDetector
from src.stat_align import StatAligner
from src.delta_codec import encode, decode, _affine_fit, _q8t
from src.mapeval import evaluate

OUT = ROOT / "results" / "dev"; OUT.mkdir(parents=True, exist_ok=True); W = ROOT / "runs/toy3"; S_PT, T_PT = W / "student.pt", W / "teacher.pt"; IMGSZ = 640
ENV = sys.argv[1] if len(sys.argv) > 1 else "night"
CFG = load_adapter_cfg(W / "adapter.json"); base = onnx_inits(W / "student.onnx")
rg = np.random.default_rng(5); sa = StatAligner(S_PT, [make_img(rg, "clear")[0] for _ in range(60)])
rt = np.random.default_rng(777); train = [make_img(rt, ENV)[0] for _ in range(8)]; test = [make_img(rt, ENV) for _ in range(40)]
t0 = time.time(); srv = AdapterServer(S_PT, W / "student.onnx", T_PT, W / "server" / "probe", device="cpu", imgsz=IMGSZ, **server_kwargs(CFG)); ad_srv = srv(train, 0); print(f"서버 학습 {time.time() - t0:.0f}s", flush=True)
ad_st, _ = sa.adapter(train, base)
import pickle; pickle.dump(dict(ad_srv=ad_srv, ad_st=ad_st, test=test, train=train), open(OUT / f'ad_{ENV}.pkl', 'wb'))


def st_of(ad):
    """텐서별 (s, t): W_new = s*W_base, b_new = s*b_base + t. 채널별 상수배가 아닌 텐서는 other 로 분리."""
    out, other = {}, {}
    for k, v in ad.items():
        v = np.asarray(v, np.float32)
        if k.endswith(".weight") and v.ndim == 4 and k in base:
            s = _affine_fit(v, np.asarray(base[k], np.float32)); bn = k[:-6] + "bias"
            if s is not None and bn in base:
                bb = np.asarray(base[bn], np.float32); out[k] = (s, np.asarray(ad.get(bn, bb), np.float32) - s * bb); continue
        other[k] = v
    return out, other


s_srv, o_srv = st_of(ad_srv); s_st, o_st = st_of(ad_st)
import collections; _c = collections.Counter(); [ _c.update({("affine" if k in s_srv else "other:" + k.split(".")[-1] + str(np.asarray(v).ndim)): np.asarray(v).size}) for k, v in ad_srv.items()]; print("원소 수 구성:", dict(_c))
print(f"서버 어댑터: 텐서 {len(ad_srv)}  affine {len(s_srv)}  기타 {len(o_srv)} | 통계 정렬: affine {len(s_st)}  기타 {len(o_st)}")


def resid(k):
    if k in s_st:
        sr = s_srv[k][0] / s_st[k][0]; bn = k[:-6] + "bias"; bb = np.asarray(base[bn], np.float32); bstat = s_st[k][0] * bb + s_st[k][1]
        return sr - 1, (s_srv[k][1] + s_srv[k][0] * bb) - sr * bstat
    return s_srv[k][0] - 1, s_srv[k][1]


a = np.concatenate([np.r_[s_srv[k][0] - 1, s_srv[k][1]] for k in s_srv]); r = np.concatenate([np.r_[resid(k)[0], resid(k)[1]] for k in s_srv])
ra, rr = np.sqrt((a ** 2).mean()), np.sqrt((r ** 2).mean())
print(f"값의 크기 (RMS): base 대비 {ra:.4f}  | 통계 정렬 대비 잔차 {rr:.4f}  | 비율 {ra / max(rr, 1e-12):.2f}  (파라미터당 절약 비트 log2 = {np.log2(ra / max(rr, 1e-12)):.2f})")


def mAP(ad):
    det = OnnxDetector(W / "student.onnx")
    if ad: det.apply_adapter(ad)
    P = []
    for img, gt in test:
        d = det.infer(img); P.append((d["boxes"], d["scores"], d["cls"]))
    return evaluate(P, [g for _, g in test], 6)["map50"]


def rebuild(rec):
    ad = dict(o_srv)
    for k, (dsr, tres) in rec.items():
        sr = 1 + dsr; bn = k[:-6] + "bias"; bb = np.asarray(base[bn], np.float32)
        if k in s_st: s0, t0_ = s_st[k]; s = sr * s0; b = sr * (s0 * bb + t0_) + tres
        else: s = sr; b = bb * 0 + sr * bb + tres
        ad[k] = np.asarray(base[k], np.float32) * s.reshape(-1, 1, 1, 1); ad[bn] = b.astype(np.float32)
    return ad


def pack_res(q8):
    buf, rec = [], {}
    for k in s_srv:
        ds, tres = resid(k)
        if q8:
            qa, sa_, dsr = _q8t(ds); qt, st_, trr = _q8t(tres); buf.append(sa_.tobytes() + qa.tobytes() + st_.tobytes() + qt.tobytes()); rec[k] = (dsr, trr)
        else:
            d16, t16 = ds.astype(np.float16), tres.astype(np.float16); buf.append(d16.tobytes() + t16.tobytes()); rec[k] = (d16.astype(np.float32), t16.astype(np.float32))
    return 2 + len(zlib.compress(b"".join(buf), 9)) + 2 * len(s_srv), rec               # 텐서 번호(uint16) 포함


rows = [("기준: 적응 없음", 0, mAP(None)), ("서버 어댑터 원본(무손실)", 0, mAP(ad_srv)), ("통계 정렬 어댑터(전송 0)", 0, mAP(ad_st))]
for nm, mode in (("A. base 대비 FP16 (pack16)", "pack16"), ("B. base 대비 int8 (pack)", "pack")):
    raw = encode(ad_srv, base, mode); rows.append((nm, len(raw), mAP(decode(raw, base))))
for nm, q8 in (("C. 통계 정렬 대비 잔차 FP16", False), ("D. 통계 정렬 대비 잔차 int8", True)):
    nb, rec = pack_res(q8)
    if o_srv: nb += len(encode(o_srv, base, "pack16"))
    rows.append((nm, nb, mAP(rebuild(rec))))
print(f"\n환경 {ENV} (학습 8프레임, 평가 40프레임)\n{'방식':34s} {'바이트':>8s} {'mAP50':>7s}")
for nm, nb, m in rows: print(f"{nm:34s} {nb:>8d} {m:7.3f}")
json.dump(rows, open(OUT / f"codec_{ENV}.json", "w"), ensure_ascii=False)

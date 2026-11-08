"""어댑터 설정 튜닝 (개발용 스트림에서만 수행 — 평가 스트림은 사용하지 않아 누수가 없다).
각 '환경 변화 구간'(비/밤 등)에서: 앞쪽 pool 프레임에서 선별(hybrid, 예산 B)해 서버 학습 → 구간 뒤쪽 보류(held-out) 프레임의 mAP50 이 적응 전 대비 얼마나 오르는지 측정한다.
설정 후보: 학습 epoch, 학습률, 헤드 앞에서 추가로 학습할 레이어 수(train_layers), tau. 변화구간 평균 이득이 가장 큰 설정을 configs/adapter.json 에 저장하고,
run_experiment.py / serve.py 가 인자를 따로 주지 않으면 이 파일을 읽는다 (B, ES, D 모두 같은 설정을 쓰므로 비교가 공정하다).
  python scripts/tune_adapter.py            (전체 격자)      python scripts/tune_adapter.py --quick   (작은 격자)"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT
from src.stream import load_stream
from src.onnx_detector import OnnxDetector
from src.selectors import STRATEGIES, frame_uncertainty, frame_embedding, frame_entropy
from src.mapeval import evaluate
from src.server_core import AdapterServer, onnx_inits
from src.adapter_cfg import load_adapter_cfg, server_kwargs
from src.delta_codec import encode, decode, scale_adapter

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--budget", type=int, default=16); ap.add_argument("--device", default="auto")
ap.add_argument("--stream-dir", default="data/yolo/stream_train"); ap.add_argument("--streams-json", default="configs/streams.json"); ap.add_argument("--split", default="detector_train")
ap.add_argument("--student-pt", default="models/student.pt"); ap.add_argument("--student-onnx", default="models/student.onnx"); ap.add_argument("--teacher-pt", default="models/teacher.pt")
ap.add_argument("--out", default="configs/adapter.json"); ap.add_argument("--pool", type=int, default=100); ap.add_argument("--held", type=int, default=150); ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--batch", type=int, default=0, help="0 = 자동(증류: 프레임 수의 절반, ultra: 8)")
ap.add_argument("--trainer", default="distill", choices=["distill", "ultra"]); ap.add_argument("--modes", default=None, help="적응 방식 후보(쉼표): affine(모든 합성곱 채널별 배율·이동) | head(Detect 헤드) | affine+head. 기본 affine,head"); ap.add_argument("--lras", default=None, help="affine 학습률 후보"); ap.add_argument("--alphas", default="1.0,1.25", help="가중치 보간 계수 후보(학습 없이 보간만 바꿔 평가)"); ap.add_argument("--tone", type=int, default=0, help="1 이면 입력 톤 커브도 학습"); ap.add_argument("--sel-seeds", type=int, default=2, help="1단계에서 무작위성이 있는 선별 방식을 평가할 시드 수(1회 결과는 편차가 커서 선택이 흔들린다)"); ap.add_argument("--selectors", default="diversity,ent_hybrid,hybrid,random", help="프레임 선별 후보(쉼표). 1단계에서 기본 설정으로 비교해 하나를 고른다"); ap.add_argument("--epochs", default=None, help="쉼표로 구분한 후보, 예: 5,10,20"); ap.add_argument("--lrs", default=None); ap.add_argument("--layers", default=None); ap.add_argument("--taus", default=None); a = ap.parse_args()
dev = int(a.device) if a.device.isdigit() else a.device
EP = [int(x) for x in a.epochs.split(",")] if a.epochs else ([60] if a.quick else [30, 60]); LR = [float(x) for x in a.lrs.split(",")] if a.lrs else ([1e-2] if a.quick else [3e-3, 5e-3, 1e-2])
MODES = a.modes.split(",") if a.modes else (["affine"] if a.quick else ["affine", "head"]); LRA = [float(x) for x in a.lras.split(",")] if a.lras else ([0.03] if a.quick else [0.01, 0.03]); ALPHAS = [float(x) for x in a.alphas.split(",")]
LY = [int(x) for x in a.layers.split(",")] if a.layers else [0]; TAU = [float(x) for x in a.taus.split(",")] if a.taus else [0.5]

segs = json.load(open(ROOT / a.streams_json))[a.split]; stream = load_stream(ROOT / a.stream_dir, segs, multiple=50); env = [it["env"] for it in stream]
runs, s = [], 0
for i in range(1, len(env) + 1):
    if i == len(env) or env[i] != env[s]: runs.append((env[s], s, i)); s = i
shift = [(e, s, t) for e, s, t in runs if e != "clear"]
if not shift: sys.exit("개발용 스트림에 환경 변화 구간이 없습니다.")
base = OnnxDetector(ROOT / a.student_onnx); SELS = a.selectors.split(","); data = []
for e, s0, t0 in shift:
    pool_n = min(a.pool, (t0 - s0) // 2); pool = [stream[i]["img"] for i in range(s0, s0 + pool_n)]; held = list(range(s0 + pool_n, min(t0, s0 + pool_n + a.held)))
    dets = [base.infer(im) for im in pool]; cands = {"uncert": np.array([frame_uncertainty(d) for d in dets]), "emb": np.stack([frame_embedding(im) for im in pool]), "ent": np.array([frame_entropy(d) for d in dets]), "age": np.arange(len(pool))[::-1].astype(float)}
    picks = {(sn, sd): STRATEGIES[sn](cands, a.budget, np.random.default_rng(sd)) for sn in SELS for sd in range(a.sel_seeds)}
    imgs = [stream[i]["img"] for i in held]; gts = [stream[i]["gt"] for i in held]; pb = [(d["boxes"], d["scores"], d["cls"]) for d in map(base.infer, imgs)]
    data.append({"env": e, "pool": pool, "picks": picks, "imgs": imgs, "gts": gts, "base": evaluate(pb, gts, 6, iou_thrs=[0.5])["map50"]}); print(f"구간 {e}: pool {pool_n} → 선별 {a.budget}, 보류 {len(held)}프레임, 적응 전 mAP50 {data[-1]['base']:.3f}", flush=True)
def run_cfg(srv, sn, sd=0):
    gains, secs, mbs = [], [], []
    for k, d in enumerate(data):
        t0 = time.time(); ad = srv([d["pool"][i] for i in d["picks"][(sn, sd)]], k); det = OnnxDetector(ROOT / a.student_onnx); det.apply_adapter(ad)
        pr = [(o["boxes"], o["scores"], o["cls"]) for o in map(det.infer, d["imgs"])]; gains.append(evaluate(pr, d["gts"], 6, iou_thrs=[0.5])["map50"] - d["base"]); secs.append(time.time() - t0); mbs.append(sum(v.size for v in ad.values()) * 2 / 2**20)
    return gains, secs, mbs


# 1단계: 기본 학습 설정에서 프레임 선별 방식 비교 (같은 구간·같은 예산·같은 보류 프레임)
sel_rows = {}
if len(SELS) > 1:
    for sn in SELS:
        srv = AdapterServer(ROOT / a.student_pt, ROOT / a.student_onnx, ROOT / a.teacher_pt, ROOT / "runs/tune_work" / f"sel_{sn}", device=dev, imgsz=a.imgsz, **{**server_kwargs(load_adapter_cfg()), "trainer": a.trainer, "batch": a.batch})
        g = [x for sd in range(a.sel_seeds) for x in run_cfg(srv, sn, sd)[0]]; sel_rows[sn] = float(np.mean(g)); print(f"선별 {sn:11s}: 평균 이득 {sel_rows[sn]:+.3f} ({len(g)}회: {np.round(g, 3).tolist()})", flush=True)
BEST_SEL = max(sel_rows, key=sel_rows.get) if sel_rows else SELS[0]; print(f"→ 선별 방식 선택: {BEST_SEL}", flush=True)
B0 = onnx_inits(ROOT / a.student_onnx)


def run_grid_cfg(srv, sn, alphas):
    """학습 1회(알파 1.0) 후 알파별로 보류 프레임 이득을 잰다. 전송 크기는 실제 부호화('auto') 바이트."""
    g = {al: [] for al in alphas}; secs, mbs = [], []
    for k, d in enumerate(data):
        t0 = time.time(); ad = srv([d["pool"][i] for i in d["picks"][(sn, 0)]], k); secs.append(time.time() - t0)
        for al in alphas:
            ada = ad if al == 1.0 else scale_adapter(ad, B0, al); raw = encode(ada, B0, "auto")
            if al == alphas[0]: mbs.append(len(raw) / 2**20)
            det = OnnxDetector(ROOT / a.student_onnx); det.apply_adapter(decode(raw, B0)); pr = [(o["boxes"], o["scores"], o["cls"]) for o in map(det.infer, d["imgs"])]
            g[al].append(evaluate(pr, d["gts"], 6, iou_thrs=[0.5])["map50"] - d["base"])
    return g, secs, mbs


rows = []; base_kw = {**server_kwargs(load_adapter_cfg()), "trainer": a.trainer, "batch": a.batch, "tone": bool(a.tone)}
cfgs = []
for mode in MODES:
    for ep, tau in itertools.product(EP, TAU):
        if mode == "head": cfgs += [dict(mode=mode, epochs=ep, tau=tau, lr=lr, train_layers=ly, lr_aff=0.0) for lr in LR for ly in LY]
        elif mode == "affine": cfgs += [dict(mode=mode, epochs=ep, tau=tau, lr=0.0, train_layers=0, lr_aff=la) for la in LRA]
        else: cfgs += [dict(mode=mode, epochs=ep, tau=tau, lr=lr, train_layers=0, lr_aff=la) for lr in LR for la in LRA]
for c in cfgs:
    kw = {**base_kw, **c, "alpha": 1.0}; srv = AdapterServer(ROOT / a.student_pt, ROOT / a.student_onnx, ROOT / a.teacher_pt, ROOT / "runs/tune_work" / f"{c['mode']}_e{c['epochs']}_lr{c['lr']}_la{c['lr_aff']}_l{c['train_layers']}_t{c['tau']}", device=dev, imgsz=a.imgsz, **kw)
    g, secs, mbs = run_grid_cfg(srv, BEST_SEL, ALPHAS)
    for al in ALPHAS:
        rows.append({**c, "alpha": al, "tone": int(a.tone), "gain": float(np.mean(g[al])), "gains": [float(x) for x in g[al]], "sec": float(np.mean(secs)), "adapter_MB": float(np.mean(mbs))})
        print(f"mode={c['mode']} epochs={c['epochs']} lr={c['lr']:g} lr_aff={c['lr_aff']:g} layers={c['train_layers']} alpha={al:g}: 평균 이득 {rows[-1]['gain']:+.3f} (구간별 {np.round(g[al], 3).tolist()}) | 적응 1회 {rows[-1]['sec']:.0f}s | 어댑터 {rows[-1]['adapter_MB'] * 1024:.0f}KB", flush=True)
ok = [r for r in rows if r["adapter_MB"] <= 5.0] or rows       # 어댑터 5MB 목표를 넘는 설정은 제외 (모두 넘으면 전체에서 선택)
best = max(ok, key=lambda r: (r["gain"], -r["sec"])); out = ROOT / a.out; out.parent.mkdir(exist_ok=True)
json.dump({"epochs": best["epochs"], "lr": best["lr"] or 1e-2, "train_layers": best["train_layers"], "tau": best["tau"], "batch": a.batch, "mode": best["mode"], "alpha": best["alpha"], "lr_aff": best["lr_aff"] or 0.03, "tone": best["tone"], "aug": 0, "trainer": a.trainer, "selector": BEST_SEL, "selector_gains": sel_rows, "selected_on": f"{a.split} 보류 프레임 mAP50", "mean_gain": best["gain"], "all": rows}, open(out, "w"), indent=1)
print(f"\n선택: mode={best['mode']} epochs={best['epochs']} lr={best['lr']:g} lr_aff={best['lr_aff']:g} alpha={best['alpha']:g} train_layers={best['train_layers']} tau={best['tau']} selector={BEST_SEL} (이득 {best['gain']:+.3f}, 어댑터 {best['adapter_MB'] * 1024:.0f}KB) → {out}")
if best["gain"] <= 0.01: print("경고: 어느 설정도 보류 프레임 mAP 를 올리지 못했습니다. 적응 자체가 효과가 없는 상태이므로 본 실험 전에 원인을 해결하세요(GUIDE 6.7).")

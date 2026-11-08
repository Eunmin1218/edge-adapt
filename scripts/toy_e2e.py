"""CPU 통합 시험 (SHIFT·GPU 불필요). 합성 영상으로 '학습 → ONNX → 어댑터 → 4개 시스템 비교'를 끝까지 실제로 실행해
  파이프라인에 코드 오류가 없는지 확인한다. 성능을 측정하는 시험이 아니다 (합성 데이터이므로 수치를 보고서에 쓰지 않는다).
사용: python scripts/toy_e2e.py            (처음 실행: 모델 학습 포함. 결과는 runs/toy/ 에 캐시되어 다시 실행하면 학습을 건너뜀)
      python scripts/toy_e2e.py --quick    (학습 해상도 320, 훨씬 빠름. 동작 확인용)
합성 세계: 회색 배경 위의 색깔 사각형 6종(클래스 = 색). '밤' = 밝기 x0.22 + 파란 기운 + 노이즈 → 맑음으로만 학습한 Student 의 성능이 떨어진다."""
import argparse, os, json, pickle, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, CLASSES, IMG_W, IMG_H

ap = argparse.ArgumentParser(allow_abbrev=False); ap.add_argument("--quick", action="store_true"); ap.add_argument("--imgsz", type=int, default=None); ap.add_argument("--epochs-student", type=int, default=None)
ap.add_argument("--epochs-teacher", type=int, default=None); ap.add_argument("--adapt-epochs", type=int, default=None, help="기본: runs/toy/adapter.json(--only-tune 결과), 없으면 server_core 기본값"); ap.add_argument("--delay", type=int, default=30); ap.add_argument("--budget", type=int, default=8)
ap.add_argument("--v7", action="store_true", help="V7: T0(입력 정합, 휘도 CUSUM 켬) + 라벨 없는 검증 (+서버 gate)"); ap.add_argument("--world", default="toy", choices=["toy", "toy2", "toy3"], help="toy2: 환경 5종(scripts/toy_hard.py 로 먼저 준비)"); ap.add_argument("--verify", action="store_true", help="엣지 후보를 Teacher 프로브로 검증하는 변형"); ap.add_argument("--v2", action="store_true"); ap.add_argument("--v3", action="store_true"); ap.add_argument("--v4", action="store_true"); ap.add_argument("--v5", action="store_true"); ap.add_argument("--evorder", action="store_true", help="평가·개발 스트림의 환경 순서를 무작위로(시드별)"); ap.add_argument("--rseed", type=int, default=0); ap.add_argument("--es", default="", help="ES 변형 \"theta,probe_n,cooldown,reg\" 세미콜론으로 여러 개"); ap.add_argument("--vfinal", action="store_true"); ap.add_argument("--edge-delay", type=int, default=0); ap.add_argument("--v6", action="store_true"); ap.add_argument("--es-probe", type=int, default=4); ap.add_argument("--min-posts", default="30"); ap.add_argument("--agrees", default="0.85"); ap.add_argument("--evseed", type=int, default=22); ap.add_argument("--sens", action="store_true", help="엣지 추적의 임계값·정렬강도 민감도 (서버 없음)"); ap.add_argument("--methods", default="", help="쉼표로 구분한 이름 조각이 포함된 방법만 실행 (예: D)"); ap.add_argument("--only-tune", action="store_true", help="비교 대신 어댑터 설정 튜닝(tune_adapter.py)만 실행"); ap.add_argument("--train-layers", type=int, default=0, help="헤드 앞에서 추가로 학습할 레이어 수 (0 = 헤드만)"); a, REST = ap.parse_known_args()   # 모르는 인자는 --only-tune 일 때 tune_adapter.py 로 그대로 전달
IMGSZ = a.imgsz or (320 if a.quick else 640); ES_ = a.epochs_student or (6 if a.quick else 12); ET_ = a.epochs_teacher or (6 if a.quick else 12)
W = ROOT / {"toy": "runs/toy", "toy2": "runs/toy2", "toy3": "runs/toy3"}[a.world]; W.mkdir(parents=True, exist_ok=True)
COL = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 220, 220), (220, 0, 220), (255, 160, 0)]      # BGR, 클래스별 색


def make_img(rng, env):
    base = np.full((IMG_H, IMG_W, 3), rng.integers(100, 140), np.float32) + np.linspace(-15, 15, IMG_W)[None, :, None]
    boxes, cls = [], []
    for _ in range(rng.integers(3, 7)):
        c = int(rng.integers(0, 6)); w, h = int(rng.integers(40, 130)), int(rng.integers(40, 130)); x, y = int(rng.integers(0, IMG_W - w)), int(rng.integers(0, IMG_H - h))
        if any(not (x + w < b[0] or b[2] < x or y + h < b[1] or b[3] < y) for b in boxes): continue
        cv2.rectangle(base, (x, y), (x + w, y + h), COL[c], -1); cv2.rectangle(base, (x, y), (x + w, y + h), (30, 30, 30), 2); boxes.append([x, y, x + w, y + h]); cls.append(c)
    if env == "night": base = base * 0.22 + np.array([14, 6, 0]) + rng.normal(0, 10, base.shape)
    else: base = base + rng.normal(0, 3, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8), (np.array(boxes, float).reshape(-1, 4), np.array(cls, int))


def write_set(name, n, envs, seed):
    d = W / "data" / name; (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(exist_ok=True); rng = np.random.default_rng(seed)
    for i in range(n):
        img, (b, c) = make_img(rng, envs[i % len(envs)]); cv2.imwrite(str(d / "images" / f"{i:04d}.jpg"), img)
        (d / "labels" / f"{i:04d}.txt").write_text("\n".join(f"{k} {(x1 + x2) / 2 / IMG_W:.6f} {(y1 + y2) / 2 / IMG_H:.6f} {(x2 - x1) / IMG_W:.6f} {(y2 - y1) / IMG_H:.6f}" for k, (x1, y1, x2, y2) in zip(c, b)))


def stream(seed, segs):
    rng = np.random.default_rng(seed); out = []
    for env, n in segs:
        for _ in range(n):
            img, gt = make_img(rng, env); out.append({"img": img, "gt": gt, "env": env})
    return out


def yaml(name, tr, va):
    p = W / f"{name}.yaml"; p.write_text(f"path: {(W / 'data').resolve()}\ntrain: {tr}/images\nval: {va}/images\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))); return p


def timeit(msg, f):
    t0 = time.time(); r = f(); print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True); return r


from ultralytics import YOLO
if not (W / "data/student_train").exists():
    timeit("합성 데이터 생성", lambda: [write_set("student_train", 80, ["clear"], 1), write_set("student_val", 16, ["clear"], 2), write_set("teacher_train", 96, ["clear", "night"], 3), write_set("teacher_val", 20, ["clear", "night"], 4)])
for role, ep, tr, va in (("student", ES_, "student_train", "student_val"), ("teacher", ET_, "teacher_train", "teacher_val")):
    best = W / role / "weights/last.pt"
    if best.exists(): print(f"{role}: 학습 결과 캐시 사용"); continue
    timeit(f"{role} 학습 (epochs={ep}, imgsz={IMGSZ})", lambda: YOLO("yolo11n.pt").train(data=str(yaml(role, tr, va)), epochs=ep, imgsz=IMGSZ, batch=16, device="cpu", workers=0, plots=False, project=str(W), name=role, exist_ok=True, verbose=False, amp=False, seed=0, mosaic=0.0, optimizer="AdamW", lr0=0.004, warmup_epochs=1, val=False))
from src.server_core import export_onnx, AdapterServer
if a.world in ("toy2", "toy3"): from src.toyworld import make_img, stream
from src.onnx_detector import OnnxDetector
from src.sim import run_stream, Policy
from src.signals import window_features, ALL_COLS
from src.drift_model import make_labels, fit_binary
from src.mapeval import evaluate
S_PT, T_PT = W / "student.pt", W / "teacher.pt"
import shutil; shutil.copy(W / "student/weights/last.pt", S_PT); shutil.copy(W / "teacher/weights/last.pt", T_PT)
timeit("Student ONNX 내보내기", lambda: export_onnx(S_PT, W / "student.onnx")); timeit("Teacher ONNX 내보내기", lambda: export_onnx(T_PT, W / "teacher.onnx"))
print(subprocess.run([sys.executable, str(ROOT / "scripts/check_adapter_pipeline.py"), str(S_PT)], capture_output=True, text=True, cwd=ROOT).stdout[-900:])

SEGS = [("clear", 100), ("night", 200), ("clear", 100)]
if a.world == "toy3": SEGS = [("clear", 100), ("night", 150), ("clear", 50), ("glare", 150), ("clear", 50), ("heavy_rain", 100), ("clear_sparse", 100), ("night", 100), ("clear_dense", 100), ("clear", 100)]   # 국소 반사·과밀 빗줄기·물체 수 변화(영상은 그대로)
if a.world == "toy2": SEGS = [("clear", 100), ("night", 150), ("clear", 50), ("night_rain", 150), ("clear", 50), ("fog", 100), ("night", 100), ("clear", 100)]   # 밤 재등장(레지스트리), 밤+비(처음 보는 조합), 안개(영향 작음)
if a.only_tune:                                                    # 개발용 스트림을 YOLO 형식(meta.csv 포함)으로 써서 tune_adapter.py 를 실제 경로 그대로 실행
    d = W / "data/stream_tune"; (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(exist_ok=True); rows, segs, rng = ["seq,frame,name"], [], np.random.default_rng(33)
    for env_, n_ in [("clear", 100), ("night", 250), ("clear", 100)]:
        seq = f"{env_}{len(segs)}"; segs.append([seq, env_, n_])
        for f in range(n_):
            img, (b, c) = make_img(rng, env_); nm = f"{seq}_{f:04d}"; cv2.imwrite(str(d / "images" / f"{nm}.jpg"), img); rows.append(f"{seq},{f},{nm}")
            (d / "labels" / f"{nm}.txt").write_text("\n".join(f"{k} {(x1 + x2) / 2 / IMG_W:.6f} {(y1 + y2) / 2 / IMG_H:.6f} {(x2 - x1) / IMG_W:.6f} {(y2 - y1) / IMG_H:.6f}" for k, (x1, y1, x2, y2) in zip(c, b)))
    (d / "meta.csv").write_text("\n".join(rows)); (W / "streams.json").write_text(json.dumps({"detector_train": segs}))
    sys.exit(subprocess.run([sys.executable, str(ROOT / "scripts/tune_adapter.py"), "--stream-dir", str(d), "--streams-json", str(W / "streams.json"), "--student-pt", str(S_PT), "--student-onnx", str(W / "student.onnx"),
                             "--teacher-pt", str(T_PT), "--out", str(W / "adapter.json"), "--device", "cpu", "--budget", str(a.budget), "--pool", "100", "--held", "100"] + REST, cwd=ROOT).returncode)
# 1) 감지기 학습: 별도 시드 스트림에서 A 를 돌려 윈도우 특징·mAP 로 라벨 생성
def make_segs(seed, envs=("clear", "night", "glare", "heavy_rain", "clear_sparse", "clear_dense")):
    """무작위 환경 순서·구간 길이: 첫 구간은 맑음(기준), 모든 환경이 최소 한 번 나오고 재등장이 있으며 같은 환경이 연속하지 않는다."""
    r = np.random.default_rng(seed); order = ["clear"] + list(r.permutation(envs[1:]))
    while len(order) < 10:
        c = str(r.choice(envs));
        if c != order[-1]: order.append(c)
    return [(e, int(r.choice([50, 100, 150]))) for e in order]


TR_SEGS = make_segs(7) if a.evorder else SEGS            # 개발 스트림(드리프트 감지기 학습): 평가와 다른 순서
if a.evorder: SEGS = make_segs(a.evseed)
tr = stream(11, TR_SEGS); rA = timeit("A(detector_train)", lambda: run_stream(tr, OnnxDetector(W / "student.onnx"), Policy("none"), None, use_registry=False))
wm, env = rA["win_map"], np.array(rA["win_env"]); ref = float(np.nanmean(wm[env == "clear"])); print(f"학습 스트림 윈도우 mAP50: {np.round(wm, 2).tolist()}  환경: {env.tolist()}")
y = make_labels(wm, ref, 0.10); assert (y == 1).any() and (y == 0).any(), "합성 '밤'에서 Student 성능이 떨어지지 않았다 → Student 가 너무 약하거나(학습 부족) 이미 밤에 강함"
clf = fit_binary(np.stack([window_features(f, ALL_COLS) for f in rA["win_frame_feats"]]), y)
# 2) 평가 스트림: A / T / B / ES / D
ev = stream(a.evseed, SEGS); res = {}
res["A"] = timeit("A", lambda: run_stream(ev, OnnxDetector(W / "student.onnx"), Policy("none"), None, use_registry=False))
res["T(Teacher)"] = timeit("T", lambda: run_stream(ev, OnnxDetector(W / "teacher.onnx"), Policy("none"), None, use_registry=False))


from src.server_core import load_adapter_cfg
from src.adapter_cfg import server_kwargs
CFG = load_adapter_cfg(W / "adapter.json"); CFG["epochs"] = a.adapt_epochs or CFG["epochs"]; CFG["train_layers"] = a.train_layers or CFG["train_layers"]; print("서버 적응 설정:", CFG)


def mk_server(tag, **ov): return AdapterServer(S_PT, W / "student.onnx", T_PT, W / "server" / tag, device="cpu", imgsz=IMGSZ, **{**server_kwargs(CFG), **ov})


_SA = {}
def edge_fn_for(lam):
    from src.stat_align import StatAligner
    from src.server_core import onnx_inits
    if "sa" not in _SA:
        rg = np.random.default_rng(5); _SA["sa"] = StatAligner(S_PT, [make_img(rg, "clear")[0] for _ in range(60)]); _SA["B"] = onnx_inits(W / "student.onnx")
    return lambda fr: _SA["sa"].adapter(fr, _SA["B"], lam)[0]


def go(name, policy, sel, reg, probe=False, edge=None, noserver=False, track=False, fast=0, sthr=0.25, fmar=0.1, lam=1.0, verify=False, gate=0.08, agree=None, confirm=False, min_post=0, rk=None, **ov):
    if a.methods and not any(m in name for m in a.methods.split(",")): return
    warm = ov.pop("warm", None); srv = None if noserver else mk_server(name, **ov)
    if warm and srv is not None:
        from src.warm_server import WarmServer
        edge_fn_for(1.0); srv = WarmServer(srv, _SA["sa"], K=warm[0], bits=warm[1], alpha=(warm[2] if len(warm) > 2 else 1.0))
    kw = dict(selector=sel, budget=a.budget, apply_delay=a.delay, use_registry=reg, probe_fn=srv.agreement if (probe and srv) else None)
    if edge is not None: kw["edge_fn"] = edge_fn_for(edge if lam == 1.0 else lam)
    if track: kw["edge_sig"] = lambda fr: _SA["sa"].signature(fr); kw["env_thr"] = sthr; kw["sig_thr"] = sthr; kw["fast_margin"] = fmar; kw["fast_n"] = fast; kw["verify_edge"] = verify; kw["agree_thr"] = agree; kw["confirm"] = confirm; kw["min_post"] = min_post; kw["gate_margin"] = gate
    kw.update(rk or {}); kw["seed"] = a.rseed
    res[name] = timeit(name, lambda: run_stream(ev, OnnxDetector(W / "student.onnx"), policy, srv, **kw)); res[name]["server"] = srv.last_info if srv else None


go("B(periodic)", Policy("periodic", period=100), "random", False)
go("ES(EdgeSync-lite)", Policy("edgesync"), "edgesync", False, probe=True)
go("D(ours)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True)
go("D(헤드 증류, 이전 구성)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, mode="head", alpha=1.0, epochs=30, aug=1, tone=False, lr=0.01)
if a.world in ("toy2", "toy3"):
    go("D-stat(엣지 통계정렬만, 레지스트리)", Policy("drift", drift_model=clf, consec=1, cooldown=100), "diversity", True, edge=1.0, noserver=True)
    go("D-stat(엣지 통계정렬만, 레지스트리 없음)", Policy("drift", drift_model=clf, consec=1, cooldown=100), "diversity", False, edge=1.0, noserver=True)
    go("D+(엣지 통계정렬 후 서버 정련)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, edge=1.0)
if a.world in ("toy2", "toy3"):
    go("D★(엣지 추적, 서버 정련은 신뢰도 부족 때만)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, edge=1.0, track=True)
    go("D★F(빠른 점검 10프레임, 서버 정련 gate)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, edge=1.0, track=True, fast=10)
    go("D★FP(+ 프로브로 후보 검증)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, edge=1.0, track=True, fast=10, probe=True)
    go("D★F(빠른 점검, 서버 없음)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, edge=1.0, track=True, noserver=True, fast=10)
    go("D★(엣지 추적만, 서버 없음)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, edge=1.0, track=True, noserver=True)
if a.verify:
    go("V(빠른 점검 + 프로브 검증, 서버 정련 없음)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, edge=1.0, track=True, fast=10, probe=True, verify=True, gate=9.0)
    go("V+(빠른 점검 + 프로브 검증 + 서버 정련 gate)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, edge=1.0, track=True, fast=10, probe=True, verify=True)
if a.v2:
    for ag in [float(x) for x in a.agrees.split(",")]:
        go(f"V2(프로브 검증 + 일치도<{ag} 이면 서버 정련)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, edge=1.0, track=True, fast=10, probe=True, verify=True, gate=9.0, agree=ag)
if a.v5:
    P_ = lambda: Policy("drift", drift_model=clf, consec=1, cooldown=100)
    base_ = dict(edge=1.0, track=True, fast=10, probe=True, verify=True, gate=9.0, agree=0.85, min_post=30)
    go("V4b(V4 + 롤백 오류 수정)", P_(), CFG["selector"], True, **base_)
    go("V5(V4b + 프로브 q40 12장 + 어댑터 pack16)", P_(), CFG["selector"], True, rk=dict(probe_n=12, probe_q=40, codec="pack16"), **base_)
    go("V5i(V5 + 어댑터 int8)", P_(), CFG["selector"], True, rk=dict(probe_n=12, probe_q=40, codec="pack"), **base_)
    go("ESq(ES + 프로브 q40 + 어댑터 pack16)", Policy("edgesync", probe_n=a.es_probe), "edgesync", False, probe=True, rk=dict(probe_q=40, codec="pack16"))
for cfg_ in [c for c in a.es.split(";") if c]:
    th_, pn_, cd_, rg_ = cfg_.split(","); th_, pn_, cd_, rg_ = float(th_), int(pn_), int(cd_), int(rg_)
    go(f"ESf[th={th_},pn={pn_},cd={cd_},reg={rg_}]", Policy("edgesync", theta=th_, probe_n=pn_, cooldown=cd_), "edgesync", bool(rg_), probe=True, rk=dict(probe_q=40, codec="pack16"))
    if os.environ.get("ES_Q4"): go(f"ESq4[th={th_},pn={pn_},cd={cd_},reg={rg_}]", Policy("edgesync", theta=th_, probe_n=pn_, cooldown=cd_), "edgesync", bool(rg_), probe=True, rk=dict(probe_q=40, codec="pack4"))
def t0_params():
    rg = np.random.default_rng(5); fr = [make_img(rg, "clear")[0] for _ in range(60)]; lum = np.array([f.mean() for f in fr])
    return dict(lum_mu=float(lum.mean()), lum_sd=float(lum.std() + 1e-6), mu=np.mean([f.reshape(-1, 3).mean(0) for f in fr], 0).astype(np.float32), sd=np.mean([f.reshape(-1, 3).std(0) for f in fr], 0).astype(np.float32), k=2.0, h=6.0)
if a.v7:
    P7 = lambda: (Policy("none") if os.environ.get("P7_NONE") else Policy("drift", drift_model=clf, consec=1, cooldown=100)); b7 = dict(edge=1.0, track=True, fast=10, verify=True, agree=None, min_post=30)
    go("V7a(T0+라벨없는검증, 서버 없음)", P7(), CFG["selector"], True, noserver=True, gate=9.0, rk=dict(t0=t0_params(), lf_verify=True, probe_n=6, edge_delay=a.edge_delay), **b7)
    go("V7b(V7a + 서버 gate)", P7(), CFG["selector"], True, gate=0.12, rk=dict(t0=t0_params(), lf_verify=True, probe_n=6, probe_q=40, codec="pack16", heldout=True, gate_since=True, edge_delay=a.edge_delay), **b7)
    go("V7d(V7a + 일관성 기반 서버 정련)", P7(), CFG["selector"], True, gate=9.0, agree=0.85, min_post=30, edge=1.0, track=True, fast=10, verify=True, rk=dict(t0=t0_params(), lf_verify=True, fc_margin=0.04, probe_n=6, probe_q=40, codec="pack16", heldout=True, edge_delay=a.edge_delay))
    go("V7q(V7 일관성 기반 서버 정련 + 어댑터 4비트)", P7(), CFG["selector"], True, gate=9.0, agree=0.85, min_post=30, edge=1.0, track=True, fast=10, verify=True, rk=dict(t0=t0_params(), lf_verify=True, fc_margin=0.04, probe_n=6, probe_q=40, codec="pack4", heldout=True, edge_delay=a.edge_delay))
    go("V8(헤드만 4비트 워밍업 서버, V7 기반)", P7(), CFG["selector"], True, gate=9.0, agree=0.85, min_post=30, edge=1.0, track=True, fast=10, verify=True, rk=dict(t0=t0_params(), lf_verify=True, fc_margin=0.04, probe_n=6, probe_q=40, codec="pack16", heldout=True, edge_delay=a.edge_delay), warm=(23, 4))
    go("V8a(헤드만 4비트 워밍업 서버 α=1.25, V7 기반)", P7(), CFG["selector"], True, gate=9.0, agree=0.85, min_post=30, edge=1.0, track=True, fast=10, verify=True, rk=dict(t0=t0_params(), lf_verify=True, fc_margin=0.04, probe_n=6, probe_q=40, codec="pack16", heldout=True, edge_delay=a.edge_delay), warm=(23, 4, 1.25))
    go("V8b(헤드만 4비트 워밍업 서버, 거절은 두 지표가 모두 나쁠 때만)", P7(), CFG["selector"], True, gate=9.0, agree=0.85, min_post=30, edge=1.0, track=True, fast=10, verify=True, rk=dict(t0=t0_params(), lf_verify=True, fc_margin=0.04, probe_n=6, probe_q=40, codec="pack16", heldout=True, edge_delay=a.edge_delay, lf_or=True), warm=(23, 4, 1.0))
    go("V9(V8 + 새 서명마다 서버 1회, 4장 q60)", P7(), CFG["selector"], True, gate=9.0, agree=0.85, min_post=30, edge=1.0, track=True, fast=10, verify=True, rk=dict(t0=t0_params(), lf_verify=True, fc_margin=0.04, probe_n=6, probe_q=40, codec="pack16", heldout=True, edge_delay=a.edge_delay, srv_all=True, train_q=60), warm=(23, 4))
    go("V7e(V7d + 서버 어댑터 무검증 수락)", P7(), CFG["selector"], True, gate=9.0, agree=0.85, min_post=30, edge=1.0, track=True, fast=10, verify=True, rk=dict(t0=t0_params(), lf_verify=True, fc_margin=0.04, probe_n=6, probe_q=40, codec="pack16", heldout=True, edge_delay=a.edge_delay, srv_trust=True))
    go("V7s(V7q + 서버 쪽 Teacher 일치도 검증)", P7(), CFG["selector"], True, gate=9.0, agree=0.85, min_post=30, edge=1.0, track=True, fast=10, verify=True, rk=dict(t0=t0_params(), lf_verify=True, fc_margin=0.04, probe_n=6, probe_q=40, codec="pack4", heldout=True, edge_delay=a.edge_delay, srv_verify=True, ho_n=int(os.environ.get("HO_N", 2)), ho_boxes=int(os.environ.get("HO_BOXES", 0))))
    go("V7c(V7b 에서 T0 제거)", P7(), CFG["selector"], True, gate=0.12, rk=dict(lf_verify=True, probe_n=6, probe_q=40, codec="pack16", heldout=True, gate_since=True, edge_delay=a.edge_delay), **b7)
if a.vfinal:
    go("VF(동결 구성: V6)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, edge=1.0, track=True, fast=10, probe=True, verify=True, gate=9.0, agree=0.85, min_post=30, rk=dict(probe_n=12, probe_q=40, codec="pack16", heldout=True, edge_delay=a.edge_delay, conf_tie=0.05))
if a.v6:
    P_ = lambda: Policy("drift", drift_model=clf, consec=1, cooldown=100)
    base_ = dict(edge=1.0, track=True, fast=10, probe=True, verify=True, gate=9.0, agree=0.85, min_post=30)
    go("V6(V5 + 서버 후보는 학습에 안 쓴 프레임으로 검증)", P_(), CFG["selector"], True, rk=dict(probe_n=12, probe_q=40, codec="pack16", heldout=True), **base_)
if a.v4:
    for mp in [int(x) for x in a.min_posts.split(",")]:
        go(f"V4(V2 + 변화 이후 프레임 {mp}장 이상 모이면 서버 정련)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, edge=1.0, track=True, fast=10, probe=True, verify=True, gate=9.0, agree=0.85, min_post=mp)
if a.v3:
    go("V3(V2 + 히스테리시스 확인)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, edge=1.0, track=True, fast=10, probe=True, verify=True, gate=9.0, agree=0.85, confirm=True)
if a.sens:
    for st_, fm_, lm_ in [(0.15, 0.1, 1.0), (0.4, 0.1, 1.0), (0.25, 0.05, 1.0), (0.25, 0.2, 1.0), (0.25, 0.1, 0.5)]:
        go(f"S(sig_thr={st_}, fast={fm_}, lam={lm_}, 서버 없음)", Policy("drift", drift_model=clf, consec=1, cooldown=100), CFG["selector"], True, edge=1.0, track=True, noserver=True, fast=10, sthr=st_, fmar=fm_, lam=lm_)
nm = np.array(res["A"]["win_env"]) != "clear"
print(f"\n{'방법':20s} {'변화구간mAP50':>13s} {'전체mAP50':>9s} {'적응횟수':>7s} {'업링크KB':>8s} {'다운KB':>7s}  윈도우별 mAP50")
for k, r in res.items():
    print(f"{k:20s} {np.nanmean(r['win_map'][nm]):13.3f} {evaluate(r['preds'], r['gts'], 6)['map50']:9.3f} {sum(e[0] in ('trigger', 'edge') for e in r['events']):7d} {r['up_bytes'] / 1024:8.0f} {r['down_bytes'] / 1024:7.0f}  {np.round(r['win_map'], 2).tolist()}")
_env = np.array(res["A"]["win_env"]); _names = list(dict.fromkeys(_env.tolist()))
print(f"\n환경별 윈도우 평균 mAP50 (열: {_names}) / 엣지 추가 순전파 비율")
for k, r in res.items():
    print(f"{k[:44]:44s}", [round(float(np.nanmean(np.asarray(r['win_map'])[_env == e])), 2) for e in _names], (f"edge_extra={r['edge_extra']:.2f}" if r.get("edge_extra") else ""))
for k, r in res.items():
    if k.startswith(("B", "ES", "D", "V")): print(k, "이벤트:", r["events"], "| 서버:", r.get("server"))
# 시간 지표: 해로운 환경 변화(A 가 기준보다 0.10 넘게 떨어진 윈도우로 시작하는 구간)마다 변화 시점 → 첫 적용 이벤트까지의 지연(프레임)과 성능저하 프레임 수
_W = 50; _wmA = np.asarray(res["A"]["win_map"]); _bad = [w for w in range(1, len(_env)) if _env[w] != _env[w - 1] and _wmA[w] < ref - 0.10]
def _time_metrics(r):
    ev = sorted(e[1] for e in r["events"] if e[0] in ("apply", "edge", "t0on")); lags, deg = [], 0; wm_ = np.asarray(r["win_map"])
    for w in _bad:
        end = w + 1
        while end < len(_env) and _env[end] == _env[w]: end += 1
        t0 = w * _W; nxt = [t for t in ev if t0 <= t < end * _W]; lags.append((nxt[0] - t0) if nxt else (end - w) * _W); deg += int((wm_[w:end] < ref - 0.10).sum()) * _W
    return float(np.mean(lags)) if lags else float("nan"), deg
print(f"\n시간 지표 (해로운 변화 {len(_bad)}회; 적용지연=변화 시점→첫 적용 프레임 수, 적용 없으면 구간 길이; 저하=기준-0.10 미만 프레임 수)")
TM = {}
for k, r in res.items():
    TM[k] = _time_metrics(r); print(f"시간 {k[:40]:40s} 적용지연 {TM[k][0]:6.1f}프레임  저하프레임 {TM[k][1]:5d}")
json.dump(TM, open(W / "time_metrics.json", "w"))
json.dump({k: {"win_map": np.asarray(v["win_map"]).tolist(), "events": [[x if isinstance(x, (str, list)) else int(x) for x in e] for e in v["events"]]} for k, v in res.items()}, open(W / "toy_result.json", "w"), indent=1)
print("\n통합 시험 완료: 위 표는 합성 데이터이므로 성능 근거로 쓰지 않는다. 확인할 것: 오류 없이 끝남, A 의 변화구간 mAP < T, 적응 시스템이 A 보다 높음.")

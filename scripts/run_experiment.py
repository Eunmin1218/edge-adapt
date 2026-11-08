"""한 번의 스트림 실험을 실행하고 results/<tag>.json 에 저장한다.
  A  : 적응 없음                      B  : 주기적 적응(감지·선별 없이 period 마다, 선별=random)
  C  : TENT(--tent bn 이면 BN-Adapt)   D  : 제안(드리프트 감지 + 선별 + 서버 어댑터)
  AMS: AMS-lite 재구성                T  : Teacher 단독(회복 기준선의 상한)
  E  : 엣지 통계정렬 추적(서버·전송 없음, src/stat_align.py)   E+ : E + 신뢰도 부족 때만 서버 증류 어댑터(후보는 Teacher 프로브로 검증)
  ES : EdgeSync-lite (arXiv 2510.21781 재구성: 엔트로피+최신성 선별, Teacher 일치도 하락으로 갱신 요청)
예) python scripts/run_experiment.py --split eval --method D --selector diversity --budget 16 --seed 0 --tag D_diversity_b16_s0"""
import argparse, json, sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT
from src.stream import load_stream
from src.onnx_detector import OnnxDetector
from src.sim import run_stream, Policy
from src.signals import ALL_COLS, CONF_COLS, TEMP_COLS
from src.mapeval import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="eval", choices=["eval", "detector_train"]); ap.add_argument("--method", required=True, choices=["A", "B", "C", "D", "AMS", "RECL", "ES", "T", "E", "E+", "V7"])
ap.add_argument("--selector", default=None, help="기본: configs/adapter.json 의 selector (tune_adapter.py 가 선택), 없으면 diversity"); ap.add_argument("--budget", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--win", type=int, default=50); ap.add_argument("--period", type=int, default=200)
ap.add_argument("--delay", type=int, default=50, help="학습+다운로드 지연(프레임). 서버 학습 시간 실측으로 정한다")
ap.add_argument("--detector", default=None, help="기본: models/drift_logreg_<cols>.pkl"); ap.add_argument("--onnx", default="models/student.onnx", help="방법 A 에서만 다른 ONNX(예: INT8) 평가용"); ap.add_argument("--cols", default="all", choices=["all", "conf", "temp"])
ap.add_argument("--tent", default="tent", choices=["tent", "bn"]); ap.add_argument("--no-registry", action="store_true"); ap.add_argument("--no-cp", action="store_true", help="변화 시점 이후로 후보를 제한하지 않음(ablation)"); ap.add_argument("--consec", type=int, default=1, help="드리프트 확정에 필요한 연속 윈도우 수 (기본 1: 합성 시험에서 2는 EdgeSync-lite보다 한 윈도우 늦어 변화구간 mAP50 0.47 vs 0.65)")
ap.add_argument("--epochs", type=int, default=None, help="기본: configs/adapter.json (tune_adapter.py 결과), 없으면 30"); ap.add_argument("--lr", type=float, default=None); ap.add_argument("--train-layers", type=int, default=None); ap.add_argument("--batch", type=int, default=None); ap.add_argument("--trainer", default=None, choices=["distill", "ultra"]); ap.add_argument("--lrf", type=float, default=None); ap.add_argument("--tag", required=True)
ap.add_argument("--tau", type=float, default=None, help="Teacher 의사 라벨 신뢰도 임계값(필터). 민감도 실험에서 0.3/0.5/0.7 비교")
ap.add_argument("--theta", type=float, default=0.15, help="ES: Teacher 일치도 하락(긴급도) 임계값"); ap.add_argument("--probe-n", type=int, default=4, help="ES: 윈도우마다 서버 Teacher 로 보내는 프로브 프레임 수")
ap.add_argument("--device", default="auto", help="서버 학습·Teacher 추론 장치: auto | cpu | 0"); ap.add_argument("--env-thr", type=float, default=0.08, help="환경 서명 거리 임계값 (scripts/env_sig_check.py 로 정한다)")
ap.add_argument("--edge-delay", type=int, default=13, help="V7: 엣지 통계 정렬 계산 지연(프레임)"); ap.add_argument("--codec", default="pack4", help="V7: 어댑터 부호화 pack4 | pack16"); ap.add_argument("--es-codec", default="auto", help="ES 의 어댑터 부호화(공정 비교용: pack4 를 주면 V7 과 같은 4비트 압축을 ES 에도 적용)"); ap.add_argument("--force", action="store_true", help="이미 results/<tag>.json 이 있어도 다시 실행 (기본: 건너뜀 — 끊긴 실험을 이어서 돌리기 위함)")
a = ap.parse_args()
a.tau_explicit = a.tau is not None            # 민감도 실험(--tau 직접 지정)인지 표시
from src.server_core import load_adapter_cfg
_cfg = load_adapter_cfg()
for _k, _a in (("epochs", "epochs"), ("lr", "lr"), ("train_layers", "train_layers"), ("tau", "tau"), ("batch", "batch"), ("lrf", "lrf"), ("trainer", "trainer"), ("selector", "selector")):
    if getattr(a, _a) is None: setattr(a, _a, _cfg[_k])          # 인자를 안 주면 튜닝 결과(없으면 기본값). 결과 json 의 args 에 실제 사용한 값이 저장된다
from src.adapter_cfg import server_kwargs
SRV_KW = {**server_kwargs(_cfg), "tau": a.tau, "epochs": a.epochs, "lr": a.lr, "train_layers": a.train_layers, "batch": a.batch, "lrf": a.lrf, "trainer": a.trainer}   # 모든 방법의 서버가 같은 학습 설정(mode/alpha/aug 포함)을 쓰도록 한 곳에서 만든다 — 공정 비교
if (ROOT / "results" / f"{a.tag}.json").exists() and not a.force: print(f"건너뜀: results/{a.tag}.json 이 이미 있습니다 (--force 로 재실행)"); sys.exit(0)

cols = {"all": ALL_COLS, "conf": CONF_COLS, "temp": TEMP_COLS}[a.cols]
segs = json.load(open(ROOT / "configs/streams.json"))[a.split]
stream = load_stream(ROOT / "data/yolo" / ("stream_train" if a.split == "detector_train" else "stream_eval"), segs, multiple=a.win)
print(f"스트림 {len(stream)} 프레임, 구간 {[(s[1], s[2]) for s in segs]}")

if a.method in ("C", "T"):            # C = TENT/BN-Adapt(Student), T = Teacher 단독
    from src.torch_detector import TorchDetector
    det = TorchDetector(str(ROOT / ("models/teacher.pt" if a.method == "T" else "models/student.pt")), mode=a.tent if a.method == "C" else "none")
    policy, server = Policy("none"), None
else:
    if a.onnx != "models/student.onnx" and a.method != "A": sys.exit("--onnx 는 방법 A 에서만 쓸 수 있습니다(어댑터 교체는 FP32 기본 모델 기준)")
    det = OnnxDetector(ROOT / a.onnx); server = None
    if a.method == "A": policy = Policy("none")
    elif a.method in ("E", "E+"):                                            # 엣지 통계정렬 추적: 소스 통계는 detector_train 스트림의 맑은 프레임 60장에서 한 번 구한다
        from src.stat_align import StatAligner
        from src.server_core import onnx_inits
        _tr = load_stream(ROOT / "data/yolo/stream_train", json.load(open(ROOT / "configs/streams.json"))["detector_train"], multiple=a.win); _clear = [it["img"] for it in _tr if it["env"] == "clear"]
        _src = [_clear[i] for i in np.linspace(0, len(_clear) - 1, 60).astype(int)]; _sa = StatAligner(ROOT / "models/student.pt", _src); _B = onnx_inits(ROOT / "models/student.onnx")
        policy = Policy("drift", drift_model=pickle.load(open(ROOT / (a.detector or f"models/drift_logreg_{a.cols}.pkl"), "rb")), consec=a.consec, cols=cols, cp=not a.no_cp)
        if a.method == "E+":
            from src.server_core import AdapterServer
            from src.adapter_cfg import server_kwargs
            server = AdapterServer(ROOT / "models/student.pt", ROOT / "models/student.onnx", ROOT / "models/teacher.pt", ROOT / "runs/server_work" / a.tag, device=a.device, **SRV_KW)
    else:
        from src.server_core import AdapterServer
        server = None if a.method == "V7" else AdapterServer(ROOT / "models/student.pt", ROOT / "models/student.onnx", ROOT / "models/teacher.pt", ROOT / "runs/server_work" / a.tag, device=(int(a.device) if a.device.isdigit() else a.device), **SRV_KW)
        if a.method in ("B", "RECL"): policy = Policy("periodic", period=a.period)     # RECL-lite = 주기적 재학습 + 저장 모델 재사용(레지스트리)
        elif a.method == "AMS": policy = Policy("ams", period=a.period)
        elif a.method == "ES": policy = Policy("edgesync", theta=a.theta, probe_n=a.probe_n)
        elif a.method == "V7": policy = Policy("none")
        else: policy = Policy("drift", drift_model=pickle.load(open(ROOT / (a.detector or f"models/drift_logreg_{a.cols}.pkl"), "rb")), consec=a.consec, cols=cols, cp=not a.no_cp)
sel = "random" if a.method in ("B", "AMS", "RECL") else ("edgesync" if a.method == "ES" else a.selector)
_ek = dict(edge_fn=lambda fr: _sa.adapter(fr, _B)[0], edge_sig=_sa.signature, fast_n=10) if a.method in ("E", "E+") else {}
if a.method == "V7":                                                          # 제안 방법 최종판(V7q): GUIDE 5.6.6 — 데모(scripts/demo.py)와 같은 조립 함수를 쓴다
    from src.server_core import AdapterServer
    from src.adapter_cfg import server_kwargs
    from src.v7_agent import make_v7_runner
    _tr = load_stream(ROOT / "data/yolo/stream_train", json.load(open(ROOT / "configs/streams.json"))["detector_train"], multiple=a.win); _clear = [it["img"] for it in _tr if it["env"] == "clear"]
    _src = [_clear[i] for i in np.linspace(0, len(_clear) - 1, 60).astype(int)]
    server = AdapterServer(ROOT / "models/student.pt", ROOT / "models/student.onnx", ROOT / "models/teacher.pt", ROOT / "runs/server_work" / a.tag, device=a.device, **SRV_KW)
    _rn = make_v7_runner(det, ROOT / "models/student.pt", _src, server, budget=a.budget, win=a.win, edge_delay=a.edge_delay, codec=a.codec, apply_delay=a.delay, seed=a.seed)
    for it in stream: _rn.step(it["img"], it["gt"], it["env"])
    r = _rn.result()
else:
  r = run_stream(stream, det, policy, server, selector=sel, budget=a.budget, win=a.win, apply_delay=a.delay, use_registry=(a.method in ("D", "RECL", "E", "E+") and not a.no_registry), seed=a.seed, env_thr=(0.25 if a.method in ("E", "E+") and a.env_thr == 0.08 else a.env_thr), probe_fn=(server.agreement if a.method in ("ES", "E+") else None), codec=(a.es_codec if a.method == "ES" else "auto"), **_ek)

def _jsonable(x):
    """이벤트 튜플(문자열·정수·실수·리스트 혼합; V7 의 probe 이벤트는 리스트를 포함)을 JSON 저장 가능한 값으로 바꾼다."""
    if isinstance(x, (list, tuple)): return [_jsonable(v) for v in x]
    if isinstance(x, (str, bool)) or x is None: return x
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, (float, np.floating)): return float(x)
    return str(x)


env = np.array(r["win_env"]); wm = r["win_map"]
shift = np.array([e != "clear" for e in env])
out = {"tag": a.tag, "args": vars(a), "win_map": wm.tolist(), "win_env": r["win_env"], "events": [_jsonable(e) for e in r["events"]],
       "map_all": evaluate(r["preds"], r["gts"], 6), "map_shift_windows": float(np.nanmean(wm[shift])) if shift.any() else None,
       "map_fog_windows": float(np.nanmean(wm[env == "fog"])) if (env == "fog").any() else None,
       "up_bytes": r["up_bytes"], "down_bytes": r["down_bytes"], "n_trigger": sum(e[0] in ("trigger", "edge") for e in r["events"]), "edge_extra": r.get("edge_extra"), "lat_ms": r["lat_ms"]}
(ROOT / "results").mkdir(exist_ok=True); json.dump(out, open(ROOT / "results" / f"{a.tag}.json", "w"), indent=1)
if a.method in ("A", "T"):                                                   # 감지기 학습/평가용 윈도우 특징 저장
    np.savez(ROOT / "results" / (f"winfeats_{a.split}.npz" if a.method == "A" else f"winfeats_T_{a.split}.npz"), feats=r["win_frame_feats"], win_map=wm, env=env)
print(json.dumps({k: out[k] for k in ["tag", "map_all", "map_shift_windows", "map_fog_windows", "up_bytes", "down_bytes", "n_trigger", "lat_ms"]}, indent=1))
print("이벤트:", out["events"])

"""실시간 시연 데모 (노트북). 같은 영상을 세 시스템에 동시에 흘려 한 화면에서 비교한다.
  ① 적응 없음(Student 고정)   ② EdgeSync-lite (arXiv 2510.21781 재구성, 비교 대상)   ③ 제안(드리프트 감지 + 프레임 선별 + 환경별 어댑터 저장)
영상: 평가 스트림(맑음→비→맑음→밤→…) / 녹화 영상 파일 / 웹캠.  웹캠·영상 파일에는 정답이 없으므로 mAP 대신 '윈도우 평균 신뢰도' 곡선을 그린다.
예)  python scripts/demo.py --source stream --server http://127.0.0.1:8000
     python scripts/demo.py --source drive.mp4 --server http://127.0.0.1:8000
     python scripts/demo.py --source 0 --server http://127.0.0.1:8000 --agents static,ours
     python scripts/demo.py --source stream --server http://127.0.0.1:8000 --record results/demo.mp4     (발표 백업 영상)
키: q 종료 / SPACE 일시정지 / r 모든 시스템의 어댑터 초기화 / s 스크린샷 저장"""
import argparse, sys, json, pickle
import numpy as np
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, IMG_W, IMG_H
from src.onnx_detector import OnnxDetector
from src.sim import StreamRunner, Policy
from src.edge_client import EdgeClient
from src.demo_core import run_demo

ap = argparse.ArgumentParser(); ap.add_argument("--source", default="stream"); ap.add_argument("--server", default=None, help="예: http://127.0.0.1:8000 (없으면 적응 요청이 '네트워크 단절' 상태로 남는다)")
ap.add_argument("--agents", default="static,edgesync,ours"); ap.add_argument("--detector", default="models/drift_logreg_all.pkl", help="(구버전 호환용, 제안 방법 V7q 에서는 쓰지 않음)"); ap.add_argument("--models", default="models", help="student.onnx / student.pt 가 있는 폴더 (합성 시험: models_toy)"); ap.add_argument("--edge-delay", type=int, default=13, help="엣지 통계 정렬 계산 지연(프레임)"); ap.add_argument("--codec", default="pack4"); ap.add_argument("--budget", type=int, default=16)
ap.add_argument("--win", type=int, default=50); ap.add_argument("--fps", type=float, default=10.0); ap.add_argument("--consec", type=int, default=1); ap.add_argument("--cooldown", type=int, default=100)
ap.add_argument("--env-thr", type=float, default=0.08); ap.add_argument("--scale", type=float, default=0.65, help="패널 크기 배율 (노트북 해상도가 작으면 0.5)")
ap.add_argument("--record", default=None); ap.add_argument("--headless", action="store_true"); ap.add_argument("--snap-dir", default=None); ap.add_argument("--max-frames", type=int, default=None)
ap.add_argument("--out", default="results/demo_summary.json"); a = ap.parse_args()

if a.source == "stream":
    from src.stream import load_stream
    items = load_stream(ROOT / "data/yolo/stream_eval", json.load(open(ROOT / "configs/streams.json"))["eval"], multiple=a.win); has_gt = True
    frames = ((it["img"], it["gt"], it["env"]) for it in items); n_win = len(items) // a.win
elif a.source == "toy":
    from src.toyworld import stream as toy_stream
    SEGS = [("clear", 100), ("night", 150), ("clear", 50), ("glare", 150), ("clear", 50), ("heavy_rain", 100), ("clear_sparse", 100), ("night", 100), ("clear_dense", 100), ("clear", 100)]
    items = toy_stream(201, SEGS); has_gt = True; frames = ((it["img"], it["gt"], it["env"]) for it in items); n_win = len(items) // a.win
else:
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source); has_gt = False; n_win = None
    if not cap.isOpened(): sys.exit(f"영상을 열 수 없습니다: {a.source}")

    def _gen():
        while True:
            ok, f = cap.read()
            if not ok: return
            yield cv2.resize(f, (IMG_W, IMG_H)), None, None
    frames = _gen()

def source_frames(n=60):
    """소스 통계용 맑은 낮 프레임. toy: 합성 맑은 프레임 60장 / 그 밖: 학습 폴더(data/yolo/student_train)의 이미지를 고르게 60장."""
    import numpy as np
    if a.source == "toy":
        from src.toyworld import make_img
        rg = np.random.default_rng(5); return [make_img(rg, "clear")[0] for _ in range(n)]
    fs = sorted((ROOT / "data/yolo/student_train").rglob("*.jpg")) or sorted((ROOT / "data/yolo/student_train").rglob("*.png"))
    if not fs: sys.exit("소스 통계용 이미지가 없습니다: data/yolo/student_train (빠른 시작 가이드의 데이터 준비 확인)")
    return [cv2.resize(cv2.imread(str(fs[i])), (IMG_W, IMG_H)) for i in np.linspace(0, len(fs) - 1, n).astype(int)]


NAMES = {"static": ("A · 적응 없음", (200, 200, 200)), "edgesync": ("EdgeSync-lite (2025)", (60, 160, 255)), "ours": ("제안 (Ours)", (80, 220, 80))}


class NoServer:                                           # --server 없이 실행할 때: 적응 요청은 '큐에 보관' 상태로 남는다
    status = "queued"
    def ready(self): return False


agents = []
for kind in [k.strip() for k in a.agents.split(",")]:
    name, color = NAMES[kind]; det = OnnxDetector(ROOT / a.models / "student.onnx"); cli = EdgeClient(a.server, tag=kind, base_inits=det._base) if a.server else None
    srv = (lambda fr, env, c=cli: c.remote(fr, env)) if cli else (lambda fr, env: NoServer())
    if kind == "static": r = StreamRunner(det, Policy("none"), None, win=a.win, use_registry=False)
    elif kind == "edgesync": r = StreamRunner(det, Policy("edgesync"), srv, selector="edgesync", budget=a.budget, win=a.win, use_registry=False, probe_fn=(cli.probe if cli else (lambda f, d: 1.0)))
    else:
        from src.v7_agent import make_v7_runner
        r = make_v7_runner(det, ROOT / a.models / "student.pt", source_frames(), srv, budget=a.budget, win=a.win, edge_delay=a.edge_delay, codec=a.codec)
    agents.append({"name": name, "color": color, "runner": r})

summ = run_demo(frames, agents, win=a.win, n_win=n_win, has_gt=has_gt, show=not a.headless, fps=a.fps, record=a.record, snap_dir=a.snap_dir, snap_every=a.win * 5 if a.snap_dir else 0, max_frames=a.max_frames, scale=a.scale)
(ROOT / a.out).parent.mkdir(exist_ok=True); json.dump(summ, open(ROOT / a.out, "w"), ensure_ascii=False, indent=1)
for k, v in summ.items(): print(f"{k}: 변화구간 평균 {v['mean_shift_window']}, 업링크 {v['up_kb']:.0f}KB, 다운링크 {v['down_kb']:.0f}KB, 적응 {v['n_trigger']}회")

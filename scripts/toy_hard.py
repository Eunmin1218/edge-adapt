"""더 어려운 합성 시나리오(환경 5종)용 준비: Student(맑음만, runs/toy 에서 복사)와 Teacher(5환경 전부)를 runs/toy2 에 만들고, 환경별 적응 전 성능(A, T)을 출력한다.
적응 방식 비교는 scripts/ablate_adapter.py (--toy2), 전체 스트림 비교는 scripts/toy_stream.py 로 한다."""
import shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.common import ROOT, CLASSES
import argparse
from src.toyworld import ENVS, ENVS3, make_img, write_set, stream
from src.server_core import export_onnx
from src.onnx_detector import OnnxDetector
from src.mapeval import evaluate

_ap = argparse.ArgumentParser(); _ap.add_argument("--world", default="toy2", choices=["toy2", "toy3"]); _a = _ap.parse_args()
if _a.world == "toy3": ENVS = ENVS3
W, W1 = ROOT / f"runs/{_a.world}", ROOT / "runs/toy"; W.mkdir(parents=True, exist_ok=True)
if not (W / "student.onnx").exists():                         # Student: toy 와 같은 모델(맑음만 학습)
    for f in ("student.pt", "student.onnx"): shutil.copy(W1 / f, W / f)
if not (W / "data/teacher_train").exists():
    write_set(W, "teacher_train", 40 * len(ENVS), list(ENVS), 3); write_set(W, "teacher_val", 6 * len(ENVS), list(ENVS), 4)
if not (W / "teacher.pt").exists():
    from ultralytics import YOLO
    (W / "teacher.yaml").write_text(f"path: {(W / 'data').resolve()}\ntrain: teacher_train/images\nval: teacher_val/images\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    t0 = time.time(); YOLO("yolo11n.pt").train(data=str(W / "teacher.yaml"), epochs=14, imgsz=640, batch=16, device="cpu", workers=0, plots=False, project=str(W), name="teacher", exist_ok=True, verbose=False, amp=False, optimizer="AdamW", lr0=0.004, warmup_epochs=1, val=False, mosaic=0.0)
    print(f"Teacher 학습 {time.time() - t0:.0f}s"); shutil.copy(W / "teacher/weights/last.pt", W / "teacher.pt"); export_onnx(W / "teacher.pt", W / "teacher.onnx")
if not (W / "teacher.onnx").exists(): export_onnx(W / "teacher.pt", W / "teacher.onnx")
rng = np.random.default_rng(99)
for name in ("student", "teacher"):
    det = OnnxDetector(W / f"{name}.onnx"); row = []
    for env in ENVS:
        fr = [make_img(rng, env) for _ in range(60)]; pr = [(o["boxes"], o["scores"], o["cls"]) for o in map(det.infer, [f[0] for f in fr])]
        row.append(f"{env} {evaluate(pr, [f[1] for f in fr], 6, iou_thrs=[0.5])['map50']:.3f}")
    print(f"{name:8s} mAP50: " + " | ".join(row), flush=True)

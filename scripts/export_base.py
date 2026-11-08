"""학습된 Student(.pt)를 ONNX로 내보내 엣지 기준 모델로 쓴다. 사용: python scripts/export_base.py runs/student/weights/best.pt"""
import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT
from src.server_core import export_onnx, onnx_inits

pt = Path(sys.argv[1]); (ROOT / "models").mkdir(exist_ok=True)
shutil.copy(pt, ROOT / "models/student.pt")
out = export_onnx(ROOT / "models/student.pt", ROOT / "models/student.onnx")
inits = onnx_inits(out); n = len(__import__("ultralytics").YOLO(str(ROOT / "models/student.pt")).model.model) - 1
head = [k for k in inits if f"model.{n}." in k]
print("ONNX:", out, f"{Path(out).stat().st_size/2**20:.2f} MB, initializer {len(inits)}개 / Detect(model.{n}.) 이름 포함 {len(head)}개")
print("이름 예시:", list(inits)[:4])
print("model.%d. 로 시작하는 이름 예시:" % n, head[:4])

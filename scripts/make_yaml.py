"""YOLO 학습용 data.yaml 생성. 사용: python scripts/make_yaml.py student|teacher"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, CLASSES

role = sys.argv[1]; d = ROOT / "data/yolo"; (ROOT / "configs").mkdir(exist_ok=True)
txt = f"path: {d.resolve()}\ntrain: {role}_train/images\nval: {role}_val/images\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
(ROOT / f"configs/{role}.yaml").write_text(txt); print(txt)

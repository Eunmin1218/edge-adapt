"""변환된 YOLO 데이터의 이미지-라벨 정렬을 눈으로 확인한다. 사용: python scripts/vis_check.py data/yolo/student_train [--n 8]
무작위 n장에 정답 박스를 그려 results/vis_<폴더이름>.jpg 한 장(2열 격자)으로 저장한다. 박스가 물체에서 벗어나 있으면 좌표 변환이 틀린 것이다."""
import argparse, random, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, CLASSES, IMG_W, IMG_H
from src.stream import _gt

ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
d = Path(a.folder); imgs = sorted((d / "images").glob("*.jpg")); random.Random(a.seed).shuffle(imgs); tiles = []
for p in imgs[:a.n]:
    im = cv2.imread(str(p)); boxes, cls = _gt(d / "labels" / (p.stem + ".txt"))
    for (x1, y1, x2, y2), c in zip(boxes, cls):
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2); cv2.putText(im, CLASSES[c], (int(x1), max(12, int(y1) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(im, p.stem, (5, IMG_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1); tiles.append(cv2.resize(im, (IMG_W // 2, IMG_H // 2)))
if not tiles: sys.exit(f"{d}/images 에 이미지가 없습니다")
if len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
(ROOT / "results").mkdir(exist_ok=True); out = ROOT / "results" / f"vis_{d.name}.jpg"; cv2.imwrite(str(out), grid); print("저장:", out)

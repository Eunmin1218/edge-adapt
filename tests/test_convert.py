import json, zipfile, tempfile
from pathlib import Path
import cv2, numpy as np
from scripts.shift_to_yolo import convert


def test_convert():
    d = Path(tempfile.mkdtemp()); frames = []
    with zipfile.ZipFile(d / "img.zip", "w") as zf:
        for fi in range(0, 20):
            ok, buf = cv2.imencode(".jpg", np.full((800, 1280, 3), 100, np.uint8))
            zf.writestr(f"seqA/{fi:08d}_img_front.jpg", buf.tobytes())
            frames.append({"name": f"{fi:08d}_img_front.jpg", "videoName": "seqA", "frameIndex": fi,
                           "labels": [{"category": "car", "box2d": {"x1": 100, "y1": 100, "x2": 300, "y2": 250}},
                                      {"category": "bicycle", "box2d": {"x1": 10, "y1": 10, "x2": 14, "y2": 14}}] if fi % 2 == 0 else []})
    json.dump({"frames": frames}, open(d / "det.json", "w"))
    (d / "seqs.txt").write_text("seqA\n")
    n = convert(d / "det.json", d / "img.zip", d / "seqs.txt", d / "out", step=5, drop_empty=True)
    assert n == 2, n                     # frameIndex 0,10 (5의 배수 중 짝수, 라벨 있음); 5,15는 라벨 없음
    lab = (d / "out/labels/seqA_000000.txt").read_text().split("\n")
    assert len(lab) == 1 and lab[0].startswith("1 "), lab        # 작은 자전거 박스(4px)는 최소 크기 필터로 제외

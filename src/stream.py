"""변환된 YOLO 폴더(meta.csv 포함)에서 평가 스트림을 만든다. 이미지는 접근할 때 디스크에서 읽는다(메모리 절약)."""
import csv, cv2, numpy as np
from pathlib import Path
from .common import IMG_W, IMG_H


class StreamItem:
    __slots__ = ("path", "gt", "seq", "env")

    def __init__(self, path, gt, seq, env): self.path, self.gt, self.seq, self.env = path, gt, seq, env

    def __getitem__(self, k):
        if k == "img": return cv2.imread(str(self.path))
        return {"gt": self.gt, "seq": self.seq, "env": self.env}[k]


def _gt(label_file):
    rows = [list(map(float, l.split())) for l in Path(label_file).read_text().splitlines() if l.strip()]
    if not rows: return np.zeros((0, 4)), np.zeros(0, int)
    a = np.array(rows); c = a[:, 0].astype(int)
    xyxy = np.stack([(a[:, 1] - a[:, 3] / 2) * IMG_W, (a[:, 2] - a[:, 4] / 2) * IMG_H, (a[:, 1] + a[:, 3] / 2) * IMG_W, (a[:, 2] + a[:, 4] / 2) * IMG_H], 1)
    return xyxy, c


def load_stream(yolo_dir, segments, multiple=50):
    """segments: [(seq_id, env_tag, max_frames), ...] 를 순서대로 이어 붙인다.
    각 구간 길이를 multiple(=윈도우 길이)의 배수로 내려 윈도우가 환경 경계를 넘지 않게 한다."""
    yolo_dir = Path(yolo_dir); rows = list(csv.DictReader(open(yolo_dir / "meta.csv")))
    out = []
    for seq, env, mx in segments:
        fr = sorted([r for r in rows if r["seq"] == seq], key=lambda r: int(r["frame"]))[:mx]
        n = len(fr) // multiple * multiple
        if n == 0: raise ValueError(f"스트림 구성 실패: 시퀀스 {seq} 의 프레임이 {len(fr)}개뿐({multiple}개 미만). {yolo_dir}/meta.csv 확인")
        out += [StreamItem(yolo_dir / "images" / f"{r['name']}.jpg", _gt(yolo_dir / "labels" / f"{r['name']}.txt"), seq, env) for r in fr[:n]]
    return out

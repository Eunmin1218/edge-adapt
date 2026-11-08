"""SHIFT(Scalabel det_2d.json) → YOLO 형식 변환 + 640x400 리사이즈 + 프레임 솎아내기 + 최소 박스 필터.
사용 예:
  python scripts/shift_to_yolo.py --det2d <det_2d.json> --images <img.zip 또는 폴더> --seqs data/lists/student_train.txt \
         --out data/yolo/student_train --step 1 --drop-empty
  --step : 원본 프레임에서 몇 장마다 1장 쓸지 (1Hz 이미지 = 1, 10Hz 영상에서 1Hz = 10, 10Hz 전체 = 1)
  --drop-empty : 박스가 하나도 없는 프레임 제외 (학습용). 평가 스트림에는 쓰지 않는다.
출력: <out>/images/<seq>_<frame>.jpg, <out>/labels/<seq>_<frame>.txt, <out>/meta.csv(name,seq,frame)
같은 --out 에 여러 번 실행하면 meta.csv 가 덮어써지므로, 여러 json 을 합칠 때는 --append 를 쓴다."""
import argparse, json, csv, zipfile, sys
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import CLS2ID, IMG_W, IMG_H, MIN_BOX_PX


class ImageSource:
    """zip 또는 폴더에서 '<seq>/<frame파일명>' 으로 이미지를 읽는다."""
    def __init__(self, path):
        p = Path(path); self.zf = zipfile.ZipFile(p) if p.suffix == ".zip" else None; self.dir = None if self.zf else p
        self.names = set(self.zf.namelist()) if self.zf else None

    def read(self, seq, fname):
        rel = f"{seq}/{fname}"
        if self.zf:
            cand = [n for n in (rel, "img/" + rel, "./" + rel) if n in self.names]
            if not cand:
                return None
            data = np.frombuffer(self.zf.read(cand[0]), np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        for f in (self.dir / rel, self.dir / "img" / rel):
            if f.exists(): return cv2.imread(str(f))
        return None


def convert(det2d, images, seqs, out, step, drop_empty, min_box=MIN_BOX_PX, append=False):
    out = Path(out); (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(parents=True, exist_ok=True)
    src = ImageSource(images); data = json.load(open(det2d)); frames = data["frames"] if isinstance(data, dict) else data
    want = set(Path(seqs).read_text().split()) if seqs else None
    by_seq = {}
    for fr in frames:
        by_seq.setdefault(fr["videoName"], []).append(fr)
    rows, n_img, n_box, n_small, n_missing = [], 0, 0, 0, 0; unknown = {}
    for seq, frs in sorted(by_seq.items()):
        if want is not None and seq not in want:
            continue
        frs.sort(key=lambda f: f["frameIndex"])
        for fr in frs:
            if fr["frameIndex"] % step != 0:
                continue
            labels = []
            for lb in fr.get("labels") or []:
                if lb["category"] not in CLS2ID or "box2d" not in lb:
                    if "box2d" in lb: unknown[lb["category"]] = unknown.get(lb["category"], 0) + 1
                    continue
                b = lb["box2d"]; x1, y1, x2, y2 = [b[k] * 0.5 for k in ("x1", "y1", "x2", "y2")]   # 1280x800 → 640x400
                x1, y1 = max(x1, 0), max(y1, 0); x2, y2 = min(x2, IMG_W), min(y2, IMG_H)
                w, h = x2 - x1, y2 - y1
                if w < min_box or h < min_box:
                    n_small += 1; continue
                labels.append((CLS2ID[lb["category"]], (x1 + x2) / 2 / IMG_W, (y1 + y2) / 2 / IMG_H, w / IMG_W, h / IMG_H))
            if drop_empty and not labels:
                continue
            img = src.read(seq, fr["name"])
            if img is None:
                n_missing += 1; continue
            img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
            name = f"{seq}_{fr['frameIndex']:06d}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / f"{name}.txt").write_text("\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in labels))
            rows.append((name, seq, fr["frameIndex"])); n_img += 1; n_box += len(labels)
    meta = out / "meta.csv"; old = list(csv.reader(open(meta)))[1:] if (append and meta.exists()) else []
    with open(meta, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "seq", "frame"]); w.writerows(old); w.writerows(rows)
    print(f"이미지 {n_img}장, 박스 {n_box}개, 최소크기 미만 제외 {n_small}개, 이미지 못 찾음 {n_missing}장 → {out}")
    if unknown: print(f"경고: src/common.py 의 CLASSES 에 없는 카테고리(무시됨): {dict(sorted(unknown.items(), key=lambda kv: -kv[1])[:8])}")
    if n_img == 0:
        sys.exit("변환된 이미지가 0장입니다. --seqs 목록의 시퀀스 이름, --images 경로, zip 내부 경로(scripts/inspect_shift.py)를 확인하세요.")
    return n_img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--det2d", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--seqs", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=1); ap.add_argument("--drop-empty", action="store_true")
    ap.add_argument("--min-box", type=float, default=MIN_BOX_PX); ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    convert(a.det2d, a.images, a.seqs, a.out, a.step, a.drop_empty, a.min_box, a.append)

"""img.tar 를 앞에서부터 한 번 훑으며, 시퀀스 목록(data/lists/*.txt)에 있는 시퀀스의 파일만 추출한다(디스크 절약).
사용: python scripts/extract_seqs.py --tar <img.tar> --seqs data/lists/student_train.txt data/lists/teacher_train.txt --out data/raw/img_train --every 10
  --every N : 파일 이름 앞의 프레임 번호(예: 00000030_img_front.jpg → 30)가 N 의 배수인 프레임만 꺼낸다.
              10Hz 원본에서 1Hz 학습 이미지만 필요하면 --every 10 (용량이 1/10), 스트림용 10Hz 전체 프레임은 --every 1.
tar 안의 경로가 '<seq>/<프레임>.jpg' 이든 'img/<seq>/...' 이든 시퀀스 폴더 이름을 찾아 '<out>/<seq>/<프레임>.jpg' 로 풀어 준다."""
import argparse, re, tarfile
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--tar", required=True); ap.add_argument("--seqs", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--every", type=int, default=1)
a = ap.parse_args()
want = set()
for f in a.seqs: want |= set(Path(f).read_text().split())
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n_files, seen = 0, set()
with tarfile.open(a.tar, "r|*") as tf:                      # 스트리밍 모드: 전체를 메모리/디스크에 풀지 않는다
    for m in tf:
        if not m.isfile(): continue
        parts = Path(m.name).parts
        seq_idx = next((i for i, p in enumerate(parts[:-1]) if p in want), None)
        if seq_idx is None: continue
        mm = re.match(r"(\d+)", parts[-1])
        if a.every > 1 and mm and int(mm.group(1)) % a.every != 0: continue
        dest = out / parts[seq_idx] / parts[-1]; dest.parent.mkdir(parents=True, exist_ok=True)
        with tf.extractfile(m) as src, open(dest, "wb") as dst: dst.write(src.read())
        n_files += 1; seen.add(parts[seq_idx])
print(f"추출한 파일 {n_files}개, 시퀀스 {len(seen)}/{len(want)}개 → {out}")
if len(seen) < len(want): print("경고: 목록에 있지만 tar 에서 못 찾은 시퀀스:", sorted(want - seen)[:10], "...")

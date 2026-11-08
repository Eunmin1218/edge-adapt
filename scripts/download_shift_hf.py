"""Hugging Face 미러에서 SHIFT 이산(discrete) 10Hz 비디오 분할의 front 뷰 파일을 받는다.
공식 서버(dl.cv.ethz.ch)가 내려가 있는 경우의 대안이다. 사용: python scripts/download_shift_hf.py --split val   (먼저 val, 그다음 train)
받는 파일: discrete/videos/<split>/front/{img.tar, det_2d.json, seq.csv}  → data/raw/ 아래. 용량이 크므로 --dry-run 으로 파일 목록·크기부터 확인한다."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

REPO = "KevinConnorLee/SHIFT"
ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=["train", "val"]); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--out", default="data/raw")
a = ap.parse_args()
pats = [f"discrete/videos/{a.split}/front/img.tar", f"discrete/videos/{a.split}/front/det_2d.json", f"discrete/videos/{a.split}/front/seq.csv"]
info = HfApi().dataset_info(REPO, files_metadata=True)
files = [s for s in info.siblings if s.rfilename in pats]
if not files: raise SystemExit(f"미러에 {pats} 가 없습니다. 미러 구조가 바뀌었을 수 있으니 https://huggingface.co/datasets/{REPO}/tree/main 에서 확인하세요.")
for s in files: print(f"{s.rfilename}  {(s.size or 0)/2**30:.2f} GB")
if a.dry_run: raise SystemExit("dry-run: 다운로드하지 않음")
Path(a.out).mkdir(parents=True, exist_ok=True)
snapshot_download(REPO, repo_type="dataset", allow_patterns=pats, local_dir=a.out)
print("완료:", Path(a.out) / f"discrete/videos/{a.split}/front")

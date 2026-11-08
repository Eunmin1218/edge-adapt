"""환경 서명(src/env.py)이 평가 스트림의 환경(맑음/비/밤/안개)을 구분하는지 확인하고 --env-thr 값을 제안한다.
사용: python scripts/env_sig_check.py [--split eval] [--win 50]
같은 환경 태그의 윈도우끼리 거리(최대)가 다른 환경 태그와의 거리(최소)보다 작아야 임계값으로 구분할 수 있다."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT
from src.stream import load_stream
from src.env import env_signature

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="eval"); ap.add_argument("--win", type=int, default=50); a = ap.parse_args()
segs = json.load(open(ROOT / "configs/streams.json"))[a.split]
stream = load_stream(ROOT / "data/yolo" / ("stream_train" if a.split == "detector_train" else "stream_eval"), segs, multiple=a.win)
sigs, tags = [], []
for s in range(0, len(stream) - a.win + 1, a.win):
    sigs.append(env_signature([stream[i]["img"] for i in range(s, s + a.win)])); tags.append(stream[s]["env"])
sigs = np.stack(sigs); tags = np.array(tags); D = np.linalg.norm(sigs[:, None] - sigs[None], axis=2)
same = D[(tags[:, None] == tags[None]) & ~np.eye(len(tags), dtype=bool)]; diff = D[tags[:, None] != tags[None]]
print(f"같은 환경 윈도우 간 거리: 평균 {same.mean():.3f}, 최대 {same.max():.3f}")
print(f"다른 환경 윈도우 간 거리: 평균 {diff.mean():.3f}, 최소 {diff.min():.3f}")
for t1 in dict.fromkeys(tags):
    print(f"  {t1:6s} → " + ", ".join(f"{t2}:{D[np.ix_(tags == t1, tags == t2)].mean():.3f}" for t2 in dict.fromkeys(tags)))
if same.max() < diff.min():
    print(f"구분 가능 → --env-thr {np.sqrt(same.max() * diff.min()):.3f} 권장 (같은 환경 최대와 다른 환경 최소의 기하평균)")
else:
    print(f"구분 불가(같은 환경 최대 {same.max():.3f} >= 다른 환경 최소 {diff.min():.3f}). 시작 환경(맑음)이 시퀀스마다 다르면 이렇게 나온다."
          f" 대안: --no-registry 로 레지스트리를 끄거나 src/env.py 의 서명을 개선(예: 색 히스토그램)하고, 이 한계를 보고서에 적는다.")

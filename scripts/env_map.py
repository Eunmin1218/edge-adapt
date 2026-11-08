"""환경별 평균 mAP50 출력(적응 필요 기준 '하락 폭'을 정할 때 사용). 사용: python scripts/env_map.py results/A_dtrain.json results/T_dtrain.json"""
import json, sys
import numpy as np

for f in sys.argv[1:]:
    r = json.load(open(f)); wm, env = np.array(r["win_map"], float), np.array(r["win_env"])
    print(f, {e: round(float(np.nanmean(wm[env == e])), 3) for e in dict.fromkeys(env)})

"""RQ1: 드리프트 감지기 학습·평가. 먼저 A를 detector_train / eval 두 스트림에 실행해 results/winfeats_*.npz 를 만든 뒤 실행한다.
  python scripts/train_detector.py --drop 0.10
출력: 신호 조합(신뢰도만/시간적 불일치만/둘 다) × 모델(logreg/gbm) 및 DoC 기준선의 Recall·오경보율 (전체 / 안개 구간)."""
import argparse, sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT
from src.signals import window_features, ALL_COLS, CONF_COLS, TEMP_COLS
from src.drift_model import make_labels, fit_binary, recall_false_alarm, DoCDetector, fit_regress

ap = argparse.ArgumentParser(); ap.add_argument("--drop", type=float, default=0.10); a = ap.parse_args()
tr = np.load(ROOT / "results/winfeats_detector_train.npz", allow_pickle=True); ev = np.load(ROOT / "results/winfeats_eval.npz", allow_pickle=True)
ref = float(np.nanmean(tr["win_map"][tr["env"] == "clear"])); print(f"기준 mAP50(맑은 낮, 학습 스트림) = {ref:.3f}, 적응 필요 기준: 하락 >= {a.drop}")
ytr, yev = make_labels(tr["win_map"], ref, a.drop), make_labels(ev["win_map"], ref, a.drop)
for nm, y in (("학습", ytr), ("평가", yev)):
    if (y == 1).sum() < 1 or (y == 0).sum() < 1:
        sys.exit(f"{nm} 스트림 라벨이 한쪽 클래스뿐입니다(적응 필요 {(y==1).sum()}개/정상 {(y==0).sum()}개). --drop 값을 바꾸거나 Student의 환경별 mAP(scripts/env_map.py)를 먼저 확인하세요.")
fog = ev["env"] == "fog"; (ROOT / "models").mkdir(exist_ok=True)


def X(d, cols): return np.stack([window_features(f, cols) for f in d["feats"]])


rows = []
for name, cols in [("conf", CONF_COLS), ("temp", TEMP_COLS), ("all", ALL_COLS)]:
    for kind in ["logreg", "gbm"]:
        m = fit_binary(X(tr, cols), ytr, kind); p = m.predict(X(ev, cols)); r_all = recall_false_alarm(yev, p)
        r_fog = recall_false_alarm(np.where(fog, yev, -1), p)
        rows.append((name, kind, r_all["recall"], r_all["false_alarm"], r_fog["recall"], r_fog["n_pos"]))
        if kind == "logreg": pickle.dump(m, open(ROOT / f"models/drift_logreg_{name}.pkl", "wb"))
doc = DoCDetector().fit(X(tr, ALL_COLS)[tr["env"] == "clear"], X(tr, ALL_COLS), ytr); p = doc.predict(X(ev, ALL_COLS))
ra, rf = recall_false_alarm(yev, p), recall_false_alarm(np.where(fog, yev, -1), p)
rows.append(("DoC", "-", ra["recall"], ra["false_alarm"], rf["recall"], rf["n_pos"]))
print(f"{'신호':6s} {'모델':7s} {'Recall':>7s} {'오경보율':>8s} {'안개Recall':>10s} (안개 양성 윈도우 수)")
for r in rows: print(f"{r[0]:6s} {r[1]:7s} {r[2]:7.2f} {r[3]:8.2f} {r[4]:10.2f} ({r[5]})")
reg = fit_regress(X(tr, ALL_COLS), ref - tr["win_map"]); pred = reg.predict(X(ev, ALL_COLS)); ok = ~np.isnan(ev["win_map"])
print(f"회귀형(예상 mAP 하락량) 상관계수 r = {np.corrcoef(pred[ok], (ref - ev['win_map'])[ok])[0, 1]:.2f}")

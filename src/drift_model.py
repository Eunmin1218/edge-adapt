"""드리프트 감지기(이진/회귀)와 라벨 없는 정확도 추정 기준선(DoC). 학습 데이터는 '적응 없음(A)' 스트림 실행 결과에서 만든다."""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def make_labels(win_map, ref_map, drop):
    """윈도우 mAP가 기준(맑은 낮 평균 mAP)보다 drop 이상 낮으면 1(적응 필요), 아니면 0. nan 윈도우는 -1(제외)."""
    return np.where(np.isnan(win_map), -1, (ref_map - win_map >= drop).astype(int))


def fit_binary(X, y, kind="logreg", seed=0):
    ok = y >= 0
    clf = LogisticRegression(class_weight="balanced", max_iter=2000) if kind == "logreg" else GradientBoostingClassifier(random_state=seed)
    return make_pipeline(StandardScaler(), clf).fit(X[ok], y[ok])


def fit_regress(X, drop_amount):
    ok = ~np.isnan(drop_amount); return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[ok], drop_amount[ok])


def recall_false_alarm(y_true, y_pred):
    ok = y_true >= 0; yt, yp = y_true[ok], y_pred[ok]
    rec = float(((yp == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1)); fa = float(((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1))
    return {"recall": rec, "false_alarm": fa, "n_pos": int((yt == 1).sum()), "n_neg": int((yt == 0).sum())}


class DoCDetector:
    """Difference of Confidences(Guillory et al.)를 탐지 스트림에 옮긴 기준선: 평균 신뢰도 하락량만으로 판정."""
    def __init__(self, conf_col=1): self.col = conf_col; self.src = None; self.thr = None

    def fit(self, X_src_clear, X_all, y_all):
        self.src = float(X_src_clear[:, self.col].mean())
        ok = y_all >= 0; drops = self.src - X_all[ok, self.col]
        best, self.thr = -1, 0.0
        for t in np.unique(np.quantile(drops, np.linspace(0.05, 0.95, 37))):
            p = (drops >= t).astype(int); r = recall_false_alarm(y_all[ok], p); score = r["recall"] - r["false_alarm"]
            if score > best: best, self.thr = score, float(t)
        return self

    def predict(self, X): return ((self.src - X[:, self.col]) >= self.thr).astype(int)

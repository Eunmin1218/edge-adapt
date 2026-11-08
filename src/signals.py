"""드리프트 감지 신호 (정답 라벨 불사용). 프레임 단위 특징 → 윈도우 단위 특징."""
import numpy as np
from .mapeval import iou_matrix

FRAME_FEATS = ["n_det", "conf_mean", "conf_std", "topk_mean", "topk_std", "flicker", "iou_loss"]


class FrameSignal:
    """프레임이 들어올 때마다 (신뢰도 분포 4개 + 시간적 불일치 2개 + 검출 수) 특징을 만든다."""
    def __init__(self, match_iou=0.5):
        self.prev = None; self.match_iou = match_iou

    def update(self, det):
        s, tk = det["scores"], det["topk"]
        conf_mean = float(s.mean()) if len(s) else 0.0
        conf_std = float(s.std()) if len(s) else 0.0
        flicker, iou_loss = 0.0, 0.0
        if self.prev is not None and len(self.prev["boxes"]) > 0:
            iou = iou_matrix(self.prev["boxes"], det["boxes"]) if len(det["boxes"]) else np.zeros((len(self.prev["boxes"]), 0))
            same = iou * (self.prev["cls"][:, None] == det["cls"][None, :]) if iou.size else iou
            best = same.max(1) if same.size else np.zeros(len(self.prev["boxes"]))
            flicker = float((best < self.match_iou).mean())       # 직전 프레임 박스 중 이번 프레임에 대응이 없는 비율
            iou_loss = float(1.0 - best.mean())                   # 1 - 평균 최대 IoU
        self.prev = det
        return np.array([len(s), conf_mean, conf_std, float(tk.mean()), float(tk.std()), flicker, iou_loss], np.float32)


def window_features(frame_feats, cols=None):
    """frame_feats: (W, 7) → 윈도우 특징: 열별 평균·표준편차. cols로 신호 부분집합(Ablation) 선택."""
    f = np.asarray(frame_feats)
    if cols is not None:
        f = f[:, cols]
    return np.concatenate([f.mean(0), f.std(0)])


CONF_COLS = [0, 1, 2, 3, 4]       # 신뢰도 분포만
TEMP_COLS = [5, 6]                # 시간적 불일치만
ALL_COLS = list(range(7))         # 둘 다

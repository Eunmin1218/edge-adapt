"""환경(낮/밤/비/안개 등) 식별과 환경별 어댑터 저장소. 서명 = 최근 윈도우 프레임의 색 통계 6차원."""
import numpy as np, cv2


def env_signature(frames_bgr):
    t = np.stack([cv2.resize(f, (32, 20), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 for f in frames_bgr])
    g = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 100), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 for f in frames_bgr])
    m = g.mean() + 0.03; gx, gy = np.abs(np.diff(g, axis=2)).mean() / m, np.abs(np.diff(g, axis=1)).mean() / m    # 가로·세로 기울기 크기: 비 줄무늬·안개(대비 저하) 같은 질감 차이. 밝기 평균만으로는 밤과 밤+비가 같게 보인다
    return np.concatenate([t.mean((0, 1, 2)), t.std((1, 2)).mean(0), [gx, gy]])   # 평균 BGR(3) + 프레임 내 표준편차 BGR(3) + 기울기(2)


class EnvRegistry:
    def __init__(self, thr=0.08):
        self.sigs, self.adapters, self.thr, self.exp = [], [], thr, {}      # adapters[i]: 환경 i의 어댑터. {} = 기본(적응 전) 가중치, None = 아직 없음

    def match(self, sig, conf=None):
        """서명이 thr 안에 드는 환경 중 하나를 고른다. 영상 통계만으로는 구분되지 않는 환경(밤 vs 밤+비)이 있으므로,
        후보가 둘 이상이면 윈도우 평균 신뢰도(conf)가 각 환경 어댑터의 기대 신뢰도(exp)와 가장 가까운 것을 고른다 (라벨 불필요)."""
        if not self.sigs:
            return None
        d = [float(np.linalg.norm(sig - s)) for s in self.sigs]; cand = [i for i, v in enumerate(d) if v < self.thr]
        if not cand: return None
        if conf is None or len(cand) == 1: return min(cand, key=lambda i: d[i])
        return min(cand, key=lambda i: abs(conf - self.exp[i]) if i in self.exp else 9.0 + d[i])

    def add(self, sig):
        self.sigs.append(sig); self.adapters.append(None); return len(self.sigs) - 1

"""외부 라이브러리 없이 구현한 mAP. 윈도우 단위 mAP와 전체 스트림 mAP 둘 다 계산한다.
박스 형식: xyxy (픽셀). 예측: (box[N,4], score[N], cls[N]), 정답: (box[M,4], cls[M])."""
import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)


def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _ap_from_pr(rec, prec):
    """COCO 방식: 재현율 r=0,0.01,...,1.0 각각에서 'recall >= r 인 곳의 최대 정밀도'를 평균 (없으면 0)."""
    prec_env = np.flip(np.maximum.accumulate(np.flip(np.asarray(prec, float))))
    x = np.linspace(0, 1, 101)
    inds = np.searchsorted(rec, x, side="left")
    q = np.where(inds < len(prec_env), prec_env[np.minimum(inds, len(prec_env) - 1)], 0.0)
    return float(q.mean())


def evaluate(preds, gts, n_cls, iou_thrs=IOU_THRS):
    """반환: dict(map50, map5095). 정답이 하나도 없는 클래스는 평균에서 제외."""
    iou_thrs = np.asarray(iou_thrs, float)
    aps = np.full((n_cls, len(iou_thrs)), np.nan)
    for c in range(n_cls):
        n_gt = sum(int((g[1] == c).sum()) for g in gts)
        if n_gt == 0:
            continue
        recs = []                                     # (score, tp[len(iou_thrs)])
        for (pb, ps, pc), (gb, gc) in zip(preds, gts):
            pm = pc == c; gm = gc == c
            pb_c, ps_c = pb[pm], ps[pm]; gb_c = gb[gm]
            order = np.argsort(-ps_c); pb_c, ps_c = pb_c[order], ps_c[order]
            ious = iou_matrix(pb_c, gb_c)
            used = np.zeros((len(iou_thrs), len(gb_c)), bool)
            for i in range(len(pb_c)):
                tp = np.zeros(len(iou_thrs), bool)
                if len(gb_c):
                    for k, t in enumerate(iou_thrs):
                        cand = np.where(~used[k] & (ious[i] >= t))[0]
                        if len(cand):
                            j = cand[np.argmax(ious[i][cand])]; used[k, j] = True; tp[k] = True
                recs.append((ps_c[i], tp))
        if not recs:
            aps[c] = 0.0; continue
        recs.sort(key=lambda r: -r[0])
        tps = np.array([r[1] for r in recs], float)
        ctp = np.cumsum(tps, 0); cfp = np.cumsum(1 - tps, 0)
        for k in range(len(iou_thrs)):
            aps[c, k] = _ap_from_pr(ctp[:, k] / n_gt, ctp[:, k] / np.maximum(ctp[:, k] + cfp[:, k], 1e-9))
    valid = ~np.isnan(aps[:, 0])
    if not valid.any():
        return {"map50": float("nan"), "map5095": float("nan")}
    return {"map50": float(aps[valid, 0].mean()), "map5095": float(aps[valid].mean())}


def windowed_map(preds, gts, n_cls, win):
    """프레임 시퀀스를 win 프레임씩 자르고 윈도우별 mAP50 → np.array. 정답이 없는 윈도우는 nan."""
    out = []
    for s in range(0, len(preds) - win + 1, win):
        out.append(evaluate(preds[s:s + win], gts[s:s + win], n_cls, iou_thrs=[0.5])["map50"])
    return np.array(out)

"""ONNX Runtime 기반 엣지 탐지기. 어댑터(= Detect 헤드의 initializer 묶음)를 런타임에 교체한다."""
import numpy as np, cv2, onnx
from onnx import numpy_helper
import onnxruntime as ort
from .common import NET_W, NET_H, IMG_W, IMG_H, PAD_Y

TOPK_ANCHORS = 50


def preprocess(bgr):
    """bgr: (IMG_H, IMG_W, 3) uint8 → (1,3,NET_H,NET_W) float32, 위아래 회색(114) 패딩."""
    if bgr.shape[:2] != (IMG_H, IMG_W):
        bgr = cv2.resize(bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((NET_H, NET_W, 3), 114, np.uint8)
    canvas[PAD_Y:PAD_Y + IMG_H] = bgr
    x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(x[None])


def nms(boxes, scores, cls, iou_thr=0.5):
    """클래스별 NMS (좌표를 클래스마다 큰 오프셋만큼 밀어 서로 겹치지 않게 한 뒤 일반 NMS)."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    off = cls[:, None].astype(np.float32) * 10000.0
    b = boxes + off
    xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in b]
    idx = cv2.dnn.NMSBoxes(xywh, scores.astype(float).tolist(), 0.0, iou_thr)
    return np.array(idx, int).reshape(-1)


class OnnxDetector:
    def __init__(self, onnx_path, conf=0.25, providers=("CPUExecutionProvider",)):
        self.model = onnx.load(str(onnx_path))
        self.conf = conf
        self.providers = list(providers)
        self.tone = None; self._mod = set()                # 입력 톤 커브 (3,K) 또는 None / 현재 base 와 다른 값으로 바뀌어 있는 initializer 이름
        self._init_names = {i.name for i in self.model.graph.initializer}
        self._base = {i.name: numpy_helper.to_array(i).copy() for i in self.model.graph.initializer}
        self._build()

    def _build(self):
        so = ort.SessionOptions(); so.log_severity_level = 3
        self.sess = ort.InferenceSession(self.model.SerializeToString(), so, providers=self.providers)
        self.inp = self.sess.get_inputs()[0].name

    # ---- 어댑터 교체 -------------------------------------------------------------------
    def apply_adapter(self, adapter):
        """adapter: {initializer이름: ndarray}. 이름이 모델에 없으면 예외 → 호출자가 '전체 모델 교체'로 대체."""
        adapter = dict(adapter); tone = adapter.pop("__tone__", None); self.tone = None if tone is None else np.asarray(tone, np.float32)
        missing = [k for k in adapter if k not in self._init_names]
        if missing:
            raise KeyError(f"모델에 없는 initializer {len(missing)}개 (예: {missing[:3]})")
        for init in self.model.graph.initializer:
            if init.name in adapter:
                new = numpy_helper.from_array(adapter[init.name].astype(self._base[init.name].dtype), init.name); init.CopyFrom(new)
            elif init.name in self._mod:                      # 이전 어댑터가 바꿔 둔 값 중 이번 어댑터에 없는 것은 base 로 되돌린다 (어댑터마다 바뀌는 텐서 집합이 다를 수 있다)
                init.CopyFrom(numpy_helper.from_array(self._base[init.name], init.name))
        self._mod = set(adapter); self._build()

    def reset_adapter(self):
        """기본(적응 전) 가중치로 되돌린다."""
        self.apply_adapter({})

    # ---- 추론 --------------------------------------------------------------------------
    def infer(self, bgr):
        x = preprocess(bgr)
        if self.tone is not None:
            from .adapt_modules import apply_tone_np; x = apply_tone_np(x, self.tone)
        out = self.sess.run(None, {self.inp: x})[0][0]        # (4+nc, A)
        return decode(out, self.conf)


def decode(out, conf=0.25):
    """모델 출력 (4+nc, A) → dict(boxes[N,4] xyxy(IMG 좌표), scores[N], cls[N], topk[50] 앵커 최대 클래스 점수)."""
    out = out.T                                                              # (A, 4+nc)
    cs = out[:, 4:]; best = cs.max(1); topk = np.sort(best)[::-1][:TOPK_ANCHORS]
    keep = best >= conf
    xywh, sc, cl = out[keep, :4], best[keep], cs[keep].argmax(1)
    b = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                  xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1) if len(xywh) else np.zeros((0, 4))
    b[:, [1, 3]] -= PAD_Y                                                    # 패딩 제거
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, IMG_W); b[:, [1, 3]] = b[:, [1, 3]].clip(0, IMG_H)
    k = nms(b, sc, cl)
    return {"boxes": b[k].astype(np.float32), "scores": sc[k].astype(np.float32), "cls": cl[k].astype(int), "topk": topk}

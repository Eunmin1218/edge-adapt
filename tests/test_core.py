import os, tempfile
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
from src.mapeval import evaluate, windowed_map
from src.onnx_detector import OnnxDetector
from src.signals import FrameSignal, window_features, ALL_COLS
from src.selectors import STRATEGIES


def make_fake_onnx(path, seed=0):
    """YOLO 출력(1, 4+6, 260)과 같은 모양을 내는 가짜 모델. 'model.23.cv.*' 가 어댑터 역할의 initializer."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.5, (10, 3, 32, 32)).astype(np.float32); B = np.zeros(10, np.float32)
    scale = np.array([320, 208, 60, 60, 1, 1, 1, 1, 1, 1], np.float32).reshape(1, 10, 1)
    inits = [numpy_helper.from_array(W, "model.23.cv.weight"), numpy_helper.from_array(B, "model.23.cv.bias"),
             numpy_helper.from_array(scale, "scale"), numpy_helper.from_array(np.array([1, 10, 260], np.int64), "shape")]
    nodes = [helper.make_node("Conv", ["images", "model.23.cv.weight", "model.23.cv.bias"], ["c"], kernel_shape=[32, 32], strides=[32, 32]),
             helper.make_node("Sigmoid", ["c"], ["s"]), helper.make_node("Reshape", ["s", "shape"], ["r"]),
             helper.make_node("Mul", ["r", "scale"], ["output0"])]
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 416, 640])],
                          [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 10, 260])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 12)]); m.ir_version = 8
    onnx.save(m, path)


def test_map():
    gt = (np.array([[10, 10, 50, 50], [100, 100, 150, 160]], float), np.array([1, 2]))
    perfect = (gt[0].copy(), np.array([.9, .8]), gt[1].copy())
    assert abs(evaluate([perfect], [gt], 6)["map50"] - 1.0) < 1e-6
    shifted = (gt[0] + 30, np.array([.9, .8]), gt[1].copy())
    assert evaluate([shifted], [gt], 6)["map50"] == 0.0
    empty = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))
    assert evaluate([empty], [gt], 6)["map50"] == 0.0
    w = windowed_map([perfect] * 4, [gt] * 4, 6, 2); assert w.shape == (2,) and (w == 1).all()


def test_onnx_adapter_swap():
    d = tempfile.mkdtemp(); p = os.path.join(d, "m.onnx"); make_fake_onnx(p)
    det = OnnxDetector(p, conf=0.3); img = (np.random.default_rng(1).random((400, 640, 3)) * 255).astype(np.uint8)
    o1 = det.infer(img)
    ad = {"model.23.cv.weight": np.zeros((10, 3, 32, 32), np.float32), "model.23.cv.bias": np.full(10, 5.0, np.float32)}
    det.apply_adapter(ad); o2 = det.infer(img)
    assert len(o2["scores"]) != len(o1["scores"]) or not np.allclose(o1["topk"], o2["topk"])
    det.reset_adapter(); o3 = det.infer(img); assert np.allclose(o1["topk"], o3["topk"])
    try:
        det.apply_adapter({"nope": np.zeros(1)}); raise SystemExit("should fail")
    except KeyError:
        pass


def test_signal_select():
    d = tempfile.mkdtemp(); p = os.path.join(d, "m.onnx"); make_fake_onnx(p)
    det = OnnxDetector(p, conf=0.3); sig = FrameSignal(); rng = np.random.default_rng(0)
    F, U, E = [], [], []
    for _ in range(30):
        img = (rng.random((400, 640, 3)) * 255).astype(np.uint8); o = det.infer(img)
        F.append(sig.update(o)); U.append(1 - o["topk"][:20].mean()); E.append(rng.random(480))
    wf = window_features(np.stack(F), ALL_COLS); assert wf.shape == (14,) and np.isfinite(wf).all()
    c = {"uncert": np.array(U), "emb": np.stack(E), "ent": rng.random(len(U)) * 0.6, "age": np.arange(len(U))[::-1].astype(float)}
    for n, f in STRATEGIES.items():
        s = f(c, 8, rng); assert len(set(s.tolist())) == 8, n
    # EdgeSync-lite: 엔트로피가 같으면 최근 프레임(age 작음)이 우선, 나이가 같으면 엔트로피 큰 프레임이 우선
    n_ = 40; same_ent = {"ent": np.full(n_, 0.3), "age": np.arange(n_)[::-1].astype(float)}
    assert set(STRATEGIES["edgesync"](same_ent, 5).tolist()) == set(range(n_ - 5, n_))
    same_age = {"ent": np.arange(n_) / 100.0, "age": np.zeros(n_)}
    assert set(STRATEGIES["edgesync"](same_age, 5).tolist()) == set(range(n_ - 5, n_))

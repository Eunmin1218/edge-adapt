import os, tempfile
import numpy as np
from tests.test_core import make_fake_onnx
from src.onnx_detector import OnnxDetector
from src.sim import run_stream, Policy
from src.drift_model import make_labels, fit_binary, recall_false_alarm, DoCDetector


def _stream(rng):
    def mk(n, bright, env):
        return [{"img": np.clip(rng.random((400, 640, 3)) * 20 + bright, 0, 255).astype(np.uint8),
                 "gt": (np.array([[100, 100, 200, 200]], float), np.array([1])), "env": env} for _ in range(n)]
    return mk(100, 150, "clear") + mk(150, 40, "night") + mk(100, 150, "clear")      # 낮 → 밤 → 낮


class AlwaysDrift:
    def predict(self, X): return np.array([1])


def test_drift_policy_registry_switch_back():
    d = tempfile.mkdtemp(); p = os.path.join(d, "m.onnx"); make_fake_onnx(p)
    rng = np.random.default_rng(0); stream = _stream(rng); calls = []

    def server(frames, env):
        calls.append((len(frames), env)); return {"model.23.cv.bias": np.full(10, 0.5, np.float32)}
    det = OnnxDetector(p, conf=0.3)
    r = run_stream(stream, det, Policy("drift", drift_model=AlwaysDrift(), consec=2, cooldown=100), server, budget=8, win=50, apply_delay=20)
    kinds = [e[0] for e in r["events"]]
    assert "trigger" in kinds and all(c[0] == 8 for c in calls)
    assert sum(1 for c in calls if c[1] == 1) == 1, calls          # 밤 환경(env 1)은 정확히 1회만 학습
    sw = [e for e in r["events"] if e[0] == "switch"]
    assert sw and sw[-1][2] == 0, r["events"]           # 맑음 복귀 시 시작 환경(env 0)으로 재학습 없이 전환
    t_night_apply = [e[1] for e in r["events"] if e[0] == "apply"][-1]
    assert not [e for e in r["events"] if e[0] == "trigger" and e[2] == 1 and e[1] > t_night_apply]
    assert r["win_env"] == ["clear"] * 2 + ["night"] * 3 + ["clear"] * 2


def test_ams_and_periodic():
    d = tempfile.mkdtemp(); p = os.path.join(d, "m.onnx"); make_fake_onnx(p)
    rng = np.random.default_rng(0); stream = _stream(rng); n = []
    def server(frames, env): n.append(1); return {"model.23.cv.bias": np.full(10, 0.5, np.float32)}
    run_stream(stream, OnnxDetector(p, conf=0.3), Policy("ams", period=100, gain=20.0), server, budget=8, win=50, apply_delay=20, use_registry=False)
    run_stream(stream, OnnxDetector(p, conf=0.3), Policy("periodic", period=100), server, budget=8, win=50, apply_delay=20, use_registry=False)
    assert len(n) >= 2


def test_drift_model_fit():
    rng = np.random.default_rng(0)
    X = rng.random((60, 14)); wm = rng.random(60); y = make_labels(wm, 0.8, 0.3); m = fit_binary(X, y)
    assert set(recall_false_alarm(y, m.predict(X))) >= {"recall", "false_alarm"}
    doc = DoCDetector().fit(X[:10], X, y); assert doc.thr is not None


def test_edgesync_policy_triggers_on_agreement_drop_and_counts_probe_uplink():
    d = tempfile.mkdtemp(); p = os.path.join(d, "m.onnx"); make_fake_onnx(p)
    stream = _stream(np.random.default_rng(0)); calls = []

    def server(frames, env): calls.append(len(frames)); return {"model.23.cv.bias": np.full(10, 0.5, np.float32)}
    def probe(frames, dets): return 0.9 if np.mean([f.mean() for f in frames]) > 100 else 0.4           # 밤이면 Teacher 와 일치도 하락
    r = run_stream(stream, OnnxDetector(p, conf=0.3), Policy("edgesync", probe_n=4, theta=0.15, cooldown=100), server, selector="edgesync", budget=8, win=50, apply_delay=20, use_registry=False, probe_fn=probe)
    trig = [e for e in r["events"] if e[0] == "trigger"]
    assert len(trig) == 1 and trig[0][1] == 149, r["events"]            # 밤 첫 윈도우(프레임 100~149)가 끝나는 시점에 한 번만
    assert calls == [8] and not [e for e in r["events"] if e[0] == "switch"]      # 레지스트리 없음
    assert r["up_bytes"] > 7 * 4 * 1000                                  # 프로브 7윈도우 x 4장 + 학습용 8장이 업링크에 포함


def test_runner_accepts_async_handle_and_reports_state():
    d = tempfile.mkdtemp(); p = os.path.join(d, "m.onnx"); make_fake_onnx(p)
    from src.sim import StreamRunner

    class H:
        def __init__(self): self.status, self.n = "running", 0
        def ready(self):
            self.n += 1; self.status = "done" if self.n > 30 else "running"; return self.n > 30
        def result(self): return {"model.23.cv.bias": np.full(10, 0.5, np.float32)}
    r = StreamRunner(OnnxDetector(p, conf=0.3), Policy("drift", drift_model=AlwaysDrift(), consec=1, cooldown=100), lambda f, e: H(), budget=8, win=50, use_registry=False)
    states = []
    for it in _stream(np.random.default_rng(0)): r.step(it["img"], it["gt"], it["env"]); states.append(r.state)
    assert "adapting" in states and "recovered" in states and r.down > 0

import os, tempfile
import numpy as np
from tests.test_core import make_fake_onnx
from tests.test_sim import _stream, AlwaysDrift
from src.onnx_detector import OnnxDetector
from src.sim import StreamRunner, Policy
from src.demo_core import run_demo


def test_headless_three_agents_side_by_side():
    d = tempfile.mkdtemp(); p = os.path.join(d, "m.onnx"); make_fake_onnx(p); ad = {"model.23.cv.bias": np.full(10, 0.5, np.float32)}

    class H:
        status = "running"
        def __init__(self): self.n = 0
        def ready(self): self.n += 1; self.status = "done" if self.n > 20 else "running"; return self.n > 20
        def result(self): return ad
    probe = lambda fr, de: 0.9 if np.mean([f.mean() for f in fr]) > 100 else 0.4
    mk = lambda pol, **k: StreamRunner(OnnxDetector(p, conf=0.3), pol, lambda f, e: H(), win=50, budget=8, **k)
    agents = [{"name": "A", "color": (200, 200, 200), "runner": StreamRunner(OnnxDetector(p, conf=0.3), Policy("none"), None, win=50, use_registry=False)},
              {"name": "ES", "color": (60, 160, 255), "runner": mk(Policy("edgesync"), selector="edgesync", use_registry=False, probe_fn=probe)},
              {"name": "Ours", "color": (80, 220, 80), "runner": mk(Policy("drift", drift_model=AlwaysDrift(), consec=1, cooldown=100))}]
    stream = _stream(np.random.default_rng(0)); snap = os.path.join(d, "snap")
    out = os.path.join(d, "demo.mp4")
    s = run_demo(((i["img"], i["gt"], i["env"]) for i in stream), agents, win=50, n_win=7, show=False, record=out, snap_dir=snap, snap_every=100, scale=0.4)
    assert set(s) == {"A", "ES", "Ours"} and all(len(v["curve"]) == 7 for v in s.values())
    assert s["A"]["n_trigger"] == 0 and s["ES"]["n_trigger"] >= 1 and s["Ours"]["n_trigger"] >= 1
    assert s["ES"]["up_kb"] > 0 and os.path.exists(os.path.join(snap, "final.png")) and os.path.getsize(out) > 0
    if os.environ.get("DEMO_PNG"):
        import shutil; shutil.copy(os.path.join(snap, "final.png"), os.environ["DEMO_PNG"])

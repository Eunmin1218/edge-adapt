import os, tempfile
import numpy as np, cv2
from src.ui import render, STATE


def test_render_layout_and_nonuniform(tmp_path=None):
    rng = np.random.default_rng(0); fr = (rng.random((400, 640, 3)) * 255).astype(np.uint8); nw = 20
    mk = lambda v: [float(x) for x in v] + [float("nan")] * (nw - len(v))
    panels = [{"name": n, "color": c, "frame": fr, "state": s, "lines": ["윈도우 mAP50 0.61", "업 120KB · 다운 340KB"]} for n, c, s in
              [("A", (200, 200, 200), "static"), ("EdgeSync", (60, 160, 255), "adapting"), ("Ours", (80, 220, 80), "recovered")]]
    curves = [("A", (200, 200, 200), mk([.7, .7, .3, .3, .3, .3])), ("EdgeSync", (60, 160, 255), mk([.7, .7, .3, .3, .5, .6])), ("Ours", (80, 220, 80), mk([.7, .7, .3, .55, .68, .7]))]
    img = render(panels, curves, [(0, 2, "clear"), (2, 10, "night"), (10, 20, "clear")], nw, "title", "env: night", markers=[(2, (80, 220, 80), "trigger"), (4, (80, 220, 80), "apply")])
    assert img.ndim == 3 and img.shape[2] == 3 and img.shape[1] == 3 * 416 + 4 * 12 and img.std() > 20
    for ko in (True, False): assert render(panels, curves, [], nw, "t", "e", ko=ko).shape == img.shape
    assert set(STATE) >= {"static", "normal", "drift", "adapting", "offline", "recovered", "switched", "rollback", "error"}
    out = os.environ.get("UI_PNG")
    if out: cv2.imwrite(out, img)

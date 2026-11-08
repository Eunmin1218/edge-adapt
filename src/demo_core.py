"""실시간 데모 루프. 같은 프레임을 여러 에이전트(StreamRunner)에 동시에 흘리고 한 화면에 그린다.
GUI/서버 없이도 호출할 수 있도록 scripts/demo.py 와 분리했다 (tests/test_demo.py 가 헤드리스로 검증)."""
import time, json
from pathlib import Path
import numpy as np, cv2
from .common import CLASSES
from .mapeval import evaluate
from .ui import render, ENV_KO, korean_ok

BOX = [(0, 200, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255), (0, 128, 255), (255, 255, 0)]


def draw_boxes(img, det):
    v = img.copy()
    for (x1, y1, x2, y2), s, c in zip(det["boxes"], det["scores"], det["cls"]):
        cv2.rectangle(v, (int(x1), int(y1)), (int(x2), int(y2)), BOX[c % 6], 2); cv2.putText(v, f"{CLASSES[c]} {s:.2f}", (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX[c % 6], 1)
    return v


def _jsonable(x):
    """이벤트 튜플(문자열·정수·실수·리스트 혼합)을 JSON 으로 저장 가능한 값으로 바꾼다."""
    if isinstance(x, (list, tuple)): return [_jsonable(v) for v in x]
    if isinstance(x, (str, bool)) or x is None: return x
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, (float, np.floating)): return float(x)
    return str(x)


def run_demo(frames, agents, win=50, n_win=None, has_gt=True, show=True, fps=10.0, record=None, snap_dir=None, snap_every=0, max_frames=None, title=None, scale=0.65):
    """frames: 반복자 (img_bgr 400x640, gt|None, env|None).  agents: [dict(name, color, runner)].  반환: 요약 dict"""
    ko = korean_ok(); title = title or ("엣지-클라우드 환경 적응 비교 — 같은 영상, 같은 서버" if ko else "Edge-cloud adaptation: same video, same server")
    pw, ph = int(640 * scale), int(400 * scale); curves = {a["name"]: [] for a in agents}; seen = {a["name"]: 0 for a in agents}; markers, envs = [], []
    writer, paused, out, t = None, False, None, 0; ylabel = "윈도우 mAP50" if has_gt else "윈도우 평균 신뢰도"
    if not ko: ylabel = "window mAP50" if has_gt else "window mean confidence"
    snap = Path(snap_dir) if snap_dir else None
    if snap: snap.mkdir(parents=True, exist_ok=True)
    for img, gt, env in frames:
        t0 = time.perf_counter()
        for a in agents: a["runner"].step(img, gt, env)
        envs.append(env)
        if (t + 1) % win == 0:
            for a in agents:
                r = a["runner"]
                v = evaluate(r.preds[-win:], r.gts[-win:], 6, iou_thrs=[0.5])["map50"] if has_gt else float(np.mean([f[1] for f in r.fbuf[-win:]]))
                curves[a["name"]].append(float("nan") if v is None else float(v))
        for a in agents:                                                    # 새로 생긴 이벤트를 곡선 위 마커로
            ev = a["runner"].events
            for e in ev[seen[a["name"]]:]:
                if e[0] in ("trigger", "apply"): markers.append((e[1] // win, a["color"], e[0]))
            seen[a["name"]] = len(ev)
        bands, s = [], 0
        wenv = [envs[i * win] for i in range(len(envs) // win)]
        for i in range(1, len(wenv) + 1):
            if i == len(wenv) or wenv[i] != wenv[s]:
                if wenv[s]: bands.append((s, i, wenv[s]))
                s = i
        panels = []
        for a in agents:
            r = a["runner"]; c = curves[a["name"]]; cur = c[-1] if c else float("nan"); ntr = sum(e[0] == "trigger" for e in r.events)
            l0 = (f"{ylabel} {cur:.2f}" if c and not np.isnan(cur) else f"{ylabel} -") + (f"   |   적응 {ntr}회" if ko else f"   |   {ntr} adapt")
            l1 = (f"업 {r.up/1024:.0f}KB · 다운 {r.down/1024:.0f}KB" if ko else f"up {r.up/1024:.0f}KB · down {r.down/1024:.0f}KB")
            panels.append({"name": a["name"], "color": a["color"], "frame": draw_boxes(img, r.last_det), "state": r.state, "lines": [l0, l1]})
        el = ("환경(정답): " + ENV_KO.get(env, env)) if env else ("환경: 웹캠/영상" if ko else "source: camera/video")
        if not ko and env: el = "environment: " + env
        total = n_win or max(len(envs) // win, 1)
        out = render(panels, [(a["name"], a["color"], curves[a["name"]] + [float("nan")] * (total - len(curves[a["name"]]))) for a in agents], bands, total, title, el, markers, pw, ph, ylabel, ko)
        if record:
            if writer is None: writer = cv2.VideoWriter(str(record), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0]))
            writer.write(out)
        if snap and snap_every and (t + 1) % snap_every == 0: cv2.imwrite(str(snap / f"t{t + 1:05d}.png"), out)
        if show:
            cv2.imshow("edge-adapt demo", out); k = cv2.waitKey(max(1, int(1000 / fps - (time.perf_counter() - t0) * 1000))) & 0xFF
            if k == ord("q"): break
            if k == ord(" "): cv2.waitKey(0)
            if k == ord("r"):
                for a in agents: a["runner"].det.reset_adapter()
            if k == ord("s"): cv2.imwrite(f"demo_{t + 1:05d}.png", out)
        t += 1
        if max_frames and t >= max_frames: break
    if writer: writer.release()
    if snap and out is not None: cv2.imwrite(str(snap / "final.png"), out)
    summ = {}
    for a in agents:
        r = a["runner"]; c = np.array(curves[a["name"]], float); wenv = [envs[i * win] for i in range(len(c))]; sh = np.array([e not in (None, "clear") for e in wenv], bool)
        summ[a["name"]] = {"curve": c.tolist(), "mean_shift_window": float(np.nanmean(c[sh])) if sh.any() else None, "up_kb": r.up / 1024, "down_kb": r.down / 1024, "n_trigger": sum(e[0] == "trigger" for e in r.events), "events": [_jsonable(e) for e in r.events]}
    if show:
        cv2.waitKey(1)
        if out is not None:
            cv2.imshow("edge-adapt demo", out); print("끝났습니다. 아무 키나 누르면 종료 (결과 화면 유지)"); cv2.waitKey(0)
        cv2.destroyAllWindows()
    return summ

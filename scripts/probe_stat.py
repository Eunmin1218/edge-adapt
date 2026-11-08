"""통계 정렬 어댑터 점검 (toy2): 환경별 보류 프레임 mAP50 (적응 전 / lam 별), 환경 서명 간 거리. 합성 데이터이므로 성능 근거가 아니라 동작 점검이다."""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT
from src.toyworld import stream as tw_stream, make_img
from src.onnx_detector import OnnxDetector
from src.mapeval import evaluate
from src.server_core import onnx_inits
from src.stat_align import StatAligner
ap = argparse.ArgumentParser(); ap.add_argument("--lams", default="1.0,0.7,0.5"); ap.add_argument("--n", type=int, default=12); ap.add_argument("--envs", default="night,fog,rain,night_rain"); a = ap.parse_args()
E = a.envs.split(","); rng = np.random.default_rng(5); src = [make_img(rng, "clear")[0] for _ in range(60)]
t0 = time.time(); sa = StatAligner(ROOT / "runs/toy2/student.pt", src); B = onnx_inits(ROOT / "runs/toy2/student.onnx"); print(f"소스 통계 {time.time() - t0:.1f}s, 대상 합성곱 {len(sa.layers)}개", flush=True)
stream = tw_stream(7, [(e, 200) for e in E]); env = np.array([it["env"] for it in stream]); lams = [float(x) for x in a.lams.split(",")]
def mAP(det, items):
    pr = [(o["boxes"], o["scores"], o["cls"]) for o in map(det.infer, [it["img"] for it in items])]; return evaluate(pr, [it["gt"] for it in items], 6, iou_thrs=[0.5])["map50"]
sigs = {}
for e in E:
    idx = np.where(env == e)[0]; fr = [stream[i]["img"] for i in idx[:100]]; held = [stream[i] for i in idx[100:200]]
    pick = [fr[i] for i in np.linspace(0, 99, a.n).astype(int)]; base = OnnxDetector(ROOT / "runs/toy2/student.onnx"); row = {"base": mAP(base, held)}
    for lam in lams:
        t1 = time.time(); ad, sig = sa.adapter(pick, B, lam); det = OnnxDetector(ROOT / "runs/toy2/student.onnx"); det.apply_adapter(ad); row[lam] = mAP(det, held); row["sec"] = time.time() - t1
    print(e, {k: round(v, 3) for k, v in row.items()}, flush=True)
    sigs[e] = [sa.signature([stream[i]["img"] for i in idx[j:j + 50:4]]) for j in (0, 50, 100, 150)]
print("서명 거리 (행: 환경 첫 윈도우, 열: 환경 둘째 윈도우) / 같은 환경 내 거리:")
for x in E: print(f"{x:11s}", [round(float(np.linalg.norm(sigs[x][0] - sigs[y][1])), 2) for y in E], "내부", round(float(np.mean([np.linalg.norm(sigs[x][0] - sigs[x][k]) for k in (1, 2, 3)])), 2))

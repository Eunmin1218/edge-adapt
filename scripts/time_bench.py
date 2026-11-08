"""엣지 연산·종단 적응 시간 실측 (CPU 1스레드 기준, 같은 기계의 Student ONNX 추론 시간을 단위로 삼아 비율도 낸다).
다른 학습/평가 작업이 CPU 를 쓰는 동안 실행하면 값이 왜곡되므로 단독 실행할 것."""
import sys, time, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.toyworld import make_img
from src.onnx_detector import OnnxDetector
from src.stat_align import StatAligner
from src.server_core import onnx_inits
W = ROOT / "runs/toy3"; S_PT = W / "student.pt"; rg = np.random.default_rng(5)
fr = [make_img(rg, "night")[0] for _ in range(12)]; src = [make_img(rg, "clear")[0] for _ in range(60)]
det = OnnxDetector(W / "student.onnx")
def tm(f, n=3):
    f(); t = []
    for _ in range(n): t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
    return float(np.median(t))
t_inf = tm(lambda: [det.infer(x) for x in fr[:6]]) / 6
sa = StatAligner(S_PT, src); B = onnx_inits(W / "student.onnx")
t_fit = tm(lambda: sa.fit(fr[:5]))                 # 서명 5프레임
t_adapt = tm(lambda: sa.adapter(fr, B, 1.0), 2)    # 12프레임 정렬 + 접기 + ONNX 내보내기
t_probe = 3 * 12 * t_inf                           # 후보 3개 x 12프레임 (엣지 추론; Teacher 는 서버 쪽 상자를 쓰므로 제외)
out = dict(infer_ms=t_inf * 1e3, sig_ms=t_fit * 1e3, adapter_ms=t_adapt * 1e3, probe_ms=t_probe * 1e3)
out["edge_total_s"] = t_fit + t_adapt + t_probe
# ES: 업로드 + 서버 학습(실측 sec 는 로그의 [server] sec; 여기선 인자로) + 다운로드. 대역폭별 종단 시간
srv_sec = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0; up_kb = float(sys.argv[2]) if len(sys.argv) > 2 else 400.0; dn_kb = float(sys.argv[3]) if len(sys.argv) > 3 else 70.0
out["server_sec"] = srv_sec
for mbps in (1, 5, 20):
    out[f"ES_end2end_s@{mbps}Mbps"] = up_kb * 8 / 1000 / mbps + srv_sec + dn_kb * 8 / 1000 / mbps
    out[f"ours_edge_s@{mbps}Mbps"] = out["edge_total_s"]
print(json.dumps(out, indent=1)); json.dump(out, open(ROOT / "results" / "time_bench.json", "w"), indent=1)

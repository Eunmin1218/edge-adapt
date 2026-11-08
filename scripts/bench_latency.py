"""엣지(노트북) 추론 지연·모델 크기 측정. 사용: python scripts/bench_latency.py models/student.onnx [--threads 4]
출력: 모델 파일 MB, 전처리+추론+후처리 전체 ms/프레임의 평균·p50·p95, 순수 session.run 평균 ms."""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.onnx_detector import OnnxDetector, preprocess
import onnxruntime as ort

ap = argparse.ArgumentParser(); ap.add_argument("onnx"); ap.add_argument("--n", type=int, default=200); ap.add_argument("--threads", type=int, default=0); a = ap.parse_args()
det = OnnxDetector(a.onnx)
if a.threads:
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; so.log_severity_level = 3
    det.sess = ort.InferenceSession(det.model.SerializeToString(), so, providers=["CPUExecutionProvider"])
img = (np.random.default_rng(0).random((400, 640, 3)) * 255).astype(np.uint8)
for _ in range(10): det.infer(img)                                   # 워밍업(첫 실행은 느리다)
tot, run = [], []
for _ in range(a.n):
    x = preprocess(img); t1 = time.perf_counter(); det.sess.run(None, {det.inp: x}); t2 = time.perf_counter(); det.infer(img); t3 = time.perf_counter()
    run.append(t2 - t1); tot.append(t3 - t2)
tot, run = np.array(tot) * 1000, np.array(run) * 1000
print(f"모델 {Path(a.onnx).stat().st_size/2**20:.2f} MB | 전체 평균 {tot.mean():.1f} ms (p50 {np.percentile(tot,50):.1f}, p95 {np.percentile(tot,95):.1f}) | session.run 평균 {run.mean():.1f} ms | 환산 {1000/tot.mean():.1f} FPS")

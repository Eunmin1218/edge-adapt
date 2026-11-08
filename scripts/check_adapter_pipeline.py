"""어댑터 파이프라인 점검 (서버에서 Student 학습 직후 1회 실행). 사용: python scripts/check_adapter_pipeline.py [models/student.pt]
점검 항목
  (1) 같은 .pt 를 두 번 ONNX 로 내보내면 initializer 이름·값이 동일한가           → 같아야 '값이 바뀐 것 = 어댑터'로 정의 가능
  (2) Detect 헤드 가중치만 흔든 모델을 내보내면, 바뀐 initializer 가 헤드에서만 나오는가
  (3) 그 어댑터를 base ONNX 에 OnnxDetector.apply_adapter 로 끼우면 흔든 모델의 ONNX 출력과 일치하는가  (엣지 교체 경로 전체 검증)
  (4) 어댑터 크기(FP32/FP16)와 5MB 목표 충족 여부"""
import sys, tempfile
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO
from src.common import ROOT
from src.server_core import export_onnx, onnx_inits, save_pt
from src.onnx_detector import OnnxDetector, preprocess

pt = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models/student.pt"
tmp = Path(tempfile.mkdtemp()); n = len(YOLO(str(pt)).model.model) - 1
base1 = export_onnx(pt, tmp / "base1.onnx"); b1 = onnx_inits(base1)
base2 = export_onnx(pt, tmp / "base2.onnx"); b2 = onnx_inits(base2)
same = set(b1) == set(b2) and all(np.array_equal(b1[k], b2[k]) for k in b1)
print(f"(1) 재현성: initializer {len(b1)}개, 이름·값 동일 = {same}"); assert same, "export 마다 이름/값이 달라짐 → 어댑터 정의 불가. ultralytics/torch 버전을 requirements 대로 맞추세요"

m = YOLO(str(pt)); torch.manual_seed(0)
for name, p in m.model.named_parameters():
    if name.startswith(f"model.{n}.") and ".dfl." not in name: p.data += 0.05 * torch.randn_like(p)       # 헤드만 흔든다
save_pt(m, tmp / "perturbed.pt"); pert = export_onnx(tmp / "perturbed.pt", tmp / "perturbed.onnx"); p1 = onnx_inits(pert)
changed = {k: v for k, v in p1.items() if k in b1 and v.shape == b1[k].shape and not np.array_equal(v, b1[k])}
print(f"(2) 바뀐 initializer {len(changed)}개 / 전체 {len(b1)}개, 예: {list(changed)[:3]}")
assert changed, "바뀐 initializer 가 0개: ONNX 가 BN 융합·상수 폴딩으로 헤드 이름을 바꿨거나 head_idx 가 틀림"
assert len(changed) < len(b1), "전부 바뀜: 백본까지 달라졌다는 뜻(어댑터가 아님)"

img = (np.random.default_rng(0).random((400, 640, 3)) * 255).astype(np.uint8)
d_base, d_pert = OnnxDetector(base1, conf=0.01), OnnxDetector(pert, conf=0.01)
d_base.apply_adapter(changed)
o_a = d_base.sess.run(None, {d_base.inp: preprocess(img)})[0]
o_b = d_pert.sess.run(None, {d_pert.inp: preprocess(img)})[0]
err = float(np.abs(o_a - o_b).max()); print(f"(3) base+어댑터 vs 흔든 모델 출력 최대 오차 = {err:.2e}"); assert err < 1e-4, "어댑터 교체 결과가 다름"
mb32 = sum(v.nbytes for v in changed.values()) / 2**20
print(f"(4) 어댑터 크기: FP32 {mb32:.2f} MB, FP16 {mb32/2:.2f} MB (목표 5MB 이하: {'충족' if mb32/2 <= 5 else '초과 → 헤드 일부만 학습하도록 줄이기'})")
print("OK")

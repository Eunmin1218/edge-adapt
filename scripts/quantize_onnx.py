"""경량화 실험: ONNX 정적 INT8 양자화(QDQ). 사용: python scripts/quantize_onnx.py models/student.onnx data/yolo/stream_train/images
주의: 양자화 모델은 initializer 가 INT8로 바뀌어 어댑터 교체가 불가능하다 → 적응 파이프라인은 FP32 기본 모델로 돌리고,
이 스크립트는 '엣지 경량화 효과(크기·지연·mAP)'를 따로 보고하는 용도이다."""
import sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.onnx_detector import preprocess
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

src, img_dir = Path(sys.argv[1]), Path(sys.argv[2]); out = src.with_name(src.stem + "_int8.onnx")
allf = sorted(img_dir.glob("*.jpg")); files = allf[:: max(1, len(allf) // 100)][:100]      # 캘리브레이션 약 100장 균등 추출


class Reader(CalibrationDataReader):
    def __init__(self): self.it = iter(files)

    def get_next(self):
        f = next(self.it, None); return None if f is None else {"images": preprocess(cv2.imread(str(f)))}


quantize_static(str(src), str(out), Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
print(f"저장: {out} ({out.stat().st_size/2**20:.2f} MB, 원본 {src.stat().st_size/2**20:.2f} MB)")

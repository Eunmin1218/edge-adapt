"""환경 점검. 서버: python scripts/00_check_env.py --server / 엣지 노트북: python scripts/00_check_env.py"""
import sys, platform

server = "--server" in sys.argv
print("python", platform.python_version(), platform.platform())
import numpy, cv2, onnx, onnxruntime, sklearn
print("numpy", numpy.__version__, "| opencv", cv2.__version__, "| onnx", onnx.__version__, "| onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers(), "| sklearn", sklearn.__version__)
if server:
    import torch, ultralytics
    print("torch", torch.__version__, "| CUDA 사용 가능:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU 없음")
    print("ultralytics", ultralytics.__version__)
    if not torch.cuda.is_available(): print("경고: NVIDIA GPU 를 못 찾았습니다. CPU 로 동작은 하지만 서버 적응 한 번이 오래 걸립니다 (GUIDE 0.5절).")
print("OK")

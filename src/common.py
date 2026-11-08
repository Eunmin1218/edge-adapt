"""공통 상수와 경로. 모든 스크립트가 여기서 import 한다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # edge-adapt/
DATA = ROOT / "data"                                 # Git에 올리지 않는다(.gitignore)
CLASSES = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]   # SHIFT 원본 6클래스
CLS2ID = {c: i for i, c in enumerate(CLASSES)}

IMG_W, IMG_H = 640, 400          # 학습/추론/지연측정 공통 해상도 (원본 1280x800의 1/2)
NET_W, NET_H = 640, 416          # ONNX 입력 크기: 높이를 32의 배수로 맞추기 위해 위·아래 8px 패딩
PAD_Y = (NET_H - IMG_H) // 2     # = 8
MIN_BOX_PX = 8                   # 640x400 기준 최소 박스 변(px). 9주차 측정 후 11주차에 확정

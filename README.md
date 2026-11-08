# edge-adapt — 환경 변화에 스스로 적응하는 엣지-클라우드 협력형 실시간 영상 분석

엣지(노트북, ONNX Runtime)는 추론·드리프트 감지·프레임 선별만 수행하고, 서버(GPU)는 Teacher 의사 라벨로 채널별 affine 어댑터(최종 구성 V7q, GUIDE 5.6.6)를 학습해 4비트로 압축해 돌려보낸다. 엣지는 통계 정렬(NEO/AdaBN 계열)을 로컬에서 먼저 적용한다.
**처음이면 `QUICKSTART.md`**(환경 설치부터 5분 데모까지, 복사·붙여넣기)를 먼저 따른다. 전체 진행 순서와 근거는 **GUIDE.md** 를 따른다.

## 폴더
- `src/` 라이브러리
  - `mapeval`(mAP) `onnx_detector`(ONNX 추론 + 어댑터 교체) `signals`(드리프트 신호) `selectors`(프레임 선별 4종)
  - `drift_model`(감지기·DoC) `env`(환경 식별) `sim`(스트림 시뮬레이션) `stream`(평가 스트림 로더)
  - `server_core`(Teacher 의사 라벨 + 어댑터 학습) `torch_detector`(TENT/BN-Adapt) `api`·`edge_client`(FastAPI 통신)
- `scripts/` 실행 스크립트, `tests/` 데이터·GPU 없이 도는 단위 테스트
- `configs/` 스트림 정의(`streams.json`)와 YOLO 학습 설정(`student.yaml`, `teacher.yaml`) — 스크립트가 생성

## 테스트 (GPU·데이터 불필요)
    python -m pytest tests -q
    python scripts/toy_e2e.py --quick     # CPU 통합 시험: 합성 영상으로 학습→어댑터→A/B/EdgeSync-lite/제안 비교를 끝까지 실행 (성능 근거 아님)

## 주요 진입점
- `colab/edge_adapt_colab.ipynb` 무료 Colab GPU에서 데이터 준비→학습→어댑터 튜닝→본 실험을 한 번에 (GUIDE 0.5절)
- `scripts/tune_adapter.py` 어댑터 설정을 개발용 스트림에서 측정해 `configs/adapter.json` 에 저장 (GUIDE 5.6절)
- `scripts/demo.py` 3-way 실시간 데모(적응 없음 / EdgeSync-lite / 제안) (GUIDE 8.3절)

## 라이선스 유의
- SHIFT: CC BY-NC-SA 4.0 (비영리 연구용, 원본 재배포 금지) — `data/` 는 .gitignore 처리됨
- Ultralytics YOLO: AGPL-3.0 → 저장소를 공개할 때 AGPL-3.0 호환 라이선스를 적용한다.

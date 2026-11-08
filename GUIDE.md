# 졸업작품 실행 가이드

**과제**: 환경 변화에 스스로 적응하는 엣지-클라우드 협력형 실시간 영상 분석 시스템 (1분반 1조)
**기준 문서**: `AI_P-실무프로젝트_과제계획서_최종수정본.docx`
**작성일**: 2026-11-05 (계획서 기준 10주차)

이 문서의 명령은 모두 이 저장소(`edge-adapt/`)의 파일과 인자 이름에 맞춰 적었다. 순서대로 따라 하면 계획서의 목표 1·2·3을 측정하고 판정(`scripts/check_goals.py`)하는 데까지 간다.

---

## 0. 먼저 읽을 것

### 0.1 검증 상태

| 구분 | 내용 |
|---|---|
| **실행해서 확인함** (단위 테스트 25개 통과, `python -m pytest tests -q`; Python 3.11 + torch 2.5.1 CPU + ultralytics 8.3.40 + onnx 1.17.0 + onnxruntime 1.20.1, 2코어 CPU) | mAP 계산, ONNX 어댑터 교체·복원, 신호 추출, 프레임 선별, 시뮬레이터, FastAPI 업로드·다운로드·단절 시 큐 보관, SHIFT→YOLO 변환, 감지기 학습·분석 스크립트, **Student/Teacher 학습, Teacher 의사 라벨, Detect 헤드 어댑터 추출(`check_adapter_pipeline`), `AdapterServer` 실제 학습, EdgeSync-lite 프로브(`agreement`), TENT/BN-Adapt 실행, `tune_adapter.py`, `scripts/toy_e2e.py`(합성 데이터 A/T/B/ES/D 전체 비교)** |
| **합성 데이터 통합 시험의 결과** (성능 근거가 아니라 "끝까지 동작하고 방향이 맞는지"만 보는 값. 최종 설정: 증류 학습, 감지 확정 1개 윈도우) | 변화구간 mAP50: A 0.399, T 0.997, B(주기) 0.583, EdgeSync-lite 0.714, 제안(D) 0.698, 제안(확정 2개 윈도우) 0.549. 업링크: B 920KB, ES 1268KB, D 343KB. 다운링크: ES·D 824KB. 적응 1회 약 35초(2코어 CPU, 프레임 8장, 60스텝) |
| **실행하지 못함** | `demo.py`의 실제 창 표시(`cv2.imshow`; 렌더링 결과는 PNG로 확인함), 웹캠·영상 파일 입력, 실제 SHIFT 파일의 열 이름·경로, 실제 규모(수백 프레임·640 해상도)에서의 서버 소요 시간과 성능, 사용자 노트북 CPU(Ryzen급)에서의 적응 시간(`DELAY` 를 이 값으로 정한다) |
| **Colab 노트북 `colab/edge_adapt_colab.ipynb`** | 셀의 구문(파이썬·bash)만 점검했고 **Colab에서 실행하지 못했다.** 첫 실행에서 실패할 수 있다 |
| **확인하지 못함** | 현재 시점에 Hugging Face 미러의 파일 구성, SHIFT `seq.csv`의 열 이름, `det_2d.json`의 필드 |

실행하지 못한 부분 때문에 **처음 한 번은 막힐 수 있다.** 이를 위해 3개의 점검 지점을 넣었다.

1. `python scripts/inspect_shift.py ...` — 실제 SHIFT 파일 구조 확인 (4.4절)
2. `python scripts/vis_check.py ...` — 변환된 박스가 이미지와 맞는지 눈으로 확인 (4.7절)
3. `python scripts/check_adapter_pipeline.py` — **서버에서 가장 먼저 돌릴 점검** (5.4절). 어댑터 추출이 맞는지 수치로 판정한다. 통과 전에는 실험(6장)에 들어가지 않는다.

### 0.1.1 같은 전송량에서 성능을 높이는 기법의 합성 시험 (`scripts/ablate_adapter.py`)

적응 전 보류 프레임 mAP50 0.439(합성 '밤'), 프레임 8장, 어댑터 전송량 약 780KB(FP16, 헤드만)로 고정하고 방법만 바꿨다. 값은 보류 프레임 mAP50(시드 2~4개 평균)이다. **합성 데이터이므로 방향을 보는 용도이며 실제 SHIFT 값이 아니다.**

| 바꾼 것 | 결과 | 판단 |
|---|---|---|
| 서버 학습: 하드 라벨(ultralytics) → Teacher 조밀 출력 증류(`src/distill.py`, 분류 소프트 타깃 + 박스 분포 KL) | 0.691 → 0.752 (8장), 0.795 → 0.819 (16장), 적응 시간 64초 → 38초 | 채택. 이득은 크지 않다(시드 편차 ±0.03) |
| 학습 스텝 수 (증류, lr 1e-2) | 30스텝 0.602, 60스텝 0.752, 120스텝 0.67~0.73 | 총 60스텝 부근이 최적(`batch 0` 자동) |
| 학습률 | 3e-2 에서 0.497 로 붕괴 | 1e-2 이하 |
| 증강(색·밝기·잡음) / 하드 라벨 손실 추가 | 0.752 vs 0.737(증강 끔) / 0.725~0.747 | 효과가 시드 편차 이내. 기본은 증강 켬, 하드 손실 끔 |
| **프레임 선별 방식** | 불확실성 0.510, hybrid(불확실성→다양성) 0.741, 엔트로피 0.875, 랜덤 0.888, 엔트로피→다양성 0.923, **다양성 0.963**, 엔트로피+최신성(EdgeSync식) 0.975 | **가장 큰 효과.** 기존 기본(hybrid)이 학습에 불리한 프레임을 뽑고 있었다. 단 합성 데이터의 성질일 수 있어 `tune_adapter.py`가 실제 데이터로 고른다 |
| 델타 부호화(`src/delta_codec.py`): int8 델타 | 정확도 손실 없음(최대 오차 1.4e-4), 용량 절감은 작음 | 모듈만 포함, 전송 경로에는 연결하지 않음 |
| 델타 희소화(상위 10~40%만 전송) | 0.435, 0.542 로 성능 하락 | 비채택 |
| 학습 레이어 확대 + int8 델타 | 3개 층 0.753(헤드만과 동일, +26KB), 6개 층 0.491(불안정) | 이득 없음. `train_layers 0` 기본 |

참고 문헌에서 가져온 아이디어: 지식 증류의 소프트 타깃(Hinton 계열), weak-to-strong 증강(Unbiased Teacher 계열), 갱신량 희소화·양자화(ComPEFT, arXiv 2311.13171), 연속 TTA의 모니터링·복원(AMROD, arXiv 2406.16439). 이 중 앞의 둘만 채택했고 AMROD의 클래스별 임계값·복원은 구현하지 않았다.

### 0.2 용어 (이 문서 전체에서 이 뜻으로만 쓴다)

| 용어 | 정의 |
|---|---|
| 엣지 | 영상을 입력받아 추론하는 쪽. 이 과제에서는 노트북 + ONNX Runtime |
| 서버 | Teacher 의사 라벨과 어댑터 학습을 수행하는 쪽. **이 팀에서는 노트북(32GB RAM / 1TB)이 서버 역할을 겸한다** (0.5절). 이전 판에 적었던 "RTX 6000 Ada 서버"는 확인되지 않은 가정이었고 삭제한다 |
| Student / Teacher | 엣지에서 쓰는 작은 탐지 모델(YOLO11n) / 서버에서 의사 라벨을 만드는 큰 모델(YOLO11x) |
| 의사 라벨 | 사람이 단 라벨이 아니라 Teacher의 예측 결과를 라벨로 쓴 것. 신뢰도 임계값 `tau` 미만은 버린다 |
| 어댑터 | 이 구현에서는 **Student의 마지막 레이어(Detect 헤드) 가중치 전체**. 서버가 이것만 학습하고, 엣지는 이것만 받아 교체한다 (0.4절 참고) |
| 드리프트 | 환경 변화(비·밤·안개)로 정확도가 떨어진 상태. 라벨 없이 추론 통계로 감지한다 |
| 윈도우 | 연속 50프레임(10Hz 기준 5초) 묶음. 윈도우마다 mAP50을 계산하고 드리프트를 판정한다 |
| mAP50 | IoU 0.5 기준 mean Average Precision. 이 가이드의 "mAP"는 별도 표기가 없으면 mAP50. (mAP50-95는 결과 JSON의 `map_all`에 함께 저장됨) |
| 예산 | 한 번의 적응에서 서버로 보내는 프레임 수 (기본 8 / 16 / 32) |
| 레지스트리 | 환경별 어댑터를 저장해 두고, 이미 본 환경으로 돌아오면 재학습 없이 전환하는 저장소 |

### 0.3 일정 (계획서 WBS를 날짜로 환산, 9/1 기준 7일 단위)

| 주차 | 기간 | 계획서의 할 일 | 이 가이드의 장 |
|---|---|---|---|
| 9 | 10/27~11/2 | SHIFT 서브셋, YOLO 라벨, Student·Teacher 학습, 환경별 mAP 하락 측정 | 3~5장, 6.1 |
| **10 (오늘 11/5)** | 11/3~11/9 | 시뮬레이션 파이프라인, Baseline A·B | 6.1~6.4 |
| 11 | 11/10~11/16 | 감지기 학습(RQ1), Baseline C, **1차 멘토링에서 수치 목표 확정** | 6.2~6.5 |
| 12 | 11/17~11/23 | 전송 예산 3단계 확정, 선별 4종 1차 실험 | 6.6 |
| 13 | 11/24~11/30 | RQ1 평가, RQ2 본 실험, A~D 종합 비교, 2차 멘토링 | 6.7~6.8 |
| 14 | 12/1~12/7 | 노트북 실시간 데모, 결과 분석 | 8장 |
| 15 | 12/8~12/14 | 확장 목표(선택), 결과 정리 | 9장 |
| 16 | 12/15~12/21 | 최종보고서, GitHub 공개, 시연 | 9장 |

**일정 위험**: 9주차 항목(데이터 다운로드, Teacher 학습)이 아직 끝나지 않았다면 이 두 작업이 가장 긴 선행 작업이다. 둘은 서로 독립이므로 동시에 진행한다(팀원 분담은 1장). Teacher 학습 시간을 줄이는 선택지(작은 모델, 적은 epoch)는 5.3절에 적었지만, **범위를 줄이는 결정은 팀과 멘토가 정한다** — 이 가이드는 기본값을 계획서 그대로 둔다.

### 0.4 계획서와 달라지거나 팀이 결정해야 하는 점

| # | 내용 | 영향 | 해야 할 일 |
|---|---|---|---|
| 1 | **연속(continuous) SHIFT 데이터를 확보하지 못할 수 있다.** 공식 서버(`dl.cv.ethz.ch`)가 내려갔다는 GitHub 이슈가 있고([#80](https://github.com/SysCV/shift-dev/issues/80)), 같은 이슈에 올라온 Hugging Face 미러에는 이산(discrete) 10Hz 비디오만 있고 `continuous/` 폴더가 없다는 보고가 있다. 이 가이드는 이산 시퀀스(500프레임, 약 50초)를 환경 순서대로 이어 붙인 스트림을 기본으로 한다 | 환경 전환이 점진적이지 않고 시퀀스 경계에서 갑자기 바뀐다. 계획서 4.3.2의 "연속 시퀀스" 표현과 다르다 | 연속 데이터를 구하면(공식 서버 복구, 다른 미러) `configs/streams.json`의 시퀀스만 교체. 못 구하면 보고서 한계에 명시 |
| 2 | **"어댑터"의 정의.** 계획서 4.1.4는 참고문헌 [10](Houlsby 어댑터, 병목 모듈 삽입 방식)을 인용한다. 이 구현의 어댑터는 병목 모듈 삽입이 아니라 **Detect 헤드만 학습**하는 방식이다 | 5MB 목표는 헤드 가중치(FP16) 크기로 측정한다 | 보고서에 "헤드 어댑터 = 마지막 레이어만 학습하는 parameter-efficient 방식"으로 정의하거나, 병목 모듈 방식을 따로 구현(ONNX 그래프 수정 필요, 이 저장소에는 없음) |
| 3 | 모델은 **YOLO11, Ultralytics 8.3.40으로 고정**. 최신 YOLO26은 NMS-free end-to-end 추론을 기본 경로로 내세우고 있어 이 코드의 출력 형식 가정(원시 출력 `(4+클래스, 앵커)`)과 달라질 수 있다 | 계획서의 "Ultralytics YOLO"와 일치 | 버전을 올리지 않는다 |
| 4 | 환경 식별은 프레임 색 통계 6차원 + 거리 임계값이다 | 시퀀스마다 맑은 환경의 색이 다르면 같은 환경으로 인식하지 못할 수 있다 | 6.3절의 `env_sig_check.py`로 판정 |
| 5 | 기존 연구 재구성은 **정책 수준의 "-lite" 버전**이다 (7장) | 원 논문의 시스템 전체를 재현하는 것이 아니다 | 보고서에 "-lite"로 표기 |

### 0.5 하드웨어와 실행 장소: 노트북 한 대(32GB / 1TB, NVIDIA GPU 없음) + 무료 Colab

**확인된 사양**: 노트북에 AMD Radeon 860M(내장 그래픽)만 있고 NVIDIA GPU는 없다. 이 가이드는 이 iGPU를 PyTorch 학습·추론 가속에 쓰지 않는다(PyTorch/Ultralytics의 GPU 경로는 CUDA 기준이며, Windows에서 이 iGPU를 PyTorch에 쓰는 공식 경로는 확인하지 못했다). ONNX Runtime의 DirectML로 엣지 추론을 가속할 수는 있으나 검증하지 않아 쓰지 않는다. 따라서 **노트북은 CPU로만 계산한다.** GPU는 **무료 Colab T4**를 쓴다.

**무엇을 어디서 하나**

| 작업 | 장소 | 이유 |
|---|---|---|
| SHIFT 다운로드, 시퀀스 선택, 프레임 추출, YOLO 변환 | **Colab** | img.tar가 크고(크기 미확인, 4.1절 `--dry-run`), Colab의 네트워크가 빠르며, 노트북 디스크·시간을 쓰지 않는다 |
| Student·Teacher 학습 | **Colab GPU** | CPU로는 학습 시간이 비현실적이다(미측정이지만 GPU 대비 수십 배 이상 느리다는 것이 일반적이다) |
| Baseline A·T·C(TENT), 감지기 학습, 본 실험(`run_grid.sh`) | **Colab GPU** | 서버 적응이 수십~수백 회 일어나므로 GPU가 필요하다. 실시간이 아니어도 되는 작업이다 |
| **실시간 데모**(3-way 화면) | **노트북** | 발표 현장에서 돌려야 한다. 서버(Teacher 의사 라벨 + 헤드 어댑터 학습)도 같은 노트북에서 **CPU**로 돈다 |
| 팀원 협업(코드, 결과 표) | GitHub | 3장 |

**Colab 원큐 노트북**: `colab/edge_adapt_colab.ipynb` (4~6장의 명령을 순서대로 담았다). 사용법은 아래.

**무료 Colab의 제약** (Google이 정책을 바꿀 수 있고 한도는 공개되지 않아 시간은 단정하지 않는다)
- GPU 할당이 보장되지 않는다. 사용량이 많으면 몇 시간 뒤 끊기거나, 한동안 GPU를 못 받을 수 있다. 노트북은 **끊겨도 이어지도록** 만들었다: 학습은 `last.pt`에서 자동 이어서, 실험은 이미 있는 결과를 건너뛰며, 결과·모델은 구글 드라이브에 저장된다.
- 실행 중 브라우저 탭을 닫거나 노트북이 절전에 들어가면 세션이 끊길 수 있다. 전원 연결, 절전 끔, 탭 유지. (끊김을 막는 자동 클릭 같은 우회 스크립트는 쓰지 않는다. 무료 서비스 약관상 문제가 될 수 있다.)
- RAM 약 12GB, 드라이브 무료 용량 약 15GB. 큰 파일(원본 tar, 추출 프레임)은 드라이브에 저장하지 않는다. **위험(미확인)**: SHIFT의 `det_2d.json`이 매우 크면 `shift_to_yolo.py`가 이를 통째로 읽으므로 12GB RAM에서 메모리 부족이 날 수 있다. 그 경우 노트북의 32GB RAM에서 이 변환 단계만 실행하는 방법이 있다(그러려면 json과 이미지 일부를 노트북으로 받아야 한다). 오류가 나면 메시지를 공유한다.
- 그래서 노트북의 기본 규모는 **`PROFILE = "free"`** 로 줄여 두었다.

| | `free`(기본) | `full`(계획서 값) |
|---|---|---|
| Student용 시퀀스 / Teacher용 시퀀스 | 200 / 450 | 300 / 750 |
| Student epoch / Teacher epoch | 30 / 15 | 50 / 30 |
| 본 실험 예산 단계 / 시드 | 16 한 가지 / 0 한 개 | 8·16·32 / 0·1·2 |

`free`는 **무료 한도를 고려한 임시 축소**이므로 (1) 결과를 보고서에 "무료 Colab 한도로 축소해 수행"이라고 적고, (2) 목표 2(예산 3단계 중 2단계 이상에서 Random보다 우수)는 예산이 한 단계뿐이라 **판정할 수 없다.** 세션이 끊기지 않고 시간이 남으면 `full`로 바꿔 다시 돌린다. 어느 규모로 갈지는 팀이 정한다. `full`로 돌리려면 노트북에서 `PROFILE = "full"` 로 바꾸고 드라이브 연결 셀부터 다시 실행한다(데이터도 `full`용으로 다시 만든다).

**Colab 노트북 사용법 (처음부터 끝까지)**
1. 브라우저에서 https://drive.google.com 에 로그인 → 왼쪽 위 **+ 새로 만들기** → **새 폴더** → 이름 `edge-adapt` → **만들기**. 폴더를 더블클릭해 열고, 받은 `edge-adapt.zip` 파일을 폴더 안으로 끌어다 놓는다.
2. 같은 폴더에 `colab/edge_adapt_colab.ipynb` 도 올린다 (zip을 풀어 `colab` 폴더 안의 파일을 올린다). 올린 `.ipynb` 파일을 **더블클릭**하면 미리보기가 열린다. 위쪽 **Google Colaboratory로 열기**(또는 **연결할 앱 → Google Colaboratory**)를 클릭한다. 목록에 없으면 같은 위치의 **연결할 앱 더 보기** 에서 Colaboratory를 설치한다.
3. Colab 화면에서 메뉴 **런타임 → 런타임 유형 변경 → 하드웨어 가속기: T4 GPU → 저장**.
4. 첫 셀 왼쪽의 **▶ 버튼**을 누른다. 오래 도는 셀은 끝날 때까지 기다린다(셀 왼쪽의 원이 돌고 있으면 실행 중). 각 셀 위의 설명에 **확인할 것**이 있다. 확인한 뒤 다음 셀로 간다.
5. 끝나면 드라이브 `edge-adapt` 폴더에 `laptop_bundle.zip`이 생긴다. 이것을 노트북으로 다운로드(우클릭 → **다운로드**)해 저장소 폴더에 푼다.

**노트북(CPU 서버 + 엣지) 준비** (Colab 학습이 끝난 뒤, Windows 기준)
1. 2.3절대로 Python, Git, 가상환경을 준비한다. 가상환경을 켠 뒤:
   ```
   pip install -U pip
   pip install -r requirements-edge.txt
   pip install torch==2.5.1 torchvision==0.20.1
   pip install ultralytics==8.3.40 fastapi "uvicorn[standard]" python-multipart onnxslim
   ```
   (Windows의 PyPI `torch`는 CPU 빌드다. 설치가 오래 걸릴 수 있다. 실패하면 오류 전체를 공유한다.)
2. `laptop_bundle.zip`을 저장소 폴더에서 풀어 `models/`, `configs/`, `data/`, `results/` 에 파일이 생기게 한다 (우클릭 → **모두 압축 풀기** → 대상 폴더를 저장소 폴더로).
3. `models/student_best.pt`로 **ONNX를 노트북에서 다시 내보낸다.** (Colab과 노트북의 torch 버전이 다르면 ONNX 내부 이름이 어긋나 어댑터가 맞지 않을 수 있어서, 서버와 엣지가 **같은 환경에서 만든 ONNX**를 쓰게 한다.)
   ```
   python scripts/export_base.py models/student_best.pt
   python scripts/check_adapter_pipeline.py
   ```
   `check_adapter_pipeline.py`의 (1)~(3)이 통과해야 한다.
4. 노트북 CPU에서 **적응 1회 시간을 잰다**: 
   ```
   python scripts/run_experiment.py --split eval --method D --selector diversity --budget 16 --seed 0 --delay 100 --device cpu --epochs 5 --tag D_cpu_try
   ```
   출력의 `[server] {'sec': …}`가 이 노트북의 적응 1회 시간이다. 이 값이 크면 데모에서 `서버 적응 중...`이 그만큼 오래 보인다. 줄이려면 `--epochs`를 줄이거나 `--budget 8`로 한다. `DELAY = ceil(sec×10)`을 Colab 본 실험 셀에 적어 다시 돌리면 실험과 데모의 지연이 같아진다.
5. 데모 중 서버가 같은 CPU를 쓰므로 적응 때 영상이 잠깐 느려진다. 발표 때 한계로 말한다. 느리면 `--agents static,ours` 또는 `--scale 0.5 --fps 5`.

---

## 1. 전체 구조와 팀 분담

```
[엣지 역할]                                            [서버 역할]   (같은 노트북이어도 된다: 주소 127.0.0.1)
 프레임 ─▶ ONNX Student(+어댑터) ─▶ 박스·신뢰도
              │ 신호(신뢰도 분포, 프레임 간 박스 불일치)
              ▼
         드리프트 감지기 ─(연속 2회 "적응 필요")─▶ 프레임 선별기(예산 B)
                                                    │  JPEG 업로드 (POST /v1/frames)
                                                    ▼
                                        Teacher 의사 라벨 → tau 필터 → Detect 헤드만 학습
                                                    │  어댑터 npz(FP16) (GET /v1/adapters/…)
              ◀──────── 어댑터 교체, 신뢰도 악화 시 이전 어댑터로 롤백 ◀─┘
```

핵심 실험은 서버 한 대에서 엣지를 **모사**해 수행한다(계획서 4.1.6). 이때 `run_experiment.py`가 서버 학습 함수를 같은 프로세스에서 직접 호출한다. FastAPI 통신(`api.py`, `edge_client.py`)은 데모(8장)에서 쓴다.

| 담당 (계획서 역할분담) | 파일 | 주로 하는 장 |
|---|---|---|
| 김동은 | `shift_to_yolo.py`, `select_seqs.py`, `extract_seqs.py`, `torch_detector.py`(TENT), Student·Teacher 학습, 실행 A·T·C | 4, 5, 6.1, 7 |
| 김승룡 | `selectors.py`, `server_core.py`, 실행 B·D·예산 grid | 5.4, 6.6 |
| 강서윤 | `signals.py`, `drift_model.py`, `train_detector.py`, `env_sig_check.py` | 6.2, 6.3 |
| 이은민 | `onnx_detector.py`, `export_base.py`, `quantize_onnx.py`, `bench_latency.py`, `api.py`, `edge_client.py`, `serve.py`, `demo.py` | 5.3, 8 |

---

## 2. 설치

### 2.0 순서 (이 순서를 지킨다)

| 순서 | 할 일 | 누가 | 의존 |
|---|---|---|---|
| 1 | 2.1절 GitHub SSH 키 만들기·등록 | 모든 팀원 | GitHub 계정 |
| 2 | 2.3절 (a) Python·Git 설치 | 모든 팀원 | 없음 |
| 3 | `edge-adapt.zip` 풀기 → **3장 GitHub 저장소 만들기·첫 push·팀원 초대** | 저장소를 만드는 1명 | 1, 2 |
| 4 | 초대 수락 → 2.3절 (b) `git clone` | 나머지 팀원 | 3 |
| 5 | 2.3절 (c) 가상환경·패키지 설치 | 모든 팀원 | 코드 폴더(3 또는 4) |

**Colab 학습(0.5절)은 이 표와 무관하다.** 드라이브에 올린 zip으로 시작하므로 GitHub 저장소가 없어도 된다. 이 표는 팀원끼리 코드를 공유하고 노트북에서 데모를 돌릴 때를 위한 순서다.

### 2.1 공통: GitHub SSH 키 (서버와 모든 팀원 노트북에서 각각 1회)

1. 터미널을 연다.
   - Windows: 시작 버튼 클릭 → 검색창에 `PowerShell` 입력 → **Windows PowerShell** 클릭
   - macOS: Spotlight(⌘+Space) → `터미널` 입력 → Enter
   - 서버: 노트북 터미널에서 `ssh 계정@서버주소`
2. 키를 만든다. 질문이 나오면 Enter를 세 번 누른다(경로 기본값, 비밀번호 없음).
   ```
   ssh-keygen -t ed25519 -C "GitHub에 등록한 이메일"
   ```
3. 공개키를 출력해 복사한다.
   - macOS/Linux/서버: `cat ~/.ssh/id_ed25519.pub`
   - Windows PowerShell: `type $env:USERPROFILE\.ssh\id_ed25519.pub`
   - `ssh-ed25519 AAAA…` 로 시작하는 **한 줄 전체**를 드래그해 복사한다.
4. 브라우저에서 https://github.com 에 로그인 → 오른쪽 위 **프로필 사진** 클릭 → **Settings** → 왼쪽 메뉴 **SSH and GPG keys** → 초록색 **New SSH key** 버튼.
5. **Title**에 기기 이름(예: `lab-server`, `eunmin-laptop`), **Key type**은 `Authentication Key` 그대로, **Key** 칸에 3번에서 복사한 줄을 붙여넣고 **Add SSH key** 클릭.
6. 확인: `ssh -T git@github.com` 을 실행하고, 처음 묻는 `Are you sure you want to continue connecting` 에 `yes`를 입력. `Hi <아이디>! You've successfully authenticated` 가 나오면 성공.

(GitHub 화면의 글자는 업데이트로 조금 다를 수 있다. 위치는 "프로필 → Settings → SSH and GPG keys → New SSH key"로 찾는다.)

### 2.2 (참고) Linux GPU 서버를 직접 쓸 때만. 이 팀의 기본 경로는 0.5절(노트북 CPU + 무료 Colab)이므로 건너뛴다

```bash
nvidia-smi                      # NVIDIA GPU가 있으면 표가 나온다. 'command not found'면 NVIDIA GPU가 없는 것이므로 이 절을 건너뛰고 0.5절을 따른다
python3 --version               # 3.10 ~ 3.12 이어야 한다. 3.13 이상이면 설치하지 말고 3.11을 별도로 준비
sudo apt update && sudo apt install -y git tmux python3-venv python3-pip   # sudo 권한이 없으면 관리자에게 git, tmux, venv 설치를 요청
```

저장소 가져오기는 3장(GitHub 저장소 만들기)을 끝낸 뒤에 한다. 그 뒤:

```bash
git clone git@github.com:Eunmin1218/edge-adapt.git     # 3장에서 만든 저장소 주소
cd edge-adapt
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124      # NVIDIA GPU가 있을 때. GPU가 없으면: pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements-server.txt
python scripts/00_check_env.py --server     # 마지막 줄이 OK, "CUDA 사용 가능: True" 여야 한다
python -m pytest tests -q                   # 10 passed
```

- 새 터미널을 열 때마다 `cd ~/edge-adapt && source .venv/bin/activate` 를 먼저 실행한다.
- 오래 걸리는 작업은 **tmux** 안에서 돌린다(SSH가 끊겨도 계속 실행된다).
  - 시작: `tmux new -s exp` → 작업 실행 → 빠져나오기: **Ctrl+b 를 누른 뒤 손을 떼고 d** → 다시 들어가기: `tmux attach -t exp`

### 2.3 노트북 준비 (Windows 기준. 순서가 중요하다: 2.0절의 표를 먼저 본다)

**(a) Python과 Git 설치 — 모든 팀원, 저장소와 무관하므로 가장 먼저**
1. 브라우저에서 https://www.python.org/downloads/ → **Download Python 3.11.x** 클릭 → 받은 설치 파일 실행 → 첫 화면 **아래쪽 `Add python.exe to PATH` 체크** → **Install Now**.
2. https://git-scm.com/download/win → 설치 파일 실행 → 모든 화면에서 **Next**를 눌러 기본값으로 설치 → **Finish**.
3. 확인: PowerShell(2.1절 방법)을 **새로** 열고 `python --version`(3.11.x), `git --version` 이 각각 버전을 출력해야 한다.

**(b) 코드 폴더 만들기 — 두 경우 중 자신에게 맞는 것 하나만**
- **저장소를 만드는 사람(1명)**: 받은 `edge-adapt.zip`을 **우클릭 → 모두 압축 풀기 → 압축을 풀 폴더 선택(예: `C:\Users\본인\work`) → 압축 풀기**. 생긴 `edge-adapt` 폴더가 작업 폴더다. 이 사람은 `git clone`을 하지 않는다. 3장(GitHub 저장소 만들기)을 이 폴더에서 진행하고, 끝낸 뒤 (c)로 간다.
- **나머지 팀원**: 3장이 끝나고, GitHub에서 온 초대 메일의 **Accept invitation**을 누른 **뒤에만** 가능하다. PowerShell에서 `cd C:\Users\본인\work` 로 원하는 위치로 이동해 `git clone git@github.com:Eunmin1218/edge-adapt.git` 을 실행한다 (저장소 주소는 3장에서 만든 것). `Permission denied (publickey)`가 나오면 2.1절 6번을 다시 확인한다.

**(c) 가상환경 만들기와 패키지 설치 — 작업 폴더 안에서**
가상환경은 이 프로젝트의 Python 라이브러리를 컴퓨터 전체가 아니라 폴더 안의 `.venv`에만 설치하는 방식이다(버전 충돌 방지). 만들기는 처음 1회, 켜기는 터미널을 새로 열 때마다 한다.
1. 탐색기에서 작업 폴더(`edge-adapt`)를 열고 위쪽 **주소창을 클릭** → `powershell` 입력 → **Enter**. (그 폴더에서 PowerShell이 열린다.)
2. 만들기(1회): `python -m venv .venv`
3. 켜기: `.\.venv\Scripts\Activate.ps1` — 줄 맨 앞에 `(.venv)`가 보이면 성공. `이 시스템에서 스크립트를 실행할 수 없으므로` 오류가 나면 `Set-ExecutionPolicy -Scope Process Bypass` 입력 후 **Enter**, 다시 3번.
4. 설치와 점검:
   ```
   pip install -U pip
   pip install -r requirements-edge.txt
   python scripts\00_check_env.py        # 마지막 줄 OK
   python -m pytest tests -q             # 엣지 쪽 패키지만 있으면 fastapi 등이 없어 일부가 오류일 수 있다
   ```
   (서버 역할까지 하는 노트북은 0.5절의 "노트북 준비"에서 torch 등을 추가로 설치한다.)

**macOS**: `brew install python@3.11 git` (Homebrew가 없으면 https://brew.sh 의 안내 한 줄을 터미널에 붙여넣는다) → 위 (b), (c)와 같고, 가상환경 켜기만 `source .venv/bin/activate`, 경로는 `/`.

---

## 3. GitHub 저장소 만들기 (대표 1인이 1회, 이후 팀원은 초대 수락)

저장소 소유자는 계획서의 GitHub 계정 중 한 명(예: 이은민 `Eunmin1218`)으로 한다.

1. https://github.com 로그인 → 오른쪽 위 **`+`** 아이콘 클릭 → **New repository**.
2. 입력:
   - **Owner**: 본인 계정
   - **Repository name**: `edge-adapt`
   - **Description**: `Self-adaptive edge-cloud collaborative video analytics via continual learning`
   - 공개 범위: 개발 중에는 **Private** 선택 (16주차에 Public으로 변경)
   - **Add a README file**, **Add .gitignore**, **Choose a license**: 모두 **선택하지 않는다** (이미 로컬에 있다. 선택하면 첫 push가 충돌한다)
3. 초록색 **Create repository** 클릭.
4. 나오는 화면에서 **SSH** 탭을 클릭하고 `git@github.com:<아이디>/edge-adapt.git` 주소를 복사한다.
5. 로컬에서 — **2.3절 (b)의 "저장소를 만드는 사람"이 `edge-adapt.zip`을 푼 폴더**에서 PowerShell을 열고(폴더 주소창에 `powershell` 입력 → Enter) 실행한다:
   ```bash
   git config --global user.name "이은민"
   git config --global user.email "GitHub에 등록한 이메일"
   git init -b main
   git add .
   git status                       # data/, models/ 가 목록에 없는지 확인 (.gitignore 때문에 없어야 한다)
   git commit -m "Initial commit: edge-adapt pipeline"
   git remote add origin <4번에서 복사한 SSH 주소>
   git push -u origin main
   ```
6. 브라우저에서 저장소 페이지를 새로고침해 파일이 올라갔는지 확인한다.
7. **팀원 초대**: 저장소 페이지 위쪽 **Settings** 탭 → 왼쪽 **Collaborators** → **Add people** 버튼 → 아이디 입력(`kseoyun0214`, `ehddms0320`, `ryong02`) → 목록에서 선택 → **Add … to this repository**. 초대받은 사람은 GitHub 알림/이메일의 **Accept invitation** 을 클릭한다.
8. **라이선스 파일**: 저장소 페이지 **Add file** ▾ → **Create new file** → 파일 이름 칸에 `LICENSE` 입력 → 오른쪽에 나타나는 **Choose a license template** 클릭 → **GNU Affero General Public License v3.0** 선택 → **Review and submit** → 아래 **Commit changes...** → **Commit changes**. 이후 로컬에서 `git pull`.
   (Ultralytics YOLO가 AGPL-3.0이므로 이를 호환하는 라이선스를 쓴다. SHIFT 데이터는 CC BY-NC-SA 4.0으로 `data/`는 저장소에 올리지 않는다.)

**협업 규칙** (실수로 서로의 작업을 덮어쓰지 않기 위한 최소 규칙)
```bash
git switch -c feat/<이름>-<주제>        # 작업 시작 전 브랜치 생성. 예: feat/dongeun-teacher
# ...파일 수정...
git add <수정한 파일>
git commit -m "무엇을 왜 바꿨는지"
git push -u origin feat/<이름>-<주제>
```
GitHub에서 노란 **Compare & pull request** 버튼 → **Create pull request** → 다른 팀원 1명이 **Files changed → Review changes → Approve** 후 **Merge pull request → Confirm merge**. `main`에는 직접 push하지 않는다. 실험 결과 JSON(`results/*.json`, `results/summary.md`)은 커밋하고, 데이터·모델·`*.npz`는 커밋하지 않는다.

---

## 4. 데이터: SHIFT

> **이 장과 5·6장의 명령은 0.5절의 Colab 노트북 셀로 이미 구성돼 있다.** 노트북을 쓰면 이 장은 "각 명령이 무엇을 하는지" 이해하는 용도로 읽고, 직접 실행하는 경우(예: 연구실 GPU 서버)에만 그대로 입력한다.

SHIFT는 CC BY-NC-SA 4.0(비영리 연구용)이다. 받은 데이터는 재배포하지 않는다(계획서 4.3.5).

### 4.1 받을 파일 (이산 10Hz 비디오, front 뷰, 2D 박스만)
서버에서 실행한다. 폴더는 `data/raw/discrete/videos/<split>/front/` 에 생기며 `img.tar`, `det_2d.json`, `seq.csv` 가 들어간다.

```bash
pip show huggingface_hub >/dev/null && echo ok          # requirements-server.txt 에 포함됨
python scripts/download_shift_hf.py --split val --dry-run     # 파일 목록과 GB 크기만 출력, 다운로드하지 않음
python scripts/download_shift_hf.py --split train --dry-run
df -h .                                                       # 남은 용량 확인. img.tar 크기의 약 1.3배 이상 필요
tmux new -s dl
python scripts/download_shift_hf.py --split val
python scripts/download_shift_hf.py --split train
```
- `미러에 ... 가 없습니다` 가 나오면 미러 구조가 바뀐 것이다. 브라우저로 https://huggingface.co/datasets/KevinConnorLee/SHIFT/tree/main 을 열어 실제 경로를 확인하고, 그래도 없으면 [SHIFT GitHub 이슈 #80](https://github.com/SysCV/shift-dev/issues/80)에서 최신 대체 경로를 찾는다. 공식 서버가 복구됐다면 `python download.py --view "[front]" --group "[img, det_2d]" --split "all" --framerate "[videos]" --shift "discrete" ./data/raw` (SHIFT 공식 [shift-dev](https://github.com/SysCV/shift-dev) 저장소의 `download.py`) 를 쓴다.
- `401`/`403` 오류: https://huggingface.co 로그인 → 오른쪽 위 프로필 → **Settings** → **Access Tokens** → **Create new token**(Read) → 복사 → 서버에서 `huggingface-cli login` 후 붙여넣기.

### 4.2 이산 시퀀스의 규격 (논문 기준)
이산 시퀀스는 약 4,250개, 시퀀스당 500프레임(10Hz, 약 50초)이다([논문](https://arxiv.org/pdf/2206.08367)). 그래서 `select_seqs.py`의 스트림 구간 길이 기본값을 500프레임으로 했다. 실제 프레임 수가 다르면 `load_stream`이 윈도우(50프레임)의 배수로 잘라 쓰고, 50프레임 미만이면 오류로 멈춘다.

### 4.3 구조 확인 (처음 1회, 반드시)
```bash
python scripts/select_seqs.py --seq-csv data/raw/discrete/videos/val/front/seq.csv --inspect
```
- 출력의 **열 이름 목록**에서 시퀀스 이름 열, 날씨 열, 시간대 열을 찾는다.
- 현재 코드가 찾는 이름: 시퀀스 `video/videoName/seq/name/id`, 날씨 `weather_coarse/weather`, 시간대 `timeofday_coarse/timeofday/time_of_day`. 다르면 `scripts/select_seqs.py` 맨 위의 `WEATHER_KEYS`, `TIME_KEYS`, `ID_KEYS` 목록에 실제 열 이름을 추가한다.
- 값에 `clear`, `rain`, `fog`, `day`, `night` 문자열이 들어 있는지 본다(부분 문자열 일치). 다르면 `WEATHER_MAP`/`TIME_MAP`을 고친다.

### 4.4 시퀀스 선택
```bash
python scripts/select_seqs.py \
  --train-csv data/raw/discrete/videos/train/front/seq.csv \
  --val-csv   data/raw/discrete/videos/val/front/seq.csv \
  --seed 0 --n-student 300 --n-teacher 750 --seg-frames 500
```
만드는 것 (`data/lists/*.txt`, `configs/streams.json`):

| 파일 | 내용 | 쓰임 |
|---|---|---|
| `student_train.txt` | 맑은 낮 시퀀스 300개 | Student 학습 |
| `teacher_train.txt` | 맑은 낮·비·밤 각 250개 (**안개 제외**) | Teacher 학습 |
| `student_val.txt`, `teacher_val.txt` | val에서 맑은 낮 40개 / 환경별 15개 | 학습 중 검증 |
| `stream_train.txt` | train에서 학습에 쓰지 않은 시퀀스로 구성한 스트림(맑음→비→맑음→밤→맑음) | 감지기 학습 |
| `stream_eval.txt` | val에서 구성한 스트림(위 + 안개→맑음) | A~D 비교, 감지기 평가 |

- 출력된 "환경별 시퀀스 수"를 확인한다. 필요한 수보다 적으면 `스트림용 '…' 시퀀스가 …개뿐` 으로 멈춘다. 이때 `--n-student`, `--n-teacher`를 줄인다(줄이는 것은 팀이 정한다).
- 스트림 시퀀스는 모델 학습·검증에 쓴 시퀀스와 겹치지 않게 뽑는다(평가 누수 방지).
- 시퀀스 수(300/750)는 계획서 4.3.2의 "사전 측정 후 확정"에 해당한다.

### 4.5 필요한 프레임만 꺼내기 (디스크 절약)
10Hz 전체를 풀면 시퀀스당 500장이다. 학습용은 1Hz(10프레임마다 1장)만 필요하므로 `--every 10`으로 1/10만 꺼낸다. 스트림은 10Hz 전체가 필요하므로 `--every 1`.
```bash
D=data/raw/discrete/videos
# train
python scripts/extract_seqs.py --tar $D/train/front/img.tar --seqs data/lists/student_train.txt data/lists/teacher_train.txt --out data/raw/img_train --every 10
python scripts/extract_seqs.py --tar $D/train/front/img.tar --seqs data/lists/stream_train.txt --out data/raw/img_train --every 1
# val
python scripts/extract_seqs.py --tar $D/val/front/img.tar --seqs data/lists/student_val.txt data/lists/teacher_val.txt --out data/raw/img_val --every 10
python scripts/extract_seqs.py --tar $D/val/front/img.tar --seqs data/lists/stream_eval.txt --out data/raw/img_val --every 1
```
각 명령은 tar를 처음부터 끝까지 읽으므로 시간이 걸린다(tmux 안에서). 끝에 `시퀀스 N/N개` 로 전부 찾았는지 확인한다. `못 찾은 시퀀스` 경고가 나오면 4.3절의 시퀀스 이름 열이 틀렸을 가능성이 높다.

### 4.6 구조 확인과 YOLO 형식 변환
```bash
python scripts/inspect_shift.py $D/val/front/det_2d.json data/raw/img_val
```
확인할 것: ① `카테고리 분포`에 `pedestrian, car, truck, bus, motorcycle, bicycle` 가 있는지(다르면 `src/common.py`의 `CLASSES` 수정), ② `name`과 `폴더 예`가 같은 파일 이름 형식인지.

변환 (좌표를 1280×800 → 640×400으로 절반 축소, 최소 박스 크기 미만 제외, 클래스 6개):
```bash
Y="python scripts/shift_to_yolo.py"; TJ=$D/train/front/det_2d.json; VJ=$D/val/front/det_2d.json
$Y --det2d $TJ --images data/raw/img_train --seqs data/lists/student_train.txt --out data/yolo/student_train --step 10 --drop-empty
$Y --det2d $TJ --images data/raw/img_train --seqs data/lists/teacher_train.txt --out data/yolo/teacher_train --step 10 --drop-empty
$Y --det2d $VJ --images data/raw/img_val   --seqs data/lists/student_val.txt   --out data/yolo/student_val   --step 10 --drop-empty
$Y --det2d $VJ --images data/raw/img_val   --seqs data/lists/teacher_val.txt   --out data/yolo/teacher_val   --step 10 --drop-empty
$Y --det2d $TJ --images data/raw/img_train --seqs data/lists/stream_train.txt  --out data/yolo/stream_train --step 1
$Y --det2d $VJ --images data/raw/img_val   --seqs data/lists/stream_eval.txt   --out data/yolo/stream_eval  --step 1
```
- `--step 10`: 10Hz 프레임 중 frameIndex가 10의 배수인 것만 사용(= 1Hz). 계획서 4.3.3의 "이미지 단위 학습 1Hz, 객체 없는 프레임 제외"와 같다.
- 스트림은 `--step 1`, `--drop-empty` 없음: 10Hz 전체 프레임이 필요하고, 객체 없는 프레임도 평가·신호 계산에 포함해야 한다.
- `이미지 0장입니다` → 4.6절 `inspect_shift.py` 출력과 `--images` 경로를 대조한다. `이미지 못 찾음`이 많으면 `--every`와 `--step`이 어긋난 것이다(둘 다 10 또는 둘 다 1).
- 최소 박스 크기는 `src/common.py`의 `MIN_BOX_PX = 8`(640×400 기준 픽셀)이다. 11주차에 환경별 mAP를 본 뒤 확정한다(계획서 4.3.3).

### 4.7 변환 결과를 눈으로 확인 (필수)
```bash
python scripts/vis_check.py data/yolo/student_train --n 8
python scripts/vis_check.py data/yolo/stream_eval --n 8
```
`results/vis_student_train.jpg` 를 노트북으로 가져와(`scp 계정@서버:~/edge-adapt/results/vis_*.jpg .`) 연다. **초록 박스가 차·사람 위에 정확히 얹혀 있어야 한다.** 일정하게 어긋나면 좌표 변환 문제이므로 학습으로 넘어가지 않는다.

---

## 5. 모델 학습과 어댑터 파이프라인 점검 (서버)

### 5.1 학습 설정 파일
```bash
python scripts/make_yaml.py student
python scripts/make_yaml.py teacher
```
`configs/student.yaml`, `configs/teacher.yaml`이 생긴다(절대경로 사용).

### 5.2 Student 학습 (맑은 낮만)
```bash
tmux new -s student
yolo detect train data=configs/student.yaml model=yolo11n.pt imgsz=640 epochs=50 batch=32 device=0 seed=0 project=runs name=student
```
- 첫 실행 시 `yolo11n.pt`와 글꼴을 인터넷에서 받는다. 서버가 외부에 연결되지 않으면 실패하므로 노트북에서 받아 `scp`로 올린다.
- 완료 후 `runs/student/weights/best.pt` 가 생긴다. 학습 로그 마지막 `mAP50` 값을 기록한다(보고서 Baseline 근거).
- 시간: 첫 epoch가 끝나는 시간을 재고 `× epochs` 로 총 시간을 계산한다(미측정 — 이 가이드는 시간을 단정하지 않는다).

### 5.3 Teacher 학습 (맑음·비·밤, 안개 제외)
```bash
tmux new -s teacher
yolo detect train data=configs/teacher.yaml model=yolo11x.pt imgsz=640 epochs=30 batch=16 device=0 seed=0 project=runs name=teacher
```
- `CUDA out of memory` → `batch=8`.
- 시간이 부족하면 선택지: `model=yolo11l.pt`, `epochs`를 줄임, `fraction=0.5`(학습 데이터 절반). **어느 것을 쓸지는 팀과 멘토가 정한다.** 어떤 선택이든 보고서에 모델·epoch를 적는다.
- Student와 Teacher는 서로 다른 GPU 프로세스이므로 GPU 메모리가 충분하면 동시에 돌려도 된다.

### 5.4 기본 모델 내보내기와 **어댑터 파이프라인 점검**
```bash
mkdir -p models
cp runs/teacher/weights/best.pt models/teacher.pt
python scripts/export_base.py runs/student/weights/best.pt      # models/student.pt, models/student.onnx 생성
python scripts/check_adapter_pipeline.py
```
`export_base.py`가 출력하는 initializer 개수와 `model.<번호>.` 이름 예시를 기록한다(`<번호>`는 YOLO11n에서 Detect 레이어 번호).

`check_adapter_pipeline.py`가 확인하는 4가지와 통과 기준:

| 번호 | 확인 | 통과 기준 | 실패 시 의미 |
|---|---|---|---|
| (1) | 같은 `.pt`를 두 번 내보낸 ONNX의 initializer 이름·값 | 동일 | 어댑터를 "값이 바뀐 initializer"로 정의할 수 없다 → `pip list`로 torch 2.5.1, ultralytics 8.3.40, onnx<1.18 확인 |
| (2) | 헤드만 흔든 모델의 바뀐 initializer | 1개 이상이고 전체보다 적음 | 0개: ONNX가 이름을 바꿔 헤드를 못 찾음. 전체: 백본까지 바뀜 |
| (3) | base ONNX에 어댑터를 끼운 출력 vs 흔든 모델 출력 | 최대 오차 < 1e-4 | 엣지 교체 경로가 틀림 |
| (4) | 어댑터 크기 | FP16 5MB 이하 | 초과 시 0.4절 #2의 정의 재논의 |

**(1)~(3)이 통과하기 전에는 6장으로 넘어가지 않는다.** 실패하면 출력된 메시지와 함께 팀에 공유하고 10장의 표를 본다.

### 5.5 파일 옮기기 (서버 역할 컴퓨터와 엣지 노트북이 다를 때만. 한 노트북이면 건너뛴다)
서버에서 노트북으로:
```bash
scp 계정@서버주소:~/edge-adapt/models/student.onnx models/
```
(`models/`가 없으면 `mkdir models`.) Windows PowerShell도 `scp`가 동일하게 동작한다.

### 5.6 어댑터 설정 튜닝 (본 실험 전에 반드시, 약 몇 분~수십 분: 미측정)

**무엇을 하나**: 서버 적응의 **프레임 선별 방식**, 학습 epoch, 학습률, **헤드 앞에서 추가로 학습할 레이어 수(`train_layers`)** 를 감으로 정하지 않고 **개발용 스트림(`detector_train`)에서 측정해 고른다.** 각 환경 변화 구간(비, 밤)에서 앞쪽 100프레임 중 16장을 선별해 서버 학습 → 구간 뒤쪽 **보류 프레임**(적응에 쓰지 않은 프레임)의 mAP50이 적응 전보다 얼마나 오르는지(= 이득)를 잰다. **평가 스트림(`stream_eval`)은 사용하지 않으므로** 튜닝이 최종 결과를 오염시키지 않는다. 1단계에서 선별 방식(`--selectors`, 기본 `diversity,ent_hybrid,hybrid,random`)을 기본 학습 설정으로 비교하고(무작위성이 있는 방식은 `--sel-seeds` 시드로 반복), 2단계에서 선택된 선별 방식으로 학습 설정 격자를 돈다. 합성 시험에서 선별 방식에 따라 보류 mAP50이 0.51~0.98로 크게 달랐으므로(0.5.1절 표) 이 단계를 생략하지 않는다.

```
python scripts/tune_adapter.py            # 전체 격자: epoch {10,30,60} × 학습률 {3e-3,5e-3,1e-2} × train_layers {0,3,6}
python scripts/tune_adapter.py --quick    # 작은 격자 (Colab 노트북의 free 규모)
python scripts/tune_adapter.py --epochs 30 --layers 0 --lrs 1e-2 --selectors diversity,random,edgesync   # 직접 지정
```
학습 방식은 기본 `--trainer distill`(Teacher 조밀 출력 증류, `src/distill.py`)이고, `--trainer ultra`(Teacher 하드 라벨 + ultralytics 학습)와 비교할 수 있다. `batch 0` 은 자동(증류에서 프레임 수의 절반 → epoch당 2스텝, 총 60스텝)이다. 스텝 수가 성능을 좌우했다(30스텝: 0.60, 60스텝: 0.75, 120스텝: 0.67~0.73. 합성 시험).
출력: 구간별 적응 전 mAP50, 설정마다 `평균 이득`, 적응 1회 시간, 어댑터 크기(MB). 마지막 줄 `선택: ...` 이 `configs/adapter.json`에 저장되고, **이후 `run_experiment.py`와 `serve.py`가 인자 없이도 이 설정을 읽는다.** B, EdgeSync-lite, 제안 모두 같은 서버 설정을 쓰므로 비교가 공정하다. 어댑터가 5MB를 넘는 설정은 선택에서 제외된다(목표 3).

- `train_layers = k` 는 Detect 헤드 앞의 k개 레이어도 함께 학습한다는 뜻이다. 커질수록 적응 능력은 늘지만 어댑터 크기와 적응 시간이 늘고, 소수 프레임에 과적합할 위험이 있다. 어느 값이 좋은지는 **데이터가 정한다.**
- `경고: 어느 설정도 보류 프레임 mAP 를 올리지 못했습니다`: 적응 자체가 효과가 없는 상태다. 본 실험(6장)으로 가지 말고 다음 순서로 점검한다. ① Teacher 단독(T)의 mAP가 낮으면 Teacher 재학습(의사 라벨이 나쁘면 적응이 안 된다) ② `--taus 0.3,0.5,0.7` ③ `--layers 6,9 --epochs 20,40` ④ `--budget 32`. 그래도 안 되면 이 사실을 팀과 멘토에게 알리고 연구 질문을 조정한다(숨기지 않는다).

### 5.6.1 용량을 유지하며 성능을 올리는 어댑터 재구성 (affine 어댑터 + 증류 + 외삽 + 델타 코덱)

**문제**: 헤드(Detect) 가중치를 fp16 으로 보내는 어댑터는 약 360KB 이고, 소수 프레임(8장)으로 이만큼의 자유도를 학습하면 과적합한다. 다른 문제(분포 이동 적응·모델 병합·모델 압축)의 기법을 가져와 **보내는 바이트를 줄이면서 정확도를 올리는** 구성을 만들었다. 바이트가 줄면 같은 다운링크 예산에서 어댑터를 더 자주 보낼 수 있다는 점도 이득이다.

| 가져온 기법 (원 분야) | 우리 연구에서의 구현 | 기대 이유 (메커니즘) | 코드 |
|---|---|---|---|
| 채널별 affine 만 학습 (테스트 시 적응: TENT·BN 적응 계열) | 융합된 모델의 모든 합성곱 출력 채널에 `(1+ds)*y+t` 를 학습, 학습 후 가중치·편향에 접는다 (접기 오차 3e-4 로 검증). 174개 합성곱, 15,796 파라미터 | 조명·날씨 변화는 채널 통계의 평행이동·배율로 상당 부분 설명된다. 자유도가 작아 8장으로도 과적합이 적다 | `src/adapt_modules.py`, `src/distill.py` |
| 조밀 증류 (Unbiased/Mean Teacher, Probabilistic Teacher 계열) | Teacher 의 클래스 확률 전체와 박스 분포를 손실로 사용 (하드 라벨 대신) | tau 아래 약한 검출과 박스 위치 분포 정보가 버려지지 않는다 | `src/distill.py` |
| WiSE-FT 식 보간·외삽 | 어댑터 차이를 α=1.25 배로 외삽 (`scale_adapter`) | 소수 프레임 학습은 목표보다 덜 가는 경향이 있다. α 는 개발 스트림에서 골라야 한다 | `src/delta_codec.py` |
| 델타 코덱 (모델 압축·연합학습 통신 압축) | affine 이 접힌 합성곱은 `(s-1)`, `t` 만 fp16 으로 보내고 나머지는 채널별 int8 차이. 실제 바이트는 압축 파일 크기로 계산 | 접히기 전의 저차원 구조를 그대로 전송하면 가중치 전체(4MB)를 보낼 필요가 없다 | `src/delta_codec.py`, `src/sim.py` `ship()` |
| 입력 톤 커브 (이미지 신호처리 적응) | 구현·시험함 (`tone=True`). affine 위에서는 이득 없음 → 기본 꺼짐 | | `src/adapt_modules.py` |

**측정 (합성 장난감 세계, `runs/toy2`; 실제 SHIFT 수치가 아니다)**
- 소거 실험(보류 프레임 mAP50): affine 만(72KB, α=1.25, 60 epoch, lr_aff 0.03) **0.955**, 헤드 증류(362KB) 0.857, 헤드+톤 0.908. 톤은 affine 위에서 이득 없음. affine+헤드는 학습률이 높으면 불안정.
- 전체 스트림(`python scripts/toy_e2e.py --world toy2`): A 0.472, T 0.990, **D(affine) 0.713 (다운링크 72KB)**, D(헤드, 이전 구성) 0.668 (361KB), B 0.730, EdgeSync-lite 0.747 (변화구간 mAP50).
- **한계(해결하지 못한 것)**: 이 스트림에서 D 는 B·EdgeSync-lite 보다 낮다. 원인은 밤+비(night_rain) 구간에서 레지스트리가 환경 서명(색·기울기 통계)이 밤과 구분되지 않아 밤 어댑터를 재사용하고 새로 적응하지 않는 것이다 (서명 거리 0.004). 서명에 기울기를 추가해도 구분되지 않았다. 재사용한 어댑터의 신뢰도를 기대값과 비교해 새로 적응시키는 규칙(`verify_margin`, 기본 꺼짐)을 구현했으나, 맑음→안개 구간에서 해로운 어댑터를 만들어 전체가 나빠져 기본으로 쓰지 않는다. 실제 SHIFT 에서의 이 문제의 크기는 측정 전이다.
- 주의: 헤드 어댑터는 변화가 작을 때(안개) 오히려 해롭다. α=1.25 와 lr_aff 는 같은 장난감 분포에서 골랐으므로, **반드시 `tune_adapter.py` 로 SHIFT 개발 스트림에서 다시 고른다** (격자: `--modes affine,head --lras ... --alphas 1.0,1.25`; 이 새 격자는 끝까지 실행해 보지 않았다).
- 시도했으나 채택하지 않은 것: 엔트로피 가중 손실(`uw`), 첫 판정 이후 프레임으로 후보 제한, 전환 후 한 윈도우 유예(효과 없음). 구현하지 않은 것: LoRA, 하이퍼네트워크, 복사-붙여넣기 증강, Teacher TTA, 업링크 JPEG 품질 절충.

### 5.6.2 엣지 통계 정렬 추적 (E / E+) — 서버·전송 없이 즉시 적응하고, 이전 구성요소는 역할을 바꿔 유지

**조사에서 가져온 것**: 테스트 시 적응에서 정규화 통계를 맞추는 방법(AdaBN 계열), 소스 통계를 목표로 쓰는 방법(Sensitivity-guided pruning 이 소스 통계를 쓰는 것과 같은 취지), 저차원 부분공간 적응(intrinsic dimension, VeRA; 이번에는 구현하지 않음). ICCV'25 파라미터 생성(확산 모델)은 보고된 이득이 0.5~1.5점이고 매 스텝 확산 샘플링 비용이 정량화되어 있지 않아 제외했다.

**메커니즘 (`src/stat_align.py`)**: 융합된 합성곱 출력(활성함수 앞)의 채널별 평균·표준편차는 입력의 밝기·대비·잡음에 따라 변한다. 소스(맑은 날) 프레임에서 잰 값 (mu_s, sd_s) 을 배포 때 한 번 저장하고, 새 환경 프레임 12장에서 같은 값을 재서 `s = sd_s/sd_b`, `t = mu_s - s*mu_b` 로 맞춘다. 앞 층을 보정한 뒤의 활성을 다음 층에서 재도록 한 번의 순전파에서 층 순서대로 계산하고, 가중치·편향에 접는다 (마지막 클래스·박스 출력 합성곱은 제외). **라벨·Teacher·서버·학습이 필요 없고 전송 바이트가 0이다.** 같은 계산에서 나오는 채널 통계 서명(층별 로그 배율·이동 평균, 348차원)을 환경 식별에도 쓴다.

**추적 구조 (`src/sim.py` `_edge_track`, `_fast_check`)**
1. 평균 신뢰도 모니터(추가 연산 없음): 최근 10프레임 신뢰도가 기준(첫 윈도우=소스)보다 0.1 넘게 낮을 때만 5장으로 서명을 계산한다. 서명이 바뀌지 않았으면 한 윈도우 동안 쉰다. 윈도우 끝에서도 서명 변화와 **RQ1 드리프트 감지기** 판정을 함께 본다.
2. 서명이 바뀌면 **레지스트리**(서명 거리 0.25 이내)에서 같은 환경의 어댑터를 찾아 적용하고, 없으면 통계 정렬 어댑터를 즉시 만들어 적용·저장한다.
3. **서버 증류 어댑터**(affine, 델타 코덱, 다양성 선별)는 적용 후 온전한 윈도우의 신뢰도가 여전히 기준보다 0.08 넘게 낮을 때만 요청한다. 후보는 **EdgeSync-lite 의 프로브**(최근 6장의 Teacher 일치도)로 현재 어댑터와 비교해 0.02 넘게 나을 때만 받아들이고 (업링크 바이트 계산에 포함), 기존 **롤백**도 유지한다.

**측정 (합성 장난감 세계, toy2, 변화구간 mAP50 / 업링크 KB / 다운링크 KB)**
| 방법 | mAP50 | 업 | 다운 |
|---|---|---|---|
| A 적응 없음 / T Teacher | 0.472 / 0.990 | 0 | 0 |
| B 주기적 | 0.730 | 1849 | 569 |
| ES (EdgeSync-lite) | 0.747 | 2361 | 142 |
| D 이전 구성(affine 서버 어댑터) | 0.713 | 341 | 72 |
| D★ (추적, 윈도우 끝 점검만, 서버 없음) | 0.750 | 0 | 0 |
| **D★F (빠른 점검 10프레임, 서버 없음)** | **0.883** | 0 | 0 |
| D★F + 서버 정련 gate | 0.871 | 342 | 70 |
| D★F + 프로브 검증 | 0.871 | 599 | 70 |
- 보류 프레임 점검(`scripts/probe_stat.py`): 통계 정렬 어댑터 하나만으로 밤 0.40→0.947, 밤+비 0.285→0.949, 안개 0.917→0.949, 비 0.974→0.970 (서버 증류 affine 은 같은 조건에서 0.955). 안개처럼 변화가 작은 환경에서 해롭지 않았다.
- 개선의 대부분은 **반응 지연**에서 나온다: 윈도우(50프레임) 끝에서만 점검하면 환경이 바뀐 직후 한 윈도우가 통째로 이전 어댑터로 처리된다. 10프레임 빠른 점검으로 이 구간이 0.27~0.40 → 0.74~0.80 이 되었다.
- 이 장난감에서 서버 증류 어댑터는 추가 이득이 없었고(차이는 잡음 수준), 프로브는 6장에서 포화되어 후보를 거르지 못했다. 즉 **서버 단계의 가치는 이 시험으로 확인되지 않았다.**

**꼭 알아둘 한계**
- 장난감 환경(밤·안개·비)은 거의 전역 밝기·대비·잡음 변화라서 통계 정렬에 유리하다. 실제 SHIFT(전조등, 국소 반사, 비 줄무늬)에서는 통계 정렬이 부족할 가능성이 크고, 그때 서버 정련 단계가 필요한지는 측정 전이다. `E` 와 `E+` 를 SHIFT 에서 비교해야 한다.
- 레지스트리 없이 통계 정렬만 쓴 변형(0.850)은 밤 어댑터를 끝까지 유지한 우연의 결과(맑음 0.87)이므로 근거로 쓰지 않는다.
- 엣지 연산이 늘어난다: 서명 계산에 프레임 일부를 추가로 순전파한다 (이 시험의 비율은 결과 json `edge_extra`). 이 구현은 PyTorch 로 통계를 구하므로 노트북 엣지에서는 되지만, ONNX Runtime 만 있는 기기에는 층별 통계 출력이 필요해 구현하지 않았다. 시뮬레이터의 지연에는 어댑터 내보내기 시간이 반영되지 않는다.
- 신호 한계: 신뢰도 기준 0.1, 서명 임계값 0.25, 정렬 강도(`lam`=1)는 같은 장난감에서 정했다. SHIFT 개발 스트림에서 다시 정한다.

실행: `python scripts/run_experiment.py --split eval --method E --tag E_s0` (엣지만) 또는 `--method E+` (서버 gate 포함). 소스 통계는 `detector_train` 스트림의 맑은 프레임 60장에서 구한다. 이 두 방법의 실제 SHIFT 실행은 아직 해 보지 않았다.

### 5.6.3 한계·위험 시뮬레이션 (toy3: 통계 정렬의 가정을 깨는 환경) — ES·B·이전 D 와 같은 스트림

**목적**: 5.6.2 의 결과는 장난감 환경이 전역 밝기·대비 변화라서 유리했다. 통계 정렬이 틀리는 경우를 일부러 만들어 확인했다 (`python scripts/toy_hard.py --world toy3` 로 Teacher 준비, `python scripts/toy_e2e.py --world toy3 --sens --verify`).
환경 6종 (Student 는 맑음만 학습): night(0.37), **glare**(밤 + 국소 전조등 반사, 0.42: 국소 변화라 전역 통계 정렬로는 못 고침), **heavy_rain**(빗줄기 과밀, 적응 불필요 0.96), **clear_sparse / clear_dense**(영상 질감은 맑음과 같고 물체 수만 다름, 적응 불필요 0.96 / 0.96: 내용 통계만 바뀌는 경우 → 통계 정렬이 없는 문제를 만들어 해칠 수 있음). Teacher 는 전 환경 mAP50 1.0 (너무 쉬운 설정이라 Teacher 가 상한).
스트림: clear100, night150, clear50, glare150, clear50, heavy_rain100, sparse100, night100, dense100, clear100 (20 윈도우).

| 방법 | 변화구간 mAP50 | 전체 mAP50 | 업 KB | 다운 KB |
|---|---|---|---|---|
| A 적응 없음 / T Teacher | 0.642 / 1.000 | 0.745 / 1.000 | 0 | 0 |
| B 주기적 | 0.782 | 0.840 | 2259 | 719 |
| **ES (EdgeSync-lite)** | **0.890** | **0.912** | 2630 | 72 |
| D 이전 구성(서버 affine) | 0.804 | 0.857 | 691 | 144 |
| E: 통계 정렬 추적만 (윈도우 끝 점검) | 0.741 | 0.737 | 0 | 0 |
| E + 10프레임 빠른 점검 (D★F, 서버 없음) | 0.852 | 0.731 | 0 | 0 |
| D★F + 서버 정련 gate | 0.863 | 0.842 | 443 | 144 |
| **V: 빠른 점검 + 프로브 검증 (서버 정련 없음)** | **0.899** | **0.911** | 668 | 0 |
| V + 서버 정련 gate | 0.895 | 0.909 | 1049 | 72 |

환경별 윈도우 평균 mAP50 (clear / night / glare / heavy_rain / sparse / dense):
A [0.98 0.40 0.41 0.97 0.97 0.95], ES [0.98 0.81 0.85 0.95 0.98 0.99], B [0.99 0.46 0.90 0.99 1.00 0.99], D★F 서버 없음 [0.96 0.86 0.80 0.90 0.96 **0.75**], **V [0.97 0.91 0.80 0.90 0.96 0.94]**.

**발견된 위험 (예측이 맞았다)**
1. **내용 통계 변화에 속는다**: dense 환경에서 통계 정렬이 적응이 필요 없는 환경을 해쳤다 (A 0.95 → 0.75, 전체 mAP50 이 A 0.745 보다 낮은 0.731). heavy_rain 도 0.97 → 0.90. 신뢰도 하락만 보고 정렬하면 "환경이 바뀌었다"와 "물체 수가 바뀌었다"를 구분하지 못한다.
2. **국소 변화는 못 고친다**: glare 에서 통계 정렬은 0.41 → 0.80 까지 올리지만 ES(0.85)·B(0.90) 보다 낮다. 그리고 서버 정련 단계를 붙여도 (D★F gate, V+) glare 가 0.80 에서 올라가지 않았다. 이 정련 단계가 왜 효과가 없었는지(요청 gate 가 열리지 않았는지, 후보가 거절됐는지)는 이벤트별로 확인하지 못했다.
3. **EdgeSync-lite 는 이 어려운 세계에서도 강했다**: 환경을 가리지 않고 평균 0.890. 단 업링크 2630KB (프로브를 윈도우마다 보냄).

**대응 (이전 구성요소 결합)**: 통계 정렬 후보를 적용하기 전에 **ES 의 프로브(Teacher 일치도)** 로 {후보, 어댑터 없음, 현재 어댑터} 를 같은 프레임 몇 장에서 비교해, 후보가 0.02 넘게 나을 때만 적용한다 (결정 한 번당 프레임 5~6장, 서명별로 한 번만, 업링크 계산에 포함). 결과 V 는 dense 0.75 → 0.94 로 회복했고, ES 와 비슷한 정확도(0.899 vs 0.890)를 업링크 668KB(ES 의 25%)로 얻었다.
- **조심할 점**: 이 비교는 ES 에게 불리하지 않게 같은 스트림·같은 프로브 수로 했지만, 차이 0.009 는 단일 스트림·단일 시드의 잡음 범위이므로 "V 가 ES 보다 낫다"가 아니라 "비슷한 정확도에서 업링크가 줄었다"로만 읽어야 한다. glare·heavy_rain 은 V 가 ES·B 보다 낮다.
- 프로브가 필요하므로 V 는 서버(Teacher)가 필요하다. 서버·전송이 전혀 없는 E 는 dense 같은 환경에서 해로울 수 있다는 것이 이번 시험의 결론이다.

**임계값 민감도 (서버 없음, 변화구간 mAP50 / 전체 mAP50)**: sig_thr 0.15 → 0.875/0.867, 0.25 → 0.852/0.731, 0.4 → 0.875/0.759; fast_margin 0.05 → 0.882/0.832, 0.1 → 0.852/0.731, 0.2 → 0.857/0.735; 정렬강도 lam 0.5 → 0.875/0.830, 1.0 → 0.852/0.731. 변화구간 값은 0.85~0.88 로 비슷하지만 **전체 mAP50 은 0.73~0.87 로 크게 흔들린다** (해로운 적용이 몇 번 들어갔느냐에 좌우). 즉 임계값을 한 값으로 고정해 쓰는 것은 위험하며, 개발 스트림에서 전체 mAP50 기준으로 골라야 한다. 엣지 추가 순전파 비율은 프레임당 0.24~0.33 (윈도우 끝 점검만 0.24, 빠른 점검 포함 0.29).

**남은 한계**: 단일 스트림·단일 시드; 장난감 Teacher 는 상한 1.0 이라 서버 어댑터의 실제 이득이 과소 평가되었을 수 있다; glare 에서 서버 단계가 효과 없던 원인 미확인; SHIFT 에서의 검증 없음.

### 5.6.4 V2: 통계 정렬 + 프로브 검증 + 일치도 기반 서버 정련 (ES 와 같은 스트림, 시드 3개)

**V2 의 구성 (모든 요소는 앞 절의 것을 재사용)**
1. 엣지 통계 정렬 추적 + 10프레임 빠른 점검 (5.6.2): 환경이 바뀌면 곧바로 라벨 없는 어댑터를 적용.
2. ES 의 프로브 (5.6.3): 적용 전에 {통계 정렬 후보, 어댑터 없음, 현재 어댑터} 를 Teacher 일치도로 비교해 후보가 0.02 넘게 나을 때만 적용.
3. **ES 의 신호로 서버 정련 요청**: 검증된 최선 후보의 Teacher 일치도가 0.85 미만이면(통계 정렬로 부족한 국소 변화 등) 그때만 서버 증류 어댑터(affine, 델타 코덱, 다양성 선별)를 요청하고, 도착한 후보는 프로브로 다시 검증한 뒤 적용. 이전에는 평균 신뢰도로 요청 조건을 걸었는데 glare 에서 열리지 않았다 (5.6.3 의 발견 2).
레지스트리와 롤백은 그대로.

**결과 (toy3, 변화구간 mAP50 / 전체 mAP50 / 업링크 KB, 평가 스트림 시드 3개; `--evseed`)**
| 시드 | ES | V2 (일치도 < 0.85) |
|---|---|---|
| 22 | 0.890 / 0.912 / 2630 | 0.935 / 0.938 / 1240 |
| 33 | 0.743 / 0.783 / 3634 | 0.856 / 0.865 / 1307 |
| 44 | **0.909** / **0.931** / 2982 | 0.861 / 0.883 / 1657 |
| 평균 | 0.847 / 0.875 / 3082 | **0.884** / **0.895** / 1401 |
- 평균은 V2 가 높고 업링크는 약 45% 이지만, **시드 44 에서는 ES 가 더 높다** (0.909 vs 0.861). 따라서 "ES 보다 확실히 좋다"고 말할 수 없고, "평균은 높고 편차가 작으며 업링크가 절반"까지만 말할 수 있다. 표본이 3개라 평균 차이(0.037)의 통계적 유의성은 확인할 수 없다.
- 편차: ES 0.74~0.91 (프로브 시점·트리거 타이밍에 크게 좌우), V2 0.86~0.94.
- 시드 44 에서 V2 가 진 구간(환경별 윈도우 평균, V2 / ES): night 0.76 / 0.81, glare 0.83 / 0.91, dense 0.89 / 0.99. dense(적응 불필요)에서 통계 정렬 후보 또는 서버 후보가 일부 해로웠다 (프로브 6장의 판정이 틀렸을 가능성). 시드 22 에서는 glare 0.91 (ES 0.85), night 0.91 (ES 0.81).
- 일치도 임계값 0.92 도 시험 (시드 22): 0.937 / 0.934 / 업링크 1838KB — 0.85 와 정확도 차이가 거의 없고 비용만 늘었다.

**남은 약점과 다음 후보**: (a) dense 같은 적응 불필요 환경에서의 잔여 해악 — 프로브 프레임 수를 늘리거나 후보 vs 어댑터 없음 비교를 두 윈도우에 걸쳐 확정(히스테리시스)하는 방법이 후보. (b) 환경이 바뀐 직후 첫 10프레임은 여전히 이전 어댑터로 처리됨. (c) 서버 정련을 통계 정렬 어댑터에서 이어서 학습하는(warm start) 방법은 미구현.

### 5.6.5 V4: 서버 정련의 학습 프레임 오염 제거 (시드 5개 모두 ES 초과 — **정정: 5.6.6 의 공정 비교에서는 성립하지 않는다**)

**시도 1 — 히스테리시스(V3) 효과 없음**: 후보 적용 뒤 새 프레임 6장으로 (적용 중 vs 어댑터 없음)을 다시 비교해 되돌리는 규칙. 시드 22·33·44·55·66 에서 V2 와 정확도가 같았고(되돌린 적이 한 번도 없음) 업링크만 약 500KB 늘었다 → 채택하지 않음 (코드는 `confirm=True` 로 남아 있음).

**원인 진단 (시드 44, V2 가 ES 에 졌던 경우)**: 이벤트 로그의 프로브 점수가 `[후보 0.80, 어댑터 없음 0.34, 현재 0.34]` 로 후보는 괜찮았는데, 같은 시점(t=109)에 서버 정련이 요청됐고 그 어댑터는 프로브에서 거절되어 밤 구간이 0.78 에 머물렀다. 서버 학습 후보 풀(100프레임)에 밤이 시작된 직후라 맑은 프레임 90장·새 환경 프레임 10장이 섞여 있었고, **다양성 선별은 "가장 다른 프레임"을 고르므로 섞여 있는 맑은 프레임을 뽑아** 서버 어댑터를 나쁘게 만든 것이다. ES 는 윈도우를 채운 뒤 선별하므로 이 문제가 없다.

**수정 (V4)**: 서버 정련 요청을 환경이 바뀐 시점(빠른 점검 시점 − 10프레임) 이후 프레임이 30장 이상 모일 때까지 미루고, 그 이후 프레임만 학습 후보로 쓴다 (`min_post=30`, `since`). 나머지는 V2 와 같다 (통계 정렬 즉시 적용 → ES 프로브 검증 → 일치도 < 0.85 일 때만 서버 정련, 후보는 프로브 재검증).

**결과 (toy3, 변화구간 mAP50 / 업링크 KB, 평가 스트림 시드 5개)**
| 시드 | ES | V2 | **V4** |
|---|---|---|---|
| 22 | 0.890 / 2630 | 0.935 / 1240 | 0.932 / 1297 |
| 33 | 0.743 / 3634 | 0.856 / 1307 | 0.915 / 1417 |
| 44 | 0.909 / 2982 | 0.861 / 1657 | 0.919 / 1930 |
| 55 | 0.861 / 2994 | 0.909 / 1209 | 0.903 / 1297 |
| 66 | 0.818 / 3269 | 0.918 / 1174 | 0.855 / 1102 |
| 평균 | 0.844 / 3102 | 0.896 / 1317 | **0.905 / 1409** |
- V4 는 **5개 시드 모두에서 ES 보다 높았다** (최소 +0.010, 평균 +0.061), 업링크는 ES 의 약 45%, 다운링크는 72~144KB (ES 는 140~288KB). 최저값도 V4 0.855 로 ES 의 최저 0.743 보다 높다.
- toy2 (환경 5종 스트림, 1회): V4 0.901 / 업 380KB / 다운 0 (ES 0.747, 이전 D 0.713).
- 한계: V4 와 V2 의 평균 차이(0.009)는 시드 변동 안이다 — V4 의 의의는 V2 가 ES 에 진 시드 44 의 원인을 제거해 5개 시드 모두에서 ES 를 넘긴 것이다. 시드 66 에서는 V2(0.918)가 V4(0.855)보다 높아 수정이 모든 시드에서 이득은 아니다. 표본 5개, 같은 Teacher·같은 학습 스트림, 합성 데이터 — SHIFT 에서의 일반화는 측정 전이다.
- 실행: `python scripts/toy_e2e.py --world toy3 --v4 --evseed 44 --methods V4` (`--min-posts` 로 대기 프레임 수).

### 5.6.6 공정 비교 감사와 V7: 입력 정합(T0) + 라벨 없는 검증 + 일관성 기반 서버 정련 + 4비트 어댑터 (모두 toy3 합성 데이터)

**5.6.5 정정.** "시드 5개 모두 ES 초과"는 ES 를 튜닝하지 않고 레지스트리 없이, 평가 시드가 겹친 채로 비교한 결과였다. 감사 후 공정 규칙은 다음과 같다: (1) ES 임계값·프로브 수·쿨다운을 **개발 시드(201)에서만** 튜닝하고 동결, (2) 평가 시드는 개발·진단에 쓰지 않은 새 시드, (3) 모든 방법이 같은 전송 압축(프로브 JPEG q40, 서버 학습 프레임 q80, 어댑터 pack16)과 같은 서버 학습 설정을 사용, (4) ES 에도 레지스트리를 준 변형을 확인(오히려 낮았음), (5) 우리 고유 요소(엣지 통계 정렬, 라벨 없는 검증, 환경 변화 후 풀 제한)만 비교 대상에서 제외. 이 규칙에서 4시드(105~108)의 ES 와 VF(5.6.5 계열)는 mAP 가 거의 같았고(VF 가 약간 낮음), VF 는 업링크 -33%, 다운링크 -40%, 적용 지연이 짧았다. 즉 5.6.5 의 "모두 ES 초과"는 폐기한다.

**성능 분해와 한계 (왜 이 방향인가).** 전체 mAP J = f_h·h + (1-f_h)·n (f_h≈0.4: 해로운 환경 비율, h: 해로운 환경 mAP, n: 나머지). ES 는 서버 증류로 n≈0.984, h≈0.83. 적용 지연 D 프레임인 한 번 갱신 모델은 h ≈ m1 - (D/L)(m1-m0) 이므로 J≥0.95 에는 D≲8 프레임이 필요하다 (ES 의 지연은 약 85 프레임).

**V7 구성 (이전 구성요소를 버리지 않고 결합).**
1. T0: 휘도 CUSUM(페이지 검정, 지연≈log(A)/I)이 켜졌을 때만 프레임 채널 평균·표준편차를 소스 통계로 맞춘다 (밤 d'=7.6, glare 5.5, sparse/dense≈0). CDF 정합·푸리에 진폭 치환·강건 모멘트는 사전 등록한 마이크로 시험에서 탈락했다.
2. 엣지 통계 정렬 + 레지스트리 + 10프레임 빠른 점검(5.6.2) 유지. 엣지 계산 지연 13프레임(통계 정렬 0.4초)을 모의에 반영(`--edge-delay 13`).
3. 라벨 없는 검증: 후보 어댑터의 평균 검출 신뢰도(후보 간 순위 일치 49/55, 실제 mAP 와 r=0.87)로 {통계 정렬, 없음, 현재} 중 선택. Teacher 업링크 0 (BoS 의 라벨 없는 mAP 추정을 단순화한 것).
4. 서버 호출 조건: 뒤집기 일관성(원본/좌우 반전 예측의 양방향 최대 IoU, r=0.68)이 소스 기준보다 0.04 넘게 낮을 때만. 후보는 학습에 쓰지 않은 프레임의 평균 신뢰도로 검증 후 적용.
5. 어댑터 부호화 `pack4`: 채널별 배율·이동 (s-1, t) 를 텐서 단위 4비트 대칭 양자화 + zlib. 오프라인에서 mAP 손실이 거의 없었다 (night/glare/rain/sparse 에서 어댑터 29.8KB → 8.0KB). **이 압축은 ES 에도 그대로 적용되므로 ES 4비트 변형(`ESq4`, `ES_Q4=1`)을 함께 비교한다.**

**결과 (새 평가 시드 117~120, 튜닝·진단에 쓰지 않음; 변화구간 mAP50 / 전체 / 업링크 KB / 다운링크 KB / 적용 지연 프레임 / 저하 프레임)**
| 방법 | mAP(변화) | mAP(전체) | 업 | 다운 | 지연 | 저하 |
|---|---|---|---|---|---|---|
| ES 기본(th=0.15, pack16) | 0.923 | 0.923 | 1202 | 36.2 | 84.5 | 112 |
| ES 4비트 | 0.929 | 0.934 | 1202 | 10.5 | 84.5 | 112 |
| **V7q (V7 + 4비트)** | **0.957** | **0.949** | **448** | 12.8 | **17.2** | **62** |
| V8 (서버가 헤드만 학습) | 0.919 | 0.916 | 339 | 2.5 | 12.2 | 100 |
- V7q 는 4개 시드 모두에서 ES(기본·4비트)보다 mAP 가 높았다 (평균 +0.033, +0.028). 업링크 -63%, 적용 지연 약 1/5, 저하 프레임 약 55%.
- **다운링크는 같은 4비트 기준에서 V7q(12.8KB)가 ES(10.5KB)보다 약간 크다.** ES 기본 압축(36.2KB)보다는 작다. 서버 호출 횟수가 ES(1회)보다 많기 때문이며 이 항목은 우위가 아니다.
- 앞선 시드 109~116 (V7d = V7q 의 pack16 판): 변화구간 mAP 0.953 (ES 기본 0.922), 업 550 (1216), 다운 69 (40), 지연 10.8 (87.6). 시드 8개 중 6개에서 ES 기본 초과 (진 시드: 109, 114).

**탈락·부정 결과 (같은 틀로 검증).**
- 서버 어댑터를 검증 없이 수락(V7e): 시드 110 에서 0.982→0.914. 검증은 필요하다.
- 서버 어댑터를 통계 정렬 어댑터와의 차이로 부호화: 차이의 RMS 가 오히려 더 컸다(0.31 vs 0.22). 서버 해와 통계 정렬 해는 가깝지 않다.
- V8 (통계 정렬을 접은 Student 에서 출발해 헤드(model.23)만 4비트 학습 → 1.7KB): 오프라인에서는 glare 0.72→0.95, rain 0.94→0.99 로 좋았지만 스트림에서는 시드 118 이 0.812 로 무너져 불안정하다. 평균 다운링크 2.5KB(ES 4비트의 1/4)이지만 mAP 가 ES 수준이라 채택하지 않았다. 원인 일부는 증류 어댑터가 평균 신뢰도 검증기에 낮게 평가되어 거절되는 것(시드 113 프레임 359: 신뢰도 0.889→0.848, 일관성 0.909→0.920). 두 지표를 OR 로 완화하면(V8b) 113 은 회복되지만 110 이 0.985→0.898 로 나빠졌다.
- 새 환경 서명마다 서버 1회(V9): 서버 어댑터가 신뢰도를 올려 다음 환경(glare) 감지가 신뢰도 기반 점검에서 놓쳐졌다(감지 지연 120프레임) → 폐기.

**남은 약점.** (1) 다운링크가 ES 4비트보다 약간 큼 — 거절될 어댑터(전체의 약 1/3)까지 받는 비용. 서버가 학습 프레임 일부를 보류해 직접 검증한 뒤 보내는 방안은 미시험. (2) glare 가 가장 약함 (통계 정렬 0.7~0.8, 서버 후 0.9~). (3) 시드 109·114 에서 ES 에 짐. (4) 합성 데이터, 같은 Teacher·단일 학습 스트림. SHIFT 에서의 일반화는 측정 전이다. (5) 엣지 추가 연산(edge_extra≈0.33~0.4 s/윈도우 상당)과 소스 프레임 60장 저장이 필요하다.

**재현.** `ES_Q4=1 python scripts/toy_e2e.py --world toy3 --evorder --evseed 117 --rseed 0 --methods "ESf,ESq4,V7q,V8(" --es "0.15,4,100,0" --v7 --edge-delay 13` (시드당 약 35분, CPU 2코어). 오프라인 부호화 측정: `scripts/codec_probe.py`, `scripts/codec_sweep.py`, `scripts/codec_warm.py`. 시간 측정: `scripts/time_bench.py`.

### 5.7 CPU 통합 시험 (GPU·SHIFT 없이 코드 전체가 도는지 확인, 선택이지만 권장)

SHIFT를 받고 Colab 시간을 쓰기 전에, **노트북 CPU에서** 합성 영상으로 `학습 → ONNX → 어댑터 → A/B/EdgeSync-lite/제안 비교`를 끝까지 실제로 돌려 코드 오류를 먼저 찾는다. 합성 데이터이므로 **성능 근거로 쓰지 않는다.**
```
python scripts/toy_e2e.py --quick      # 해상도 320, 빠른 동작 확인 (모델이 약해 mAP 비교는 무의미할 수 있음)
python scripts/toy_e2e.py              # 해상도 640, 모델을 충분히 학습 (2코어 CPU 기준 수십 분, 실측은 0.1절 표)
```
확인할 것: ① 오류 없이 `통합 시험 완료` ② `check_adapter_pipeline` 의 (2)(3)(4) 통과 ③ A의 변화구간 mAP가 Teacher(T)보다 낮음 ④ 적응하는 방법들이 A보다 높음. 결과는 `runs/toy/`에 캐시되어 다시 실행하면 학습을 건너뛴다.

---

## 6. 실험

모든 실험은 서버에서, tmux 안에서 실행한다. 한 번 실행하면 `results/<tag>.json`이 생긴다.

### 6.1 Baseline A(적응 없음)와 Teacher 단독(T) — 환경별 mAP 하락 폭 측정
무엇을 확인하나: 고정된 Student가 환경 변화에서 얼마나 떨어지는지(= "적응 필요" 구간의 정의 근거)와, 같은 스트림에서 Teacher가 낼 수 있는 상한(= 회복 기준선).
```bash
python scripts/run_experiment.py --split detector_train --method A --tag A_dtrain
python scripts/run_experiment.py --split eval --method A --tag A
python scripts/run_experiment.py --split eval --method T --tag T
python scripts/env_map.py results/A_dtrain.json results/A.json results/T.json
```
- 출력 `{'clear': 0.xx, 'rain': ..., 'night': ..., 'fog': ...}` 는 환경별 평균 윈도우 mAP50이다. **이 표를 1차 멘토링(11주차)에 가져간다.**
- 기록: `A_dtrain`의 `clear`를 기준 mAP라고 한다.

**하락 기준값(`--drop`) 정하는 법**: 맑은 윈도우들의 mAP 표준편차(`results/A_dtrain.json`의 `win_map`에서 `win_env`가 clear인 값)를 σ라 할 때, `drop`은 **2σ 이상**이면서 비·밤 중 최소 하나의 평균 하락 폭보다 작아야 한다. 이 조건으로 "정상 구간의 우연한 요동"과 "실제 하락"을 가른다. (계획서 6.1.3의 "하락 기준값은 11주차에 확정"에 해당.)

### 6.2 환경 서명 점검
```bash
python scripts/env_sig_check.py --split eval
```
같은 환경 윈도우 간 최대 거리와 다른 환경 간 최소 거리를 출력한다.
- `구분 가능 → --env-thr X 권장` 이면 이후 D·RECL 실행에 `--env-thr X`를 붙인다.
- `구분 불가` 이면(맑은 시퀀스끼리 색이 달라 흔히 이렇게 나온다) 선택지는 ① 레지스트리 없이 진행(`--no-registry`)하고 보고서에 한계를 적거나 ② `src/env.py`의 `env_signature`를 개선한다. **이 점검을 건너뛰면 "이미 본 환경으로 복귀" 기능이 동작하는지 알 수 없다.**

### 6.3 RQ1: 드리프트 감지기 학습·평가
무엇을 확인하나: 라벨 없이 얻는 신호(신뢰도 분포, 프레임 간 박스 불일치)만으로 "적응 필요"를 판정할 수 있는지.
```bash
python scripts/train_detector.py --drop 0.10        # 0.10 은 6.1절에서 정한 값으로 교체
```
출력 표의 행: 신호 `conf`(신뢰도만) / `temp`(시간적 불일치만) / `all`(둘 다) × 모델 `logreg`·`gbm`, 그리고 `DoC` 기준선. 열: Recall, 정상 구간 오경보율, 안개 구간 Recall(안개는 학습에 쓰지 않은 환경).
- **목표(계획서 6.1.2)**: Recall ≥ 0.8, 오경보율 ≤ 0.2, 안개 Recall ≥ 0.7. 마지막 줄의 회귀 상관계수는 확장 목표(예상 mAP 하락량) 근거다.
- 이 표가 **신호별 Ablation 결과**(계획서 6.1.4)이다. 표를 그대로 보고서에 쓴다.
- `라벨이 한쪽 클래스뿐입니다` → `--drop`이 너무 크거나 작다. 6.1절 값으로 다시 정한다.
- 저장되는 파일: `models/drift_logreg_{conf,temp,all}.pkl` (`run_experiment.py`가 `--cols`에 맞춰 자동으로 읽는다).

### 6.4 서버 기동 없이 첫 파이프라인 실행 (Baseline B와 D 시험)
```bash
python scripts/run_experiment.py --split eval --method D --selector diversity --budget 16 --seed 0 --delay 50 --tag D_try
```
- 서버 로그 `[server] {'sec': …, 'n_train': …, 'adapter_MB': …}` 가 적응 한 번의 실제 소요 시간과 어댑터 크기다.
- 끝에 출력되는 `이벤트:` 목록을 읽는다.
  - `trigger`(적응 요청), `apply`(어댑터 적용), `switch`(저장 어댑터로 전환), `rollback`(이전 어댑터로 복귀).
  - `trigger`가 0개 → 감지기가 한 번도 "적응 필요"를 못 냄(6.3의 Recall 확인). 너무 많음 → 오경보(`--consec 3`, 쿨다운은 `Policy(cooldown)`).
- **적용 지연(`--delay`) 확정**: `delay = ceil(sec × 10)` (10Hz에서 서버 시간이 몇 프레임에 해당하는지). 이후 모든 B·D·AMS·RECL 실행에 같은 값을 쓴다 — 그래야 방법 간 비교가 공정하다.

### 6.5 Baseline B·C와 1차 비교 (1차 멘토링 자료)
```bash
python scripts/run_experiment.py --split eval --method B --budget 16 --delay <6.4에서 정한 값> --tag B_b16_s0       # 200프레임마다 적응, 선별은 random
python scripts/run_experiment.py --split eval --method C --tent tent --tag C_tent                                   # TENT
python scripts/run_experiment.py --split eval --method C --tent bn   --tag C_bn                                     # BN-Adapt(TENT가 불안정할 때 대체)
python scripts/analyze.py
```
`results/summary.md` 표와 `results/curves.png`가 생긴다. 열의 뜻:

| 열 | 정의 |
|---|---|
| 변화구간 mAP50 | 맑음이 아닌 윈도우(비·밤·안개)의 평균 mAP50 (계획서의 "환경 변화 구간 평균 mAP") |
| 회복시간 | 환경 구간 시작부터 윈도우 mAP50이 해당 환경의 Teacher 단독 mAP50의 90%에 처음 도달할 때까지의 프레임 수. 도달 못 하면 구간 길이로 세고 `미회복`에 센다 |
| 업링크/다운링크 KB | 서버로 보낸 JPEG 바이트 / 받은 어댑터(FP16) 바이트 |
| 추론ms | 서버 CPU에서 측정한 프레임당 지연(엣지 노트북 지연은 8장에서 따로 측정) |

**TENT가 NaN을 내거나 mAP가 붕괴하면** `src/torch_detector.py`의 `lr=1e-3`을 `1e-4`로 낮춰 재시도하고, 그래도 안 되면 `--tent bn`을 Baseline C로 쓴다(계획서 6.1.1이 허용: "구현 어려울 경우 BN Adapt로 대체").

**1차 멘토링(11주차)에 가져갈 것**: 6.1 환경별 mAP 표, 6.3 감지기 표, 6.5 A/B/C/D 초안 표. 멘토와 합의해 계획서의 수치 목표(Recall 0.8 등)를 **실측값 기준으로 확정**한다(계획서 6.1.2 각주).

### 6.6 RQ2: 선별 전략 × 예산 본 실험 (12~13주차)
무엇을 확인하나: 전송 프레임 수(예산)가 같을 때 Random·Uncertainty·Diversity·Hybrid 중 어느 선별이 mAP 회복이 빠르고 큰지.

1. 예산 3단계 확정: 기본 `8 16 32`. `scripts/run_grid.sh` 맨 위 `BUDGETS`, `DELAY`를 6.4절 값으로 수정한다.
2. 총 실행 수 확인: 기본 설정으로 방법 A·T·C(2)·시드 3개 × (예산 3 × (B, AMS, RECL, 선별 4종) + Ablation 3 + 민감도 2) = 약 80회. 한 번의 실행 시간은 6.4의 `D_try` 실측치로 곱해 총 시간을 계산한다(미측정).
3. 실행:
   ```bash
   tmux new -s grid
   bash scripts/run_grid.sh 2>&1 | tee runs/grid.log
   ```
   중간에 SSH가 끊겨도 `tmux attach -t grid` 로 돌아온다. 실패한 실행만 다시 하려면 로그의 마지막 `--tag`를 찾아 그 명령만 수동으로 실행한다.
4. 판정:
   ```bash
   python scripts/analyze.py          # 전체 표 → results/summary.md
   python scripts/check_goals.py      # 계획서 목표 2, 3 달성 여부 출력
   ```
   `check_goals.py`는 (목표2) 같은 예산에서 Random보다 mAP 회복량과 회복 시간이 **모두** 나은 전략이 있는 예산 단계가 3단계 중 2단계 이상인지, (목표3) D(hybrid, B=16)가 A보다 변화구간 mAP가 5%p 이상 높은지·어댑터가 5MB 이하인지를 출력한다.
5. 포함된 추가 실험의 목적 (`ES` = EdgeSync-lite, 7장):
   - `ES_b*`, `ES_th0.08/0.25`: 비교 대상. **임계값을 바꿔 본 것 중 가장 좋은 설정과 제안을 비교**해야 "비교 대상을 불리하게 설정했다"는 반론을 막는다.
   - `Dsel_edgesync`: 제안 파이프라인에서 선별만 EdgeSync 방식으로 바꾼 것 — 제안의 이득 중 선별에서 온 몫을 분리.
   - `Dconsec2`: 제안의 감지 확정을 2개 윈도우 연속으로 늘린 변형(기본은 1개). 합성 통합 시험에서 2개 연속은 EdgeSync-lite(첫 윈도우에서 바로 요청)보다 한 윈도우(5초) 늦게 적응을 시작해 변화구간 mAP50 이 0.47(기본 1개: 0.65, ES 0.67)이었다. 오경보 억제와 반응 속도의 교환을 분리해서 본다.
   - `Dconf`, `Dtemp`: 신호 Ablation을 **시스템 수준**(mAP 회복)에서 확인.
   - `Dnoreg`: 환경별 어댑터 저장·전환의 효과(레지스트리 유무 비교).
   - `Dtau0.3`, `Dtau0.7`: 의사 라벨 신뢰도 임계값 민감도(계획서 4.1.6의 "소규모 민감도 실험").

### 6.7 목표가 미달일 때 점검 순서
계획서 목표는 **검증 대상**이므로 미달도 보고할 수 있는 결과다. 다만 구현 문제를 먼저 배제한다.

| 증상 | 확인 | 조치 |
|---|---|---|
| D의 mAP가 A와 같음 | 결과 JSON의 `n_trigger` | 0이면 감지기 문제(6.3). 0이 아니면 아래 |
| `n_trigger > 0` 인데 mAP 변화 없음 | `events`에 `apply`가 있는지, `[server] adapter_MB` | `apply` 없음: `--delay`가 스트림 구간보다 길다. 어댑터가 너무 작음: 학습 `--epochs` 증가(10 → 20) |
| D가 A보다 낮음 | `rollback` 이벤트, `--tau` | 의사 라벨 품질 문제 → `--tau 0.7`. Teacher 단독 T의 mAP가 낮으면 Teacher 재학습 |
| 안개에서만 낮음 | `train_detector.py`의 안개 Recall | 안개 일반화는 계획서가 "검증"으로 둔 항목이므로 낮아도 그대로 보고 |
| 시드 간 편차가 큼 | `analyze.py`의 시드수 열 | 시드를 5개로 늘림(`run_grid.sh`의 `SEEDS`) |

### 6.8 2차 멘토링(13주차) 자료
`results/summary.md`, `results/curves.png`, `check_goals.py` 출력, 6.3의 감지기 표, 7장의 재구성 표.

---

## 7. 기존 연구를 같은 환경에서 재구성

원칙: 모든 비교 대상은 **같은 평가 스트림, 같은 Student, 같은 Teacher, 같은 적용 지연, 같은 예산**으로 `run_experiment.py`에서 실행한다. 원 논문의 시스템을 그대로 재현하는 것이 아니라, 각 논문의 **적응 정책**만 같은 시뮬레이터에 올린다. 그래서 이름에 "-lite"를 붙인다. 원 논문의 공식 구현은 이 가이드에서 확인하거나 사용하지 않았다.

| 논문 | 실행 | 구현된 것 | 원 논문과 다른 점 |
|---|---|---|---|
| TENT [5] | `--method C --tent tent` | BN 계층의 scale/shift(γ, β)만 갱신, 입력 프레임마다 엔트로피 최소화, 스트림 내내 연속 적응 | 원 논문은 분류용. 탐지에는 "상위 100개 앵커의 클래스 시그모이드 확률의 이진 엔트로피"로 옮겼다(`torch_detector.py`) |
| BN-Adapt | `--method C --tent bn` | 현재 프레임의 BN 통계로 running 통계를 갱신(경사 없음) | TENT 대체용 (계획서 6.1.1) |
| DoC [6] | `train_detector.py`의 `DoC` 행 | 평균 신뢰도가 맑은 환경 대비 얼마나 떨어졌는지만으로 임계값 판정 | 원 논문은 분류 정확도 추정. 여기서는 판정 임계값을 학습 구간에서 정한 이진 감지기로 변형 |
| AutoEval [7] | `train_detector.py`의 회귀 상관계수 | 윈도우 통계 → 예상 mAP 하락량 회귀(Ridge) | 원 논문의 합성 변환 메타데이터셋은 만들지 않았다("AutoEval-lite", 충실도 낮음) |
| Ekya [1] | `--method B` | 고정 주기(200프레임)마다 재학습 요청하는 윈도우 재학습 정책 | Ekya의 핵심인 재학습·추론 GPU 자원 배분 스케줄러는 재구성하지 않았다 (단일 스트림, 서버 1대 가정) |
| AMS [2] | `--method AMS` | 장면 변화량이 클수록 갱신 간격을 `period/(1+20×변화량)`으로 줄이고, 서버가 Teacher 의사 라벨로 모델 일부(헤드)를 학습해 전송 | 원 논문의 파라미터 부분집합 선택·비동기 갱신은 구현하지 않았다 |
| RECL [3] | `--method RECL` | 주기적 재학습 + 이미 본 환경이면 저장 어댑터를 재사용(레지스트리) | 원 논문의 모델 zoo에서 최적 모델을 평가·선택하는 부분은 환경 서명 매칭으로 단순화 |
| **EdgeSync [11]** (arXiv 2510.21781, 2025-10) — **발표에서 나란히 비교하는 대상** | `--method ES` (데모: 가운데 패널) | ① 엣지 프레임을 `Q = α·출력엔트로피 + β·최신성`으로 점수화해 상위 k개 업로드(Eq.1–3, α=β=0.5로 둠) ② 윈도우마다 4프레임을 서버 Teacher에 보내 일치도를 받고, 최근 10윈도우 최대치 대비 하락(긴급도) ≥ θ(기본 0.15)이면 갱신 요청 ③ 서버가 변경 파라미터만 전송 | 논문의 다중 엣지 긴급도 순위·베이지안 최적화 하이퍼파라미터·조기 종료는 구현하지 않았다. α, β, θ, 프로브 4장은 논문 발췌에서 확인되지 않아 **내가 정한 값**이다(θ는 0.08/0.15/0.25로 민감도 실험). Eq.2의 부호가 모호해 최신 우선으로 구현했다. 논문의 공식 구현은 확인하지 못했고 논문 본문(arXiv HTML)의 서술만 근거로 했다. 같은 파이프라인(서버·어댑터·롤백·적용 지연)을 공유하고 **트리거와 선별만 다르다** |
| Shoggoth [4] | **재구성하지 않음** | - | 엣지 측 소배치 학습과 적응형 샘플링이 별도 구성이다. 보고서에 "미재구성(엣지 단독 적응은 C, 적응형 샘플링은 AMS-lite로 부분 대응)"이라고 적는다 |

보고서 표에는 "방법 / 구현 수준(전체·-lite·미구현)"을 함께 적는다. 참고문헌 [11]에 `R. Dong et al., EdgeSync: Accelerating Edge-Model Updates for Data Drift through Adaptive Continuous Learning, arXiv:2510.21781, 2025` 를 추가한다.

**왜 EdgeSync를 골랐나**: (1) 구조가 우리와 같다(경량 엣지 모델, 서버의 큰 모델이 의사 라벨, 변경된 파라미터만 엣지로 전송). (2) 2025년 10월 공개로 이 목록에서 가장 최근이다. (3) 우리와 다른 점이 명확하다 — EdgeSync는 갱신 시점을 정하려고 **프레임을 계속 서버에 보내 Teacher와 비교**(업링크 상시 소비)하고, 환경별 어댑터를 저장하지 않으며, 선별이 엔트로피+최신성이다. 우리는 **라벨·서버 비교 없이 엣지 신호만으로** 감지하고, 환경별 어댑터를 저장한다. 그래서 "업링크가 적으면서도 같거나 더 빨리 회복하는가"가 질문이 된다. 단, 이 우위는 **실험에서 확인하는 것이지 미리 보장되지 않는다.** `check_goals.py`의 "발표용" 절이 EdgeSync에 가장 유리한 θ와 비교한 결과를 출력하고, 우세하지 않으면 그렇게 출력한다.

---

## 8. 엣지 노트북: 경량화·지연 측정·실시간 데모 (이은민)

### 8.1 ONNX 모델 크기와 지연 측정 (노트북에서)
```bash
python scripts/bench_latency.py models/student.onnx --n 200
python scripts/bench_latency.py models/student.onnx --n 200 --threads 4
```
출력: 모델 MB, 프레임당 전체 지연의 평균·p50·p95(ms)와 환산 FPS(전처리+추론+후처리), 순수 `session.run` 평균. **노트북 기종, CPU 종류, 스레드 수를 결과와 함께 기록한다**(계획서 6.1.2 "엣지 모사 환경 추론 지연").

### 8.2 INT8 양자화 (경량화 보고용)
서버에서(학습 이미지 약 100장으로 보정):
```bash
python scripts/quantize_onnx.py models/student.onnx data/yolo/stream_train/images
python scripts/run_experiment.py --split eval --method A --onnx models/student_int8.onnx --tag A_int8
python scripts/bench_latency.py models/student_int8.onnx --n 200
```
- `A_int8`과 `A`의 mAP 차이, 모델 크기, 지연을 한 표로 비교한다.
- **INT8 모델은 어댑터 교체 파이프라인에 쓰지 않는다**(가중치가 INT8이라 FP32 어댑터를 끼울 수 없다). 적응 파이프라인은 FP32 기본 모델로 돌리고, INT8은 "엣지 경량화 효과" 보고용이다. 이 한계를 보고서에 적는다.

### 8.3 3-way 실시간 데모 (노트북에서 실행. 서버는 `--device cpu`): 적응 없음 / EdgeSync-lite / 제안 (같은 영상, 같은 서버)

> **빠른 길:** 처음이면 이 절 대신 `QUICKSTART.md` 를 따라 한다. 데모의 "제안(Ours)" 에이전트는 `src/v7_agent.py`(최종판 V7q, 5.6.6)이고, `--source toy --models models_toy` 로 SHIFT 없이 합성 영상 데모를 돌린다.
> 서버 터미널에는 단계별 로그(`BOOT/PROBE/RECV/TEACH/REPLAY/TRAIN/EXPORT/DELTA/CODEC/DONE/SEND`)가 찍히고 `logs/server.log` 에도 저장된다. 옵션: `scripts/serve.py --models --device --codec pack4|pack16 --port --host --epochs --tau --quiet --log`.
> SHIFT 공식 평가(`run_experiment.py`)에는 `--method V7` 을 추가했지만 **SHIFT 에서는 아직 한 번도 돌려 보지 않았다**(합성 데이터 결과만 있음).


**화면 구성** (한 창, 위에서 아래로)
- 맨 위: 제목과 오른쪽 위 **현재 환경(정답)** — 맑음(낮) / 비 / 밤 / 안개
- 가운데: 같은 프레임을 세 시스템에 동시에 넣은 **패널 3개**. 각 패널에 탐지 박스, 아래쪽 **상태 칩**(적응 없음 / 정상 / **성능 저하 감지!** / **서버 적응 중...** / **회복됨** / 저장 어댑터 전환 / 롤백 / 네트워크 단절·큐 보관), 그 아래에 현재 윈도우 mAP50·적응 횟수·업링크·다운링크 KB
- 맨 아래: **윈도우별 mAP50 곡선**(시스템별 색, 배경 띠 = 환경 구간, ▲ = 적응 요청, ● = 어댑터 적용). 비·밤으로 바뀌면 곡선이 내려가고, 제안이 먼저 ▲를 찍고 곡선이 올라오는 모습이 발표의 핵심 장면이다.

**준비 (처음 1회)**: 0.5절의 "노트북 준비"를 끝내 `models/student.onnx`(노트북에서 내보낸 것), `models/teacher.pt`, `models/student.pt`, `models/drift_logreg_all.pkl`, `configs/streams.json`, `data/yolo/stream_eval/` 이 있으면 된다(서버와 엣지가 한 노트북이므로 `scp`·SSH 터널은 필요 없다).

**실행** (터미널 2개)
1. **터미널 1 — 서버**: 저장소 폴더에서 가상환경을 켜고 (NVIDIA GPU가 있으면 `--device 0`, 없으면 `--device cpu --epochs 5`)
   ```
   python scripts/serve.py --device cpu --epochs 5 --port 8000
   ```
   `[server] device=cpu` 가 출력되고 `Uvicorn running on http://127.0.0.1:8000` 이 보이면 준비 완료. 이 창은 닫지 않는다.
2. **터미널 2 — 연결 확인**:
   ```
   python -c "import requests; print(requests.get('http://127.0.0.1:8000/v1/health').json())"
   ```
   `{'ok': True}` 가 나와야 한다.
3. **터미널 2 — 데모 실행**:
   ```
   python scripts/demo.py --source stream --server http://127.0.0.1:8000
   ```
   창이 뜨면 그대로 재생된다. 키: **q** 종료 / **스페이스** 일시정지(다시 누르면 재생) / **r** 모든 시스템의 어댑터 초기화 / **s** 스크린샷 저장.

**영상 입력 3가지**
| 입력 | 명령 | 곡선의 y축 |
|---|---|---|
| SHIFT 평가 스트림(권장, 정답이 있어 mAP 비교 가능) | `--source stream` | 윈도우 mAP50 |
| 녹화한 주행 영상 파일 | `--source C:\\videos\\drive.mp4` (Windows는 `\\` 두 번, 또는 `/`) | 윈도우 평균 신뢰도 (정답 없음) |
| 노트북 카메라 | `--source 0` (안 열리면 `1`) | 윈도우 평균 신뢰도 |

영상 파일·카메라에는 정답이 없어 mAP를 그릴 수 없다. 대신 평균 신뢰도 곡선이 떨어지고(드리프트) 올라오는(회복) 모습을 보여준다. 모델이 SHIFT 합성 영상으로 학습됐으므로 실제 영상에서는 성능이 낮다. **발표 본 시연은 `--source stream`** 으로 하고, 영상·카메라는 "실제 영상에도 연결된다"는 보조 시연으로 쓴다. 영상 파일에서 밤·비 구간을 보이려면 날씨가 바뀌는 영상이 필요하다(없으면 화면 밝기를 낮춘 영상을 편집해 쓸 수 있으나, 이는 합성 변화임을 밝힌다).

**발표용 옵션**
- 백업 영상 녹화: `--record results/demo.mp4` (발표 전에 한 번 끝까지 돌려 mp4를 만들어 둔다. 현장에서 서버가 안 돌 때 이 영상을 재생한다.)
- 느린 노트북: `--scale 0.5 --fps 5`, 또는 `--agents static,ours` / `--agents ours,edgesync`
- 화면 없이 확인만: `--headless --snap-dir results/snap` (250프레임마다 PNG 저장, 마지막은 `final.png`)
- 실행이 끝나면 `results/demo_summary.json` 에 시스템별 곡선·업링크·적응 횟수가 저장된다. **이 값은 실시간 서버 소요 시간에 따라 달라지므로 보고서의 공식 수치는 6.6절 `run_experiment` 결과를 쓴다.**

**공정성**: 세 패널은 같은 프레임, 같은 Student, 같은 서버(Teacher, 어댑터 학습, 같은 예산 16)를 쓴다. 서버가 GPU 1개를 순서대로 쓰므로(락) 두 시스템이 동시에 요청하면 한쪽이 대기한다. 서버의 에이전트별 학습 버퍼는 분리돼 있다(요청 헤더 `client`).

**시연 시나리오**(스트림 순서 = 맑음→비→맑음→밤→맑음→안개→맑음): ① 맑음에서는 세 곡선이 같다 ② 비로 바뀌면 곡선이 모두 하락, 제안이 **성능 저하 감지!** → 적응 요청 → **서버 적응 중...** → **회복됨**, EdgeSync는 프로브 일치도가 임계값 아래로 내려가야 요청 ③ 맑음으로 돌아오면 제안 패널에 **저장 어댑터 전환**(재학습 없음) ④ 밤·안개에서 반복. 업링크 KB 숫자도 비교한다(EdgeSync는 프로브 때문에 평상시에도 올라간다). **리허설을 최소 2번 한다**: 리허설 결과 제안이 우세하지 않으면 6.7절의 점검 순서로 원인을 찾고, 그래도 안 되면 결과 그대로 보고하되 5·6장의 개선 항목(`--tau`, 예산, `--epochs`, 감지기 `--drop`)을 조정해 다시 돌린다.

**네트워크 단절 시연**: 서버 창(터미널 1)에서 **Ctrl+C**로 서버를 끄면, 다음 적응 시점에 해당 시스템의 상태 칩이 **네트워크 단절·큐 보관**으로 바뀐다. 서버를 다시 켜면(`serve.py` 재실행, Teacher 로딩 때문에 수십 초 걸릴 수 있음) 큐에 있던 프레임이 전송되어 **서버 적응 중...** 으로 돌아온다. 단, 서버를 재시작하면 서버 안의 어댑터 저장소는 비워진다.

창이 안 뜨고 `cv2.imshow` 오류 → `pip uninstall -y opencv-python-headless && pip install opencv-python`. 한글이 `□`로 보이면 한글 폰트를 못 찾은 것이므로 영문 라벨로 자동 전환된다(동작에는 영향 없음). 한글을 보려면 폰트 경로를 환경변수로 지정한다: Windows PowerShell `$env:EDGE_ADAPT_FONT="C:/Windows/Fonts/malgun.ttf"`.

### 8.4 Android 앱 (확장 목표)
이 저장소에는 포함하지 않았다. 계획서가 확장 목표(15주차 여유 시)로 둔 항목이므로, 핵심 목표 판정(6.6)이 끝난 뒤 별도로 진행한다.

---

## 9. 마무리 (14~16주차)

### 9.1 최종 산출물 체크리스트
- [ ] `results/summary.md`, `results/curves.png`, `check_goals.py` 출력 (A~D 종합 비교)
- [ ] 6.3 감지기 표(신호 Ablation, DoC 기준선, 안개 Recall)
- [ ] 7장 재구성 표(구현 수준 명시)
- [ ] 8.1~8.2 엣지 지연·모델 크기·INT8 표 (노트북 기종 명시)
- [ ] 데모 리허설 영상 1개(시연 실패 대비)
- [ ] 0.4절 #1, #2의 한계를 보고서 "한계" 절에 기술
- [ ] 멘토링 피드백 반영 내역을 GitHub에 기록 (계획서 5.2): 저장소 **Issues** 탭 → **New issue** → 제목 `1차 멘토링 피드백`, 내용에 피드백과 반영 커밋 링크
- [ ] 재현성 정보 공개: `data/lists/*.txt`(시퀀스 목록), `configs/streams.json`, 난수 시드(`--seed`), 전처리 스크립트 (계획서 4.3.5). `data/lists/`는 `.gitignore`의 `data/`에 포함되므로 **`git add -f data/lists`** 로 강제로 추가한다

### 9.2 저장소 공개 (16주차)
1. 공개 전 점검:
   ```bash
   git ls-files | head -100          # data/, models/, *.pt, *.onnx 가 없어야 한다
   git log --stat | grep -E "\.(pt|onnx|tar|zip)" | head      # 과거 커밋에 큰 파일이 들어간 적이 없는지
   ```
2. 저장소 페이지 **Settings** → 맨 아래 **Danger Zone** → **Change repository visibility** → **Change to public** → 안내에 따라 저장소 이름을 입력하고 확인.
3. 저장소 첫 화면 오른쪽 **About**의 톱니바퀴 클릭 → Description, Topics(`edge-ai`, `video-analytics`, `continual-learning`, `test-time-adaptation`) 입력 → **Save changes**.
4. `README.md`에 재현 순서(이 문서 4~6장의 명령)를 링크하고, SHIFT 데이터는 직접 받아야 한다고 적는다.

---

## 10. 문제 해결

| 증상 | 원인 | 조치 |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | 저장소 루트가 아닌 곳에서 실행 | `cd ~/edge-adapt` 후 실행 (`python scripts/...` 형태 유지) |
| `열을 찾지 못함: [...]` (`select_seqs.py`) | `seq.csv` 열 이름이 다름 | 4.3절 |
| `이미지 0장입니다` (`shift_to_yolo.py`) | `--images` 경로·zip/tar 내부 경로·`--seqs` 이름 불일치 | `inspect_shift.py` 출력과 대조 |
| `CLASSES 에 없는 카테고리` 경고 | 카테고리 이름이 다름 | `src/common.py`의 `CLASSES` 수정 |
| `yolo` 명령 없음 | 가상환경 비활성 | `source .venv/bin/activate` |
| `CUDA out of memory` | batch 과대 | `batch`를 절반으로 |
| 학습 시작 시 다운로드 실패(글꼴, `yolo11n.pt`) | 서버 외부 접속 제한 | 노트북에서 받아 `scp` 후 해당 위치에 복사 |
| `어댑터가 비어 있음` (서버 로그) | 학습이 헤드를 못 바꿨거나 ONNX 이름 불일치 | 5.4 `check_adapter_pipeline.py` |
| `KeyError: 모델에 없는 initializer` | 어댑터 이름이 엣지 모델과 다름 | 엣지 `models/student.onnx`가 서버의 것과 같은 파일인지 `sha256sum` 비교 |
| 서버 오류 `state: error` | 서버 학습 예외 | 서버 터미널(8.3절 터미널 1)의 로그 확인. 엣지는 기존 어댑터로 계속 동작한다 |
| 데모에서 `Connection refused` | 서버 미기동 또는 포트 충돌 | 8.3절 터미널 1부터 다시 시작. 포트가 쓰이고 있으면 `--port 8001` 로 바꾸고 데모의 `--server` 도 같이 바꾼다 |
| PowerShell에서 `Activate.ps1` 차단 | 실행 정책 | `Set-ExecutionPolicy -Scope Process Bypass` |
| ONNX Runtime이 느림 | 스레드 설정, 전원 모드 | `bench_latency.py --threads`, 노트북 전원 연결 및 고성능 모드 |
| Hugging Face `401`/`403` | 로그인 필요 | 4.1절 토큰 |
| `git push` 거부 (`Permission denied (publickey)`) | SSH 키 미등록 | 2.1절 6번 `ssh -T git@github.com` 으로 확인 |
| TENT 결과가 NaN | 학습률 과대 | 6.5절 |

---

## 부록: 저장소 구성

```
src/
  demo_core.py, ui.py  3-way 데모 루프 / 화면 렌더러(한글은 PIL)
  common.py         상수(클래스 6개, 해상도 640x400 → ONNX 입력 640x416)
  mapeval.py        mAP50/mAP50-95, 윈도우별 mAP (외부 라이브러리 없음)
  onnx_detector.py  ONNX 추론, 어댑터 교체(apply_adapter)/복원(reset_adapter)
  signals.py        드리프트 신호(신뢰도 분포, 프레임 간 박스 불일치)
  selectors.py      프레임 선별 7종 (random / uncertainty / diversity / hybrid / entropy / ent_hybrid / edgesync)
  drift_model.py    감지기(logreg·gbm), 회귀, DoC 기준선
  env.py            환경 서명, 환경별 어댑터 레지스트리
  sim.py            스트림 시뮬레이션(정책: none/periodic/ams/drift)
  server_core.py    Teacher 의사 라벨 + 헤드 학습 + 어댑터 추출
  torch_detector.py TENT / BN-Adapt / Teacher 단독
  api.py, edge_client.py   FastAPI 서버 / 엣지 클라이언트(단절 시 큐 보관)
  stream.py, analysis.py   평가 스트림 로더 / 회복 시간 계산
scripts/
  download_shift_hf.py → select_seqs.py → extract_seqs.py → shift_to_yolo.py → vis_check.py   (4장)
  make_yaml.py, export_base.py, check_adapter_pipeline.py                                      (5장)
  run_experiment.py, env_map.py, env_sig_check.py, train_detector.py, run_grid.sh,
  analyze.py, check_goals.py                                                                   (6장)
  quantize_onnx.py, bench_latency.py, serve.py, demo.py                                        (8장)
tests/              GPU·데이터 없이 도는 단위 테스트 15개 (EdgeSync-lite 정책, 3-way 데모 루프 헤드리스, UI 렌더, 프로브 API 포함)
```

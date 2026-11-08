# QUICKSTART — 아무것도 설치 안 된 Windows 노트북에서 "그대로 따라하기"

## 읽는 법 (먼저 1분)
- 회색 상자(```) 안의 글자는 **한 줄씩 복사해서 PowerShell 창에 붙여넣고 Enter** 를 누른다. 상자 안에 줄이 여러 개면 **한 줄 붙여넣고 Enter → 끝나길 기다림 → 다음 줄** 순서다.
- PowerShell 창에서 붙여넣기: **마우스 오른쪽 버튼 한 번 클릭**(또는 Ctrl+V). 창이 파란색(또는 검은색) 글자 창이다.
- 각 단계 끝에 **"이렇게 보이면 성공"** 이 있다. 안 보이면 다음으로 넘어가지 말고 맨 아래 "막힐 때" 표를 본다.
- 정직한 전제: **A 경로(합성 데모)는 리눅스 CPU 에서 끝까지 돌려 확인**했다. Windows 에서 직접 돌려 보지는 못했고, **B 경로(SHIFT + Colab)와 Colab 노트북은 실제 Colab 에서 시험하지 못했다.**
- 이 노트북은 NVIDIA GPU 가 없어도 된다(CPU 로 동작).
- **메모리(RAM):** 한 대에서 서버+엣지를 같이 돌릴 때 실측 최대 약 **5GB**(서버 약 3GB + 데모 약 2GB)를 쓴다. **RAM 16GB 권장, 8GB 는 크롬 등을 모두 끄고 해야 하며 부족하면 프로그램이 갑자기 꺼진다.** RAM 이 부족하면 아래 "노트북 2대" 방식(서버 노트북 약 3GB, 엣지 노트북 약 2GB)을 쓴다.
- **디스크:** 가상환경(torch 등)에 약 3~5GB 필요.

---

## 단계 0. Python, Git 설치 (한 번만, 약 10분)

**0-1. Python 3.11 설치**
1. 브라우저(크롬)에서 이 주소를 연다: `https://www.python.org/downloads/release/python-3119/`
2. 페이지를 아래로 내려 **"Files" 표**에서 **"Windows installer (64-bit)"** 를 클릭 → 다운로드된 파일(`python-3.11.9-amd64.exe`)을 더블클릭.
3. 설치 창 **맨 아래의 "Add python.exe to PATH" 체크박스를 반드시 체크**한 뒤 **"Install Now"** 클릭 → "Setup was successful" 이 뜨면 **Close**.

**0-2. Git 설치**
1. 주소: `https://git-scm.com/download/win` → 맨 위 **"Click here to download"** 클릭 → 받은 파일 더블클릭.
2. 설치 창에서 **아무것도 바꾸지 말고 "Next"를 계속 눌러** 마지막에 **Install** → **Finish**.

**0-3. PowerShell 열기 (앞으로 계속 쓴다)**
1. 키보드의 **Windows 키**(왼쪽 아래 Ctrl 옆 깃발 모양)를 누른다.
2. 바로 `powershell` 이라고 타자 → 목록에 **"Windows PowerShell"** 이 뜨면 **Enter**.
3. 파란 창이 뜨고 `PS C:\Users\이름>` 같은 글자가 보이면 열린 것이다. (이 창을 "터미널"이라고 부른다.)

**0-4. 설치 확인** — 터미널에 한 줄씩:
```
python --version
git --version
```
**이렇게 보이면 성공:** `Python 3.11.9`, `git version 2.xx...`
(안 되면: 터미널을 **닫고 새로 열어서** 다시. 그래도 안 되면 Python 을 지우고 0-1 을 다시 하면서 PATH 체크 확인.)

---

## 단계 1. 코드 폴더 만들기

1. 내가 받은 **`edge-adapt.zip`** 파일(다운로드 폴더에 있음)을 **마우스 오른쪽 클릭 → "모두 압축 풀기…"**.
2. 경로 입력 칸을 지우고 **`C:\`** 만 입력 → **압축 풀기**. (그러면 `C:\edge-adapt` 폴더가 생긴다.)
3. **파일 탐색기**(노란 폴더 아이콘)에서 `내 PC → 로컬 디스크(C:) → edge-adapt` 를 열어 **`src`, `scripts`, `QUICKSTART.md` 가 바로 보이는지** 확인한다.
   - 만약 `edge-adapt` 안에 또 `edge-adapt` 폴더가 있고 그 안에 `src` 가 있다면, **안쪽 폴더**가 진짜다. 아래 모든 `C:\edge-adapt` 를 `C:\edge-adapt\edge-adapt` 로 바꿔 읽는다.

터미널에서 그 폴더로 이동:
```
cd C:\edge-adapt
dir
```
**이렇게 보이면 성공:** 목록에 `src`, `scripts`, `models_toy`, `requirements-edge.txt` 가 보인다.

> 앞으로 터미널을 새로 열 때마다 **첫 줄은 항상 `cd C:\edge-adapt`** 이다.

---

## 단계 2. 파이썬 환경 만들기 (한 번만, 15~25분, 인터넷 필요)

터미널에 **한 줄씩**:
```
python -m venv .venv
```
(아무 글자도 안 나오고 몇 초 뒤 `PS C:\edge-adapt>` 로 돌아오면 정상.)
```
.venv\Scripts\Activate.ps1
```
**이렇게 보이면 성공:** 줄 맨 앞에 **`(.venv)`** 가 붙는다.

**빨간 글씨로 "이 시스템에서 스크립트를 실행할 수 없으므로…" 가 나오면:** 아래 한 줄을 실행하고 `Y` 입력 후 Enter, 그다음 위의 `Activate.ps1` 줄을 다시 실행.
```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

`(.venv)` 가 보이는 상태에서 **한 줄씩** (각 줄은 몇 분 걸린다. 글자가 계속 올라가는 건 정상):
```
python -m pip install -U pip
pip install -r requirements-edge.txt
pip install torch==2.5.1 torchvision==0.20.1
pip install ultralytics==8.3.40 fastapi "uvicorn[standard]" python-multipart onnxslim
```
**이렇게 보이면 성공:** 마지막 줄 끝에 `Successfully installed ...` 가 나온다.

확인:
```
python -c "import torch, ultralytics, cv2, onnxruntime; print('환경 OK')"
```
**성공:** `환경 OK` (앞에 노란 경고가 한두 줄 있어도 괜찮다).

> 앞으로 터미널을 새로 열 때마다 **`cd C:\edge-adapt` → `.venv\Scripts\Activate.ps1`** 두 줄을 먼저 실행해 `(.venv)` 가 보이게 한다.

---

## 단계 3. GitHub 저장소 만들고 올리기 (5~10분)

**3-1. GitHub 에서 빈 저장소 만들기 (브라우저)**
1. `https://github.com` 접속 → 로그인(없으면 오른쪽 위 **Sign up**).
2. 오른쪽 위 **`+` 버튼 → New repository**.
3. **Repository name** 칸에 `edge-adapt` 입력, **Private** 선택. **그 아래 "Add a README / .gitignore / license" 는 모두 체크하지 않는다.** → 초록색 **Create repository**.
4. 화면에 `https://github.com/내아이디/edge-adapt.git` 같은 주소가 보인다. 이 주소를 아래 3-2 의 `<주소>` 자리에 쓴다.

**3-2. 터미널에서 올리기** (`cd C:\edge-adapt` 상태)
```
git init
git config user.name "이은민"
git config user.email "leunmin3@gmail.com"
git add .
git commit -m "init"
git branch -M main
```
`<주소>` 를 3-1 의 주소로 바꿔서:
```
git remote add origin <주소>
git push -u origin main
```
처음 push 하면 **브라우저 로그인 창**이 뜬다 → **"Sign in with your browser"** 클릭 → GitHub 에서 **Authorize** → 터미널로 돌아온다.
**이렇게 보이면 성공:** 터미널에 `Branch 'main' set up to track ...` 가 나오고, GitHub 페이지를 새로고침하면 파일들이 보인다.

(`data/ models/ runs/` 는 `.gitignore` 로 자동 제외, 작은 데모 모델 `models_toy`(약 22MB)만 올라간다.)
**팀원 초대:** GitHub 저장소 페이지 → **Settings → Collaborators → Add people** → 팀원 아이디 입력.

---

## 단계 4. 5분 데모 — 합성 영상, 노트북 1대, 터미널 2개

PowerShell 창을 **2개** 쓴다. 이름을 **[서버 창]**(클라우드 역할), **[엣지 창]**(엣지 역할)이라고 부른다.

**4-1. [서버 창] 열고 서버 켜기**
1. Windows 키 → `powershell` → Enter (창 1개 더 열림 = 서버 창).
2. 한 줄씩:
```
cd C:\edge-adapt
.venv\Scripts\Activate.ps1
python scripts/serve.py --models models_toy --device cpu --codec pack4
```
3. **이렇게 보이면 성공:** 색깔 글자로 `BOOT | ... 대기 중` 이 나오고 **커서가 멈춰 있다.** 정상이다(서버는 계속 켜져 있어야 한다). **이 창을 닫거나 Ctrl+C 를 누르지 않는다.**
4. 처음 한 번 **"Windows Defender 방화벽" 창**이 뜨면 **"액세스 허용"** 클릭.

**4-2. [엣지 창] 열고 데모 화면 켜기**
1. Windows 키 → `powershell` → Enter (또 한 개 더 열림).
2. 한 줄씩:
```
cd C:\edge-adapt
.venv\Scripts\Activate.ps1
python scripts/demo.py --source toy --models models_toy --server http://127.0.0.1:8000 --fps 2
```
3. 30초~1분 뒤 **영상 창**이 뜬다: 위에 3칸(적응 없음 / EdgeSync-lite / 제안), 아래에 윈도우별 mAP 곡선. 환경 이름이 오른쪽 위에 바뀐다(밤, 눈부심, 폭우 …).
   - 터미널에 노란/흰 글자(Ultralytics, ONNX export …)가 많이 올라오는 것은 정상이다.
4. 영상 창에서 쓰는 키: **`q` 종료 · 스페이스 일시정지 · `r` 어댑터 초기화 · `s` 스냅샷 저장.**

**4-3. [서버 창]에서 볼 것 — 클라우드가 어떻게 적응하는지**
환경이 바뀌면 엣지 화면의 "제안" 칸이 **"서버 적응 중…"(파란색)** 이 되고, **서버 창에 로그가 순서대로 흐른다:**

| 로그 단계 | 의미 |
|---|---|
| `RECV` | 엣지가 보낸 프레임 장수, JPEG KB, 누적 업링크 |
| `TEACH` | 큰 모델(Teacher)이 만든 정답 박스 수, 평균 신뢰도, 클래스 분포 |
| `REPLAY` | 학습에 쓰는 버퍼 프레임 수 |
| `TRAIN` | 학습 손실이 에폭마다 줄어드는 과정 |
| `DELTA` | 원래 모델 대비 어떤 층이 얼마나 바뀌었는지 |
| `CODEC` | 10MB 어댑터를 몇 KB 로 압축했는지 |
| `SEND` | 엣지로 보낸 KB, 누적 다운링크 |

**시간:** 적응 1회가 **CPU 에서 약 3~10분** 걸린다(내가 쓴 2코어 리눅스에서 3분). 서버와 엣지가 같은 노트북 CPU 를 나눠 쓰므로 더 걸릴 수 있다. 그동안 "서버 적응 중…" 이 켜져 있는 것이 **정상**이다. `--fps 2` 로 영상을 느리게 재생해 적응이 영상 안에서 끝나도록 했다. 끝나면 칸이 초록(정상)으로 바뀌고 곡선이 올라간다.
**끝내기:** 영상이 끝나면 "끝났습니다" 가 나온다 → 영상 창에서 아무 키나 누르면 종료되고 요약이 `results\demo_summary.json` 에 저장된다. 서버 창은 **Ctrl+C**.

**창 없이 자동 확인만 하고 싶으면** [엣지 창]에서: `python scripts/demo.py --source toy --models models_toy --server http://127.0.0.1:8000 --headless --snap-dir snaps` → `snaps\final.png` 에 마지막 화면이 저장된다.

### (선택) 노트북 2대로 하기
1. **서버 노트북**에서 서버 실행 줄 끝에 `--host 0.0.0.0` 을 붙인다: `python scripts/serve.py --models models_toy --device cpu --codec pack4 --host 0.0.0.0`
2. 서버 노트북 터미널에서 `ipconfig` → **"IPv4 주소"**(예: `192.168.0.12`) 를 적는다.
3. **엣지 노트북**에서(단계 0~2 를 똑같이 한 뒤): `python scripts/demo.py --source toy --models models_toy --server http://192.168.0.12:8000 --fps 2` (IP 는 위에서 적은 값).
4. 두 노트북이 **같은 와이파이**여야 한다. 서버에 비밀번호가 없으므로 **집/연구실처럼 믿을 수 있는 와이파이에서만** 한다.

> **Colab 은 이 실시간 데모에 필요 없다.** Colab 은 아래 단계 5(SHIFT 실데이터 준비·모델 학습)에서만 쓴다.

---

## 단계 5. (단계 4가 끝난 뒤) SHIFT 실데이터 — Colab 에서 모델을 만들고 비교 실험 *(미검증: 이 경로는 아직 한 번도 실제로 돌려 보지 못했다)*

**먼저 알아둘 것**
- SHIFT 데이터는 Colab 안에서 받는다(노트북이 Hugging Face 미러에서 내려받음; 내 노트북에 따로 받을 필요 없음). 크기는 셀 5 가 다운로드 전에 알려준다. 공식 서버가 내려가 있을 수 있어 이산(10Hz) 비디오를 환경 순서대로 이어 붙인 스트림을 쓴다(GUIDE 0.4절).
- 노트북에는 실험 두 가지가 들어 있다: ① 셀 1~13 데이터 준비·Student/Teacher 학습 ② 셀 27~30 본 실험(`run_grid.sh`). 본 실험에는 이제 **ES(기본), ES(4비트), V7(제안 최종판)** 이 모두 들어 있어 같은 조건으로 비교된다.
- 시간: `PROFILE = "free"`(셀 6, 기본)이 무료 Colab 용 축소판이다. 무료 GPU 는 사용 시간 제한이 있어 중간에 끊길 수 있는데, 학습 결과가 구글 드라이브에 저장되고 다시 실행하면 이어서 한다.

**5-1. 업로드와 열기**
1. `https://drive.google.com` → **+ 새로 만들기 → 파일 업로드** → `edge-adapt.zip` 선택. **내 드라이브 맨 위(폴더 안이 아님)** 에 있어야 한다.
2. `https://colab.research.google.com` → **파일 → 노트북 업로드 → "업로드" 탭 → `C:\edge-adapt\colab\edge_adapt_colab.ipynb`**.
3. **런타임 → 런타임 유형 변경 → T4 GPU → 저장**.

**5-2. 실행** — 셀 왼쪽 **▶ 버튼을 위에서 아래로 한 칸씩** 누른다. 한 셀이 끝나(초록 체크) 다음 셀로 간다. "드라이브 연결 허용" 창이 뜨면 **허용**.
- 셀 10(`--dry-run`)에서 **다운로드 크기와 남은 디스크**를 확인한다. 공간이 모자라면 멈추고 알려 달라.
- 셀 13은 **변환 결과를 눈으로 확인**하는 셀이다. 사진에 박스가 물체 위에 정확히 그려져 있는지 본다(어긋나 있으면 이후 결과는 전부 무의미하다).
- 셀 20은 "필수 관문"이다. 여기서 오류가 나면 더 진행하지 않는다.
- 막힌 셀 번호와 빨간 오류 마지막 줄을 알려 주면 고친다.

**5-3. 결과 보는 곳**
- 셀 30 이 `results/summary.md`(방법별 표)와 `results/curves.png` 를 만든다. 표에서 볼 행: `A`(적응 없음), `ES(B=16, theta=0.15)`(기본 ES), `ES(B=16, theta=0.15, pack4)`(4비트 ES), `V7(B=16, pack4)`(제안).
- 표의 열: 변화구간 mAP50, 업링크 KB, 다운링크 KB, 회복시간, 적응 횟수. 이 표를 그대로 복사해서 보내 주면 해석한다.

**5-4. 내 노트북에서 실제 영상 데모 (선택, Colab 이 끝난 뒤)**
1. 셀 32 가 구글 드라이브에 `laptop_bundle.zip` 을 만든다 → 드라이브에서 다운로드.
2. `C:\edge-adapt` 에 **그 폴더 안에서 압축 풀기**. 그러면 `C:\edge-adapt\models\` 에 `student.pt`, `student.onnx`, `teacher.pt` 가 생긴다(확인: `dir models`).
3. 서버 창과 엣지 창에서(단계 4 와 같은 방식, `models_toy` 대신 `models`):
```
python scripts/serve.py --models models --device cpu --codec pack4
python scripts/demo.py --source stream --models models --server http://127.0.0.1:8000 --fps 2
```
`--source stream` 은 번들에 들어 있는 평가 스트림이고, 내 영상 파일은 `--source C:\영상\내영상.mp4`, 웹캠은 `--source 0` 이다. 영상 파일·웹캠에는 정답 라벨이 없으므로 mAP 곡선 대신 "평균 신뢰도"가 표시된다.

## 막힐 때

| 증상 | 조치 |
|---|---|
| `python` 이 "인식되지 않습니다" | 터미널을 닫고 새로 연다. 그래도면 Python 재설치 시 **Add python.exe to PATH** 체크 |
| `Activate.ps1` 에서 빨간 글씨 | 단계 2 의 `Set-ExecutionPolicy …` 줄 실행 후 재시도 |
| `(.venv)` 가 안 보임 | `cd C:\edge-adapt` 후 `.venv\Scripts\Activate.ps1` |
| `pip install torch` 가 너무 오래/실패 | 같은 줄을 다시 실행. 계속 실패하면 `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu` |
| `git push` 에서 인증 실패 | 브라우저 로그인 창을 닫았다면 `git push -u origin main` 을 다시 실행 |
| `can't open file 'scripts/serve.py'` | `cd` 가 틀렸다. 단계 1 의 안쪽 폴더 여부 확인 |
| 데모 창이 안 뜸 | 서버 창에 `BOOT | … 대기 중` 이 떠 있는지 먼저 확인. 엣지 창 마지막 빨간 줄을 복사해 질문 |
| "서버 적응 중" 이 계속 | 서버 창에 `TRAIN` 이 계속 올라오면 정상(CPU 라 느림). 멈춰 있고 `ERR` 가 보이면 그 줄을 복사해 질문 |
| `Connection refused` / 접속 오류 | 서버 창이 꺼졌거나 포트 불일치. 서버 창에서 3번 줄을 다시 실행 |
| 데모/서버가 갑자기 꺼짐(창이 닫히거나 `Killed`, 아무 오류 없이 종료) | 메모리 부족일 가능성이 높다. 크롬 등 다른 프로그램을 모두 끄고 다시 하거나, 노트북 2대 방식을 쓴다 |
| 적응이 너무 느림(노트북 CPU 가 약함) | [서버 창]을 Ctrl+C 로 끄고 서버 줄 끝에 `--epochs 30` 을 붙여 다시 실행(학습 시간이 약 절반; 이 합성 데모에서 변화구간 평균 mAP 0.92 로 확인) |
| 서버 로그를 파일로 보고 싶다 | `C:\edge-adapt\logs\server.log` |

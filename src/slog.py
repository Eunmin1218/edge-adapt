"""서버 쪽 구조화 로그. 한 줄 = `시각 | 단계 | 내용`. 터미널(색)과 logs/server.log(색 없음)에 함께 쓴다.
단계 이름: BOOT 시작 설정 · RECV 엣지 업로드 수신 · TEACH Teacher 의사 라벨 · REPLAY 재생 버퍼 · TRAIN 증류 학습 · EXPORT ONNX 내보내기 ·
          DELTA 어댑터 변화량 · CODEC 부호화 · SEND 엣지로 전송 · PROBE 일치도 질의 · DONE 작업 요약 · WARN/ERR 문제"""
import os, sys, time, threading
from pathlib import Path

_C = {"BOOT": "36", "RECV": "34", "TEACH": "35", "REPLAY": "35", "TRAIN": "33", "EXPORT": "90", "DELTA": "32", "CODEC": "32", "SEND": "34", "PROBE": "36", "DONE": "1;32", "WARN": "1;33", "ERR": "1;31"}
_lock = threading.Lock(); _file = None; quiet = False
if os.name == "nt": os.system("")                                # Windows 10+ 콘솔에서 ANSI 색을 켠다


def setup(path="logs/server.log", quiet_=False):
    global _file, quiet
    quiet = quiet_; p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); _file = open(p, "a", encoding="utf-8")


def log(stage, msg, job=None):
    t = time.time(); ts = time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"; tag = f"[{job}] " if job else ""
    plain = f"{ts} | {stage:<6} | {tag}{msg}"
    with _lock:
        if _file: _file.write(plain + "\n"); _file.flush()
        if quiet and stage not in ("BOOT", "DONE", "WARN", "ERR"): return
        if sys.stdout.isatty() or os.environ.get("FORCE_COLOR"): print(f"\033[90m{ts}\033[0m | \033[{_C.get(stage, '0')}m{stage:<6}\033[0m | {tag}{msg}", flush=True)
        else: print(plain, flush=True)


def kb(n): return f"{n / 1024:.1f}KB"

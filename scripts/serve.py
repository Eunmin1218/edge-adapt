"""서버(클라우드 역할) 실행. 엣지(scripts/demo.py)가 보낸 프레임으로 Teacher 의사 라벨 → 어댑터 학습 → 부호화해 돌려준다.
기본은 이 컴퓨터에서만 접속(127.0.0.1). 노트북 2대로 나눠 쓸 때는 --host 0.0.0.0 (같은 Wi-Fi/핫스팟 안에서만 쓰고, 끝나면 끈다).
예)  python scripts/serve.py --device cpu --epochs 5 --port 8000
     python scripts/serve.py --models models_toy --device cpu --codec pack4          (합성 영상 빠른 시험용 모델)
로그: 터미널 + logs/server.log (단계: BOOT RECV TEACH REPLAY TRAIN EXPORT DELTA CODEC SEND PROBE DONE)"""
import argparse, sys, time
from pathlib import Path
import uvicorn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT
from src import slog

ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=None, help="기본: 모델 폴더의 adapter.json, 없으면 30")
ap.add_argument("--port", type=int, default=8000); ap.add_argument("--tau", type=float, default=None); ap.add_argument("--device", default="auto", help="auto | cpu | 0 (NVIDIA GPU 번호)")
ap.add_argument("--host", default="127.0.0.1", help="다른 노트북에서 접속하게 하려면 0.0.0.0"); ap.add_argument("--models", default="models", help="student.pt / student.onnx / teacher.pt 가 있는 폴더")
ap.add_argument("--codec", default="auto", choices=["auto", "pack", "pack16", "pack4"], help="어댑터 전송 부호화. pack4 = 채널별 4비트(가장 작음), pack16 = FP16"); ap.add_argument("--quiet", action="store_true", help="요약만 출력"); ap.add_argument("--log", default="logs/server.log")
a = ap.parse_args(); slog.setup(ROOT / a.log, a.quiet)
from src.server_core import AdapterServer, load_adapter_cfg
from src.adapter_cfg import server_kwargs
from src.api import create_app
M = ROOT / a.models; cfg = load_adapter_cfg(M / "adapter.json") if (M / "adapter.json").exists() else load_adapter_cfg(); a.epochs = a.epochs or cfg["epochs"]; a.tau = cfg["tau"] if a.tau is None else a.tau
dev = int(a.device) if a.device.isdigit() else a.device
for f in ("student.pt", "student.onnx", "teacher.pt"):
    if not (M / f).exists(): sys.exit(f"[오류] {M / f} 가 없습니다. 빠른 시작 가이드의 '모델 파일 준비'를 먼저 하세요.")
t0 = time.time(); srv = AdapterServer(M / "student.pt", M / "student.onnx", M / "teacher.pt", ROOT / "runs/serve_work", device=dev, **{**server_kwargs(cfg), "tau": a.tau, "epochs": a.epochs})
slog.log("BOOT", f"모델 폴더={a.models} · Student/Teacher 로드 {time.time() - t0:.1f}s · device={srv.device} · tau={a.tau} · epochs={a.epochs} · lr={cfg['lr']} · batch={cfg['batch']} · trainer={cfg['trainer']} · mode={cfg['mode']} α={cfg['alpha']} · 학습 레이어={cfg['train_layers']} · codec={a.codec}")
slog.log("BOOT", f"http://{a.host}:{a.port} 대기 중 · 엣지는 --server http://<이 컴퓨터 주소>:{a.port} 로 접속 · 로그 파일 {a.log}")
if a.host != "127.0.0.1": slog.log("WARN", "0.0.0.0 으로 열렸습니다. 인증이 없으니 신뢰하는 네트워크(같은 Wi-Fi/핫스팟)에서만 쓰고, 끝나면 서버를 끄세요.")
uvicorn.run(create_app(srv, srv.agreement, base=srv.base, codec=a.codec), host=a.host, port=a.port, log_level="warning")

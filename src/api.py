"""서버 API (FastAPI). 업링크: POST /v1/frames  /  다운링크: GET /v1/adapters/{name}
adapt_fn(frames_bgr: list[ndarray], env_id: int) -> dict{initializer이름: ndarray}  (서버의 Teacher+어댑터 학습 함수)"""
import io, time, threading, uuid
import numpy as np, cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response


def pack(adapter, base=None, codec="auto"):           # dict -> 바이트. base(= 엣지·서버가 공유하는 base ONNX 의 initializer)가 있으면 변화량 부호화('auto'), 없으면 FP16 npz
    if base is not None:
        from .delta_codec import encode; return encode(adapter, base, codec)
    b = io.BytesIO(); np.savez(b, **{k: np.asarray(v).astype(np.float16) for k, v in adapter.items()}); return b.getvalue()


def unpack(raw, base=None):
    from .delta_codec import _MAGIC, decode
    if raw[:1] == bytes([_MAGIC]) if isinstance(_MAGIC, int) else raw[:1] == _MAGIC: return decode(raw, base)   # pack/pack16/pack4 는 npz 가 아니다
    z = np.load(io.BytesIO(raw))
    if "_mode" in z.files: return decode(raw, base)
    return {k: z[k].astype(np.float32) for k in z.files}


def create_app(adapt_fn, agree_fn=None, base=None, codec="auto"):
    from . import slog
    app = FastAPI(title="edge-adapt server"); jobs, store = {}, {}; tot = {}      # tot[client] = [업로드 바이트, 다운로드 바이트, 적응 작업 수]
    lock = threading.Lock()                                  # GPU 학습은 한 번에 하나만 실행

    def run(job, frames, env):
        t0 = time.time()
        try:
            tw = time.time()
            with lock:
                if time.time() - tw > 0.05: slog.log("WARN", f"GPU/CPU 학습 락 대기 {time.time() - tw:.1f}s (다른 작업이 학습 중)", job)
                ad = adapt_fn(frames, env)
            tc = time.time(); name = f"env{str(env).replace(':', '_')}_{job}"; store[name] = pack(ad, base, codec)
            raw = sum(np.asarray(v).nbytes for k, v in ad.items() if k != "__tone__")
            slog.log("CODEC", f"codec={codec} FP32 {slog.kb(raw)} → {slog.kb(len(store[name]))} (÷{raw / max(len(store[name]), 1):.1f}) {time.time() - tc:.2f}s", job)
            jobs[job].update(state="done", adapter=name, bytes=len(store[name]), sec=round(time.time() - t0, 2))
            slog.log("DONE", f"작업 종료 {time.time() - t0:.1f}s · 엣지가 `GET /v1/adapters/{name}` 로 받아가길 대기", job)
        except Exception as e:                               # 서버 오류는 상태로 노출, 엣지는 기존 어댑터 유지
            jobs[job].update(state="error", error=repr(e)); slog.log("ERR", f"적응 실패: {e!r}", job)

    @app.get("/v1/health")
    def health(): return {"ok": True}

    @app.post("/v1/frames")
    async def upload(env: int = Form(0), client: str = Form(""), files: list[UploadFile] = File(...)):
        frames = []; raw_up = 0
        for f in files:
            blob = await f.read(); raw_up += len(blob); img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if img is None: raise HTTPException(400, f"디코딩 실패: {f.filename}")
            frames.append(img)
        job = uuid.uuid4().hex[:8]; jobs[job] = {"state": "running", "n": len(frames)}
        t = tot.setdefault(client or "-", [0, 0, 0]); t[0] += raw_up; t[2] += 1
        slog.log("RECV", f"client={client or '-'} env={env} 프레임 {len(frames)}장 {frames[0].shape[1]}x{frames[0].shape[0]} JPEG {slog.kb(raw_up)} · 누적 업 {slog.kb(t[0])} · 적응 요청 {t[2]}회", job)
        threading.Thread(target=run, args=(job, frames, f"{client}:{env}" if client else env), daemon=True).start()   # 학습은 백그라운드로, 업로드 응답은 즉시
        return {"job": job, "n": len(frames)}

    @app.post("/v1/probe")
    async def probe(dets: str = Form(...), files: list[UploadFile] = File(...)):
        """EdgeSync-lite 용: 엣지가 보낸 프레임과 엣지 탐지 결과를 서버 Teacher 와 비교해 일치도(0~1)를 돌려준다."""
        if agree_fn is None: raise HTTPException(501, "agree_fn 이 설정되지 않음")
        import json
        frames = [cv2.imdecode(np.frombuffer(await f.read(), np.uint8), cv2.IMREAD_COLOR) for f in files]
        d = [(np.array(x["boxes"], float).reshape(-1, 4), np.array(x["scores"], float), np.array(x["cls"], int)) for x in json.loads(dets)]
        with lock:
            ag = float(agree_fn(frames, d))
        slog.log("PROBE", f"일치도 질의 프레임 {len(frames)}장 → Teacher 박스 대비 엣지 일치도 {ag:.3f}"); return {"agreement": ag}

    @app.get("/v1/jobs/{job}")
    def job_state(job: str):
        if job not in jobs: raise HTTPException(404, "없는 job")
        return jobs[job]

    @app.get("/v1/adapters/{name}")
    def get_adapter(name: str):
        if name not in store: raise HTTPException(404, "없는 어댑터")
        tot.setdefault("_down", [0, 0, 0])[1] += len(store[name]); slog.log("SEND", f"{name} {slog.kb(len(store[name]))} → 엣지 · 누적 다운 {slog.kb(tot['_down'][1])}")
        return Response(store[name], media_type="application/octet-stream")
    return app

"""적응 방식 비교(같은 선별 프레임·같은 보류 프레임): 학습 방식(ultra 하드라벨 / distill 증류) x 학습 레이어 수 x 전송 부호화(fp16/d8/sparse).
출력: 변화구간 보류 프레임 mAP50 (적응 전 대비 이득), 실제 전송 바이트(부호화 후), 적응 1회 시간.  --variants 로 목록을 바꿀 수 있고 결과는 --out(JSON)에 한 줄씩 누적 저장된다(재실행 시 건너뜀).
변형 표기: trainer[,키=값...]  키: ep lr ly(학습 레이어 수) codec(auto|fp16|d8|sparse) keep bw(박스 KL 가중) hard(하드라벨 손실 가중) aug(0/1) batch mode(head|affine|affine+head) tone(0/1) lr_aff lr_tone uw(엔트로피 가중).   예) distill,ep=30,lr=0.01,mode=affine+head,tone=1"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT
from src.stream import load_stream
from src.onnx_detector import OnnxDetector
from src.selectors import STRATEGIES, frame_uncertainty, frame_embedding, frame_entropy
from src.mapeval import evaluate
from src.server_core import AdapterServer, onnx_inits
from src.delta_codec import encode, decode, scale_adapter

DEF = ["ultra", "distill,lr=0.01", "distill,lr=0.01,batch=4", "distill,lr=0.03,batch=4", "distill,lr=0.01,batch=4,aug=0", "distill,lr=0.01,batch=4,hard=1", "distill,lr=0.01,batch=4,hard=1,aug=0"]
ap = argparse.ArgumentParser(); ap.add_argument("--variants", nargs="*", default=DEF); ap.add_argument("--budget", type=int, default=8); ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--selector", default="diversity"); ap.add_argument("--world", default="toy", choices=["toy", "toy2"], help="toy2: 환경 5종 합성 세계(runs/toy2)"); ap.add_argument("--envs", default="night,fog,rain,night_rain"); ap.add_argument("--alphas", default="1.0", help="가중치 보간 계수 목록(쉼표). 1.0 이외는 학습 없이 보간만 바꿔 평가")
ap.add_argument("--stream-dir", default="runs/toy/data/stream_tune"); ap.add_argument("--streams-json", default="runs/toy/streams.json"); ap.add_argument("--split", default="detector_train")
ap.add_argument("--student-pt", default="runs/toy/student.pt"); ap.add_argument("--student-onnx", default="runs/toy/student.onnx"); ap.add_argument("--teacher-pt", default="runs/toy/teacher.pt")
ap.add_argument("--out", default="runs/toy/ablate.json"); ap.add_argument("--pool", type=int, default=100); ap.add_argument("--held", type=int, default=100); ap.add_argument("--device", default="cpu"); a = ap.parse_args()
if a.world == "toy2": a.student_pt, a.student_onnx, a.teacher_pt = "runs/toy2/student.pt", "runs/toy2/student.onnx", "runs/toy2/teacher.pt"; a.out = a.out if a.out != "runs/toy/ablate.json" else "runs/toy2/ablate.json"
if a.world == "toy2":
    from src.toyworld import stream as tw_stream
    E = a.envs.split(","); stream = tw_stream(7, [("clear", 50)] + [(e, 200) for e in E]); env = [it["env"] for it in stream]
else:
    segs = json.load(open(ROOT / a.streams_json))[a.split]; stream = load_stream(ROOT / a.stream_dir, segs, multiple=50); env = [it["env"] for it in stream]
runs, s = [], 0
for i in range(1, len(env) + 1):
    if i == len(env) or env[i] != env[s]: runs.append((env[s], s, i)); s = i
shift = [(e, s, t) for e, s, t in runs if e != "clear"]; base = OnnxDetector(ROOT / a.student_onnx); sel = STRATEGIES[a.selector]; data = []
for sd in range(a.seeds):
    for e, s0, t0 in shift:
        pool_n = min(a.pool, (t0 - s0) // 2); pool = [stream[i]["img"] for i in range(s0, s0 + pool_n)]; held = list(range(s0 + pool_n, min(t0, s0 + pool_n + a.held)))
        dets = [base.infer(im) for im in pool]; idx = sel({"uncert": np.array([frame_uncertainty(d) for d in dets]), "emb": np.stack([frame_embedding(im) for im in pool]), "ent": np.array([frame_entropy(d) for d in dets]), "age": np.arange(len(pool))[::-1].astype(float)}, a.budget, np.random.default_rng(sd))
        imgs = [stream[i]["img"] for i in held]; gts = [stream[i]["gt"] for i in held]; pb = [(d["boxes"], d["scores"], d["cls"]) for d in map(base.infer, imgs)]
        data.append({"env": e, "frames": [pool[i] for i in idx], "imgs": imgs, "gts": gts, "base": evaluate(pb, gts, 6, iou_thrs=[0.5])["map50"]})
print("적응 전 보류 mAP50:", {e: round(float(np.mean([d["base"] for d in data if d["env"] == e])), 3) for e in dict.fromkeys(d["env"] for d in data)}, flush=True)
B = onnx_inits(ROOT / a.student_onnx); out = Path(ROOT / a.out); done = json.load(open(out)) if out.exists() else {}
for v in a.variants:
    if v in done: print(v, "(캐시)", done[v]); continue
    kv = dict(x.split("=") for x in v.split(",")[1:]); tr = v.split(",")[0]; cd = kv.get("codec", "auto"); keep = float(kv.get("keep", 0.1)); alphas = [float(x) for x in a.alphas.split(",")]
    srv = AdapterServer(ROOT / a.student_pt, ROOT / a.student_onnx, ROOT / a.teacher_pt, ROOT / "runs/ablate_work" / (a.world + "_" + v.replace(",", "_").replace("=", "")), epochs=int(kv.get("ep", 30)), lr=float(kv.get("lr", 3e-3)), device=a.device, train_layers=int(kv.get("ly", 0)),
                        trainer=tr, box_w=float(kv.get("bw", 1.0)), hard_w=float(kv.get("hard", 0.0)), aug=int(kv.get("aug", 1)), batch=int(kv.get("batch", 8)), mode=kv.get("mode", "head"), tone=bool(int(kv.get("tone", 0))),
                        lr_aff=float(kv.get("lr_aff", 5e-3)), lr_tone=float(kv.get("lr_tone", 1e-2)), uw=float(kv.get("uw", 0.0)))
    res = {al: [] for al in alphas}; kb, secs, envs = [], [], []
    for k, d in enumerate(data):
        t0 = time.time(); ad = srv(d["frames"], k); secs.append(time.time() - t0); envs.append(d["env"])
        for al in alphas:
            ada = ad if al == 1.0 else scale_adapter(ad, B, al); raw = encode(ada, B, cd, keep=keep)
            if al == alphas[0]: kb.append(len(raw) / 1024)
            det = OnnxDetector(ROOT / a.student_onnx); det.apply_adapter(decode(raw, B)); pr = [(o["boxes"], o["scores"], o["cls"]) for o in map(det.infer, d["imgs"])]
            res[al].append(evaluate(pr, d["gts"], 6, iou_thrs=[0.5])["map50"])
    base_ = np.array([d["base"] for d in data]); per_env = {}
    for al in alphas:
        ap_ = np.array(res[al]); per_env[str(al)] = {e: round(float(ap_[[x == e for x in envs]].mean()), 3) for e in dict.fromkeys(envs)}
    done[v] = {"mAP50": {str(al): float(np.mean(res[al])) for al in alphas}, "gain": {str(al): float(np.mean(np.array(res[al]) - base_)) for al in alphas}, "per_env": per_env, "KB": float(np.mean(kb)), "sec": float(np.mean(secs)), "n_tensors": len(ad)}
    json.dump(done, open(out, "w"), indent=1)
    for al in alphas: print(f"{v:44s} a={al:<4g} mAP50 {np.mean(res[al]):.3f} (이득 {np.mean(np.array(res[al]) - base_):+.3f}) 환경별 {per_env[str(al)]} | 전송 {done[v]['KB']:.0f}KB | {done[v]['sec']:.0f}s", flush=True)

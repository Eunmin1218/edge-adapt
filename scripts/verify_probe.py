"""서버 어댑터 수락 규칙의 신뢰도 측정 (스트림 시뮬레이션 없음).
환경(night, glare, heavy_rain, clear_sparse, clear_dense) × 여러 번 추출마다 (1) 엣지 통계 정렬 어댑터(현재) (2) 서버 증류 어댑터(후보)를 만들고
정답(GT) 기반 진짜 이득 = mAP50(후보) - mAP50(현재) 를 40프레임으로 재 둔다. 그리고 학습에 쓰지 않은 보류 프레임 K장으로 정답 없는 신호들이 이 부호를 맞히는지 본다:
  conf  : 평균 신뢰도 차이 (현재 V7 의 수락 규칙: 후보 >= 현재 - lf_margin)
  fc    : 뒤집기 일관성 차이
  agree : Teacher 일치도 차이 (서버가 업링크 추가 없이 계산 가능; EdgeSync-lite 프로브와 같은 지표)
  agree_tr: 학습 프레임에서의 Teacher 일치도 차이 (과적합으로 부풀려지는지 확인용)
합성 데이터 측정이다. 사용: python scripts/verify_probe.py [draws=3] [K=6]"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT
from src.toyworld import make_img
from src.server_core import AdapterServer, onnx_inits, load_adapter_cfg
from src.adapter_cfg import server_kwargs
from src.onnx_detector import OnnxDetector
from src.stat_align import StatAligner
from src.mapeval import evaluate, iou_matrix

D = int(sys.argv[1]) if len(sys.argv) > 1 else 3; K = int(sys.argv[2]) if len(sys.argv) > 2 else 6
W = ROOT / "runs/toy3"; S_PT, T_PT = W / "student.pt", W / "teacher.pt"; OUT = ROOT / "results" / "dev"; OUT.mkdir(parents=True, exist_ok=True)
CFG = load_adapter_cfg(W / "adapter.json"); base = onnx_inits(W / "student.onnx")
rg = np.random.default_rng(5); sa = StatAligner(S_PT, [make_img(rg, "clear")[0] for _ in range(60)])
srv = AdapterServer(S_PT, W / "student.onnx", T_PT, W / "server" / "vprobe", device="cpu", imgsz=640, **server_kwargs(CFG))
ENVS = ["night", "glare", "heavy_rain", "clear_sparse", "clear_dense"]


def det_with(ad):
    d = OnnxDetector(W / "student.onnx")
    if ad: d.apply_adapter(ad)
    return d


def mAP(det, test):
    P = []
    for img, gt in test:
        o = det.infer(img); P.append((o["boxes"], o["scores"], o["cls"]))
    return evaluate(P, [g for _, g in test], 6)["map50"]


def conf(det, fr): return float(np.mean([np.mean(o["scores"]) if len(o["scores"]) else 0.0 for o in map(det.infer, fr)]))


def fcons(det, fr):
    out = []
    for f in fr:
        d1 = det.infer(f); d2 = det.infer(f[:, ::-1].copy()); b2 = d2["boxes"].copy()
        if len(b2): b2[:, [0, 2]] = f.shape[1] - b2[:, [2, 0]]
        if len(d1["boxes"]) == 0 and len(b2) == 0: out.append(1.0)
        elif len(d1["boxes"]) == 0 or len(b2) == 0: out.append(0.0)
        else: M = iou_matrix(d1["boxes"], b2); out.append(float(0.5 * (M.max(1).mean() + M.max(0).mean())))
    return float(np.mean(out))


def agree(det, fr, f1=False): return (srv.agreement_f1 if f1 else srv.agreement)(fr, [(o["boxes"], o["scores"], o["cls"]) for o in map(det.infer, fr)])


rows = []
for ei, env in enumerate(ENVS):
    for d in range(D):
        rt = np.random.default_rng(1000 + 17 * ei + d); train = [make_img(rt, env)[0] for _ in range(8)]; held = [make_img(rt, env)[0] for _ in range(K)]; test = [make_img(rt, env) for _ in range(40)]
        t0 = time.time(); ad_st, _ = sa.adapter(train, base); ad_srv = srv(train, 100 * ei + d)
        cur, cand = det_with(ad_st), det_with(ad_srv)
        r = dict(env=env, draw=d, m_cur=mAP(cur, test), m_cand=mAP(cand, test))
        r["gain"] = r["m_cand"] - r["m_cur"]
        r.update(conf=conf(cand, held) - conf(cur, held), fc=fcons(cand, held) - fcons(cur, held), agree=agree(cand, held) - agree(cur, held),
                 agree2=agree(cand, held[:2]) - agree(cur, held[:2]), agree_tr=agree(cand, train) - agree(cur, train),
                 f1=agree(cand, held, True) - agree(cur, held, True), f1_2=agree(cand, held[:2], True) - agree(cur, held[:2], True), f1_4=agree(cand, held[:4], True) - agree(cur, held[:4], True),
                 f1_tr=agree(cand, train, True) - agree(cur, train, True))
        rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, f"{time.time() - t0:.0f}s", flush=True)
        json.dump(rows, open(OUT / f"verify_probe_K{K}.json", "w"))

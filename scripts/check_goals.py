"""계획서 6.1.2 의 수치 목표를 results/*.json 으로 판정한다. 사용: python scripts/check_goals.py
 목표2: 같은 예산에서 Random 선별 대비 변화구간 mAP 회복량과 회복 시간이 모두 더 좋은 전략이 있는 예산 단계 수 >= 2 (3단계 중)
 목표3: D(diversity, B=16) 변화구간 mAP - A >= 0.05(5%p), 어댑터 평균 크기(FP16) <= 5MB, B·C 대비 mAP/전송량 보고
 (감지기 목표 Recall/오경보율은 scripts/train_detector.py 출력으로 판정)"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT
from src.analysis import recovery_baselines, recovery_time

R = [json.load(open(f)) for f in sorted(glob.glob(str(ROOT / "results/*.json")))]; R = [r for r in R if r["args"]["split"] == "eval"]
T = next((r for r in R if r["args"]["method"] == "T"), None)
if not T: sys.exit("T(Teacher 단독) 결과가 필요합니다: python scripts/run_experiment.py --method T --tag T")
win = R[0]["args"]["win"]; base = recovery_baselines(T)
g = defaultdict(list)
for r in R:
    a = r["args"]
    if a["method"] == "D" and a["cols"] == "all" and not a["no_registry"] and not a.get("tau_explicit", False) and a.get("consec", 1) == 1: g[("D", a["selector"], a["budget"])].append(r)
    elif a["method"] == "ES": g[("ES", a.get("theta", 0.15), a["budget"])].append(r)
    elif a["method"] in ("A", "C", "B", "AMS", "RECL"): g[(a["method"], a.get("tent") if a["method"] == "C" else "", a["budget"] if a["method"] in ("B", "AMS", "RECL") else 0)].append(r)


def stat(rs):
    rt = [recovery_time(x, base, win)[0] for x in rs]; rt = [v for v in rt if v is not None]
    return float(np.mean([x["map_shift_windows"] for x in rs])), (float(np.mean(rt)) if rt else float("inf")), np.mean([x["up_bytes"] for x in rs]) / 1024


budgets = sorted({k[2] for k in g if k[0] == "D"}); wins = 0
print("== 목표2: 선별 전략 vs Random ==")
for b in budgets:
    rnd = g.get(("D", "random", b))
    if not rnd: print(f" B={b}: D(random) 결과 없음"); continue
    m0, t0, _ = stat(rnd); better = []
    for sel in ("uncertainty", "diversity", "hybrid", "entropy", "ent_hybrid"):
        rs = g.get(("D", sel, b))
        if rs:
            m, t, _ = stat(rs)
            if m > m0 and t < t0: better.append(sel)
            print(f" B={b:3d} {sel:11s} mAP {m:.3f} (random {m0:.3f}) / 회복 {t:.0f} (random {t0:.0f}) 프레임")
    wins += bool(better); print(f"   → B={b}: Random 보다 두 지표 모두 우수한 전략: {better or '없음'}")
print(f" 목표2 {'달성' if wins >= 2 else '미달'}: {wins}/{len(budgets)} 단계")
print("== 목표3 ==")
A = g.get(("A", "", 0)); Dh = g.get(("D", "diversity", 16))
if A and Dh:
    d = stat(Dh)[0] - stat(A)[0]; print(f" 변화구간 mAP50: D(diversity, B=16) - A = {d*100:+.1f}%p → {'달성' if d >= 0.05 else '미달'} (목표 +5%p)")
    sz = [x["down_bytes"] / max(x["n_trigger"], 1) / 2**20 for x in Dh]; print(f" 어댑터 평균 크기(FP16) {np.mean(sz):.2f} MB → {'달성' if np.mean(sz) <= 5 else '미달'} (목표 5MB 이하)")
else: print(" A 또는 D(diversity, B=16) 결과 없음")
print("== 발표용: 제안 vs EdgeSync-lite (EdgeSync 는 theta 를 바꿔 본 것 중 변화구간 mAP 가 가장 높은 설정과 비교 = 가장 유리한 조건) ==")
for b in budgets:
    D_ = g.get(("D", "diversity", b)); ES_ = {k[1]: v for k, v in g.items() if k[0] == "ES" and k[2] == b}
    if not D_ or not ES_: continue
    th = max(ES_, key=lambda x: stat(ES_[x])[0]); md, td, ud = stat(D_); me, te, ue = stat(ES_[th])
    print(f" B={b:3d}  제안 mAP {md:.3f} 회복 {td:.0f}f 업링크 {ud:.0f}KB  |  EdgeSync(theta={th}) mAP {me:.3f} 회복 {te:.0f}f 업링크 {ue:.0f}KB  → mAP {(md - me) * 100:+.1f}%p, 업링크 {(ud / max(ue, 1e-9) - 1) * 100:+.0f}%  {'제안 우세' if md > me and td <= te else '제안 우세 아님(원인 점검: GUIDE 6.7)'}")
for k in sorted(g, key=str):
    if k[0] in ("B", "C", "AMS", "RECL") and Dh:
        m, t, up = stat(g[k]); md, td, upd = stat(Dh); print(f" {k[0]}{'('+str(k[1])+')' if k[1] else ''}{' B='+str(k[2]) if k[2] else ''}: mAP {m:.3f} 회복 {t:.0f} 업링크 {up:.0f}KB  |  D: mAP {md:.3f} 회복 {td:.0f} 업링크 {upd:.0f}KB")

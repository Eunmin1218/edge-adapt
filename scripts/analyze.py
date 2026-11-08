"""results/*.json 을 모아 표와 그래프를 만든다. 사용: python scripts/analyze.py
회복 시간: 환경 구간(연속된 같은 env)마다 '윈도우 mAP50 >= 회복 기준선'에 처음 도달하는 데까지의 프레임 수. 기준선 = 해당 환경에서 Teacher 단독(T) mAP의 90%.
도달하지 못하면 구간 길이로 계산(검열값)하고 '미회복 횟수'를 따로 센다."""
import json, sys, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT

R = [json.load(open(f)) for f in sorted(glob.glob(str(ROOT / "results/*.json")))]
R = [r for r in R if r["args"]["split"] == "eval"]
if not R: sys.exit("results/ 에 eval 결과가 없습니다.")
T = next((r for r in R if r["args"]["method"] == "T"), None)
win = R[0]["args"]["win"]


from src.analysis import recovery_baselines, recovery_time

base = recovery_baselines(T) if T else {}


def recovery(r):
    return recovery_time(r, base, win)


def key(r):
    a = r["args"]; m = a["method"]
    if m == "D": return f"D({a['selector']}, B={a['budget']}, 신호={a['cols']}{', 레지스트리X' if a['no_registry'] else ''}{', consec=' + str(a['consec']) if a.get('consec', 1) != 1 else ''}{', tau=' + str(a['tau']) if a.get('tau_explicit', False) else ''})"
    if m == "ES": return f"ES(B={a['budget']}, theta={a.get('theta', 0.15)})"
    if m in ("B", "AMS", "RECL"): return f"{m}(period={a['period']}, B={a['budget']})"
    if m == "C": return f"C({a['tent']})"
    return m


g = defaultdict(list)
for r in R: g[key(r)].append(r)
rows = []
for k, rs in sorted(g.items()):
    f = lambda name: [x[name] for x in rs if x[name] is not None]
    rec = [recovery(x) for x in rs]; rv = [x[0] for x in rec if x[0] is not None]
    rows.append((k, len(rs), np.mean(f("map_shift_windows")), np.mean([x["map_all"]["map50"] for x in rs]), (np.mean(f("map_fog_windows")) if f("map_fog_windows") else float("nan")),
                 np.mean(rv) if rv else float("nan"), np.mean([x[1] for x in rec]),
                 np.mean([x["up_bytes"] for x in rs]) / 1024, np.mean([x["down_bytes"] for x in rs]) / 1024, np.mean([x["n_trigger"] for x in rs]), np.mean([x["lat_ms"] for x in rs])))
hdr = "| 방법 | 시드수 | 변화구간 mAP50 | 전체 mAP50 | 안개 mAP50 | 회복시간(프레임) | 미회복 | 업링크 KB | 다운링크 KB | 적응횟수 | 추론ms/프레임 |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
md = hdr + "\n".join(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.0f} | {r[6]:.1f} | {r[7]:.0f} | {r[8]:.0f} | {r[9]:.1f} | {r[10]:.1f} |" for r in rows)
print(md); (ROOT / "results/summary.md").write_text(md)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(10, 4))
for k, rs in g.items():
    if k in ("A", "T") or k.startswith("C(") or k.startswith("B(period") and "B=16" in k or k.startswith("D(diversity, B=16, 신호=all)") or k.startswith("ES(B=16, theta=0.15)"):
        plt.plot(np.nanmean([x["win_map"] for x in rs], 0), label=k)
plt.xlabel(f"window index ({win} frames each)"); plt.ylabel("mAP50"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(ROOT / "results/curves.png", dpi=150)
print("저장: results/summary.md, results/curves.png")

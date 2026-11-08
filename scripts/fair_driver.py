"""공평 비교 드라이버 (toy3). 1) 개발 스트림에서 ES 를 튜닝 2) 임계값을 동결하고 새 평가 스트림(무작위 환경 순서·시드)에서 ES(기본/튜닝/+레지스트리, 반복)와 VF 를 비교.
모든 방법은 같은 전송 압축(JPEG q40, pack16)과 같은 서버 학습 설정을 쓴다. 결과는 results/dev/*.log, 요약은 fair_summary.json."""
import subprocess, re, json, sys, itertools, os
from pathlib import Path
PY = sys.executable; ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "dev"; OUT.mkdir(parents=True, exist_ok=True)


def run(tag, extra):
    log = OUT / f"{tag}.log"
    if not log.exists() or "통합 시험 완료" not in log.read_text(errors="ignore"):
        subprocess.run([PY, str(ROOT / "scripts/toy_e2e.py"), "--world", "toy3", "--evorder"] + extra, stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=ROOT)
    txt = log.read_text(errors="ignore").replace("\r", "\n"); rows = {}
    for ln in txt.split("\n"):
        m = re.match(r"^(\S.*?)\s{2,}(\d\.\d+)\s+(\d\.\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\[", ln)
        if m: rows[m.group(1).strip()] = dict(change=float(m.group(2)), total=float(m.group(3)), n=int(m.group(4)), up=int(m.group(5)), down=int(m.group(6)))
    for ln in txt.split("\n"):
        m = re.match(r"^(\S.*?)\s{2,}\[.*\] edge_extra=([\d.]+)", ln)
        if m and m.group(1).strip() in rows: rows[m.group(1).strip()]["edge_extra"] = float(m.group(2))
    return rows


RS = tuple(int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0").split(","))


def main():
    # 1) 개발 스트림(시드 201)에서 ES 튜닝
    grid = [(th, pn, 100, 0) for th in (0.05, 0.10, 0.15, 0.25) for pn in (4, 8)] + [(0.10, 4, 50, 0), (0.10, 4, 100, 1)]
    dev = run("dev201", ["--evseed", "201", "--methods", "ESf", "--es", ";".join(",".join(map(str, g)) for g in grid)])
    best = max((k for k in dev if k.startswith("ESf") and "reg=0" in k), key=lambda k: dev[k]["change"]); print("개발 스트림 ES 결과:", {k: (v["change"], v["up"]) for k, v in dev.items()}, "\n선택:", best, flush=True)
    th, pn, cd = [float(x) if i == 0 else int(float(x)) for i, x in enumerate(re.findall(r"=([\d.]+)", best)[:3])]
    summ = {"dev": dev, "best": best, "final": {}}
    for sd in tuple(int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "109,110,111,112").split(",")):
        for rs in RS:
            es = ";".join([f"0.15,4,100,0", f"{th},{pn},{cd},0"])
            rows = run(f"v7final{sd}_r{rs}", ["--evseed", str(sd), "--rseed", str(rs), "--methods", "ESf,V7d", "--es", es, "--v7", "--edge-delay", "13"])
            summ["final"][f"{sd}_r{rs}"] = rows; json.dump(summ, open(OUT / "fair_summary.json", "w"), ensure_ascii=False, indent=1)
            print(sd, rs, {k: (v["change"], v["up"], v["down"]) for k, v in rows.items() if k[:2] in ("ES", "VF")}, flush=True)
    (OUT / "done").write_text("ok")


main()

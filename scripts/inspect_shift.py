"""내려받은 SHIFT 파일 구조 확인. 사용: python scripts/inspect_shift.py <det_2d.json> <img.zip 또는 이미지 폴더>"""
import json, sys, zipfile, collections
from pathlib import Path

det, zp = sys.argv[1], sys.argv[2]
d = json.load(open(det)); fr = d["frames"] if isinstance(d, dict) else d
print("프레임 수:", len(fr), "| 시퀀스 수:", len({f["videoName"] for f in fr}))
f0 = fr[0]; print("첫 프레임 키:", list(f0.keys())); print("name:", f0["name"], "| videoName:", f0["videoName"], "| frameIndex:", f0["frameIndex"])
print("첫 라벨:", (f0.get("labels") or [None])[0])
print("카테고리 분포:", collections.Counter(l["category"] for f in fr[:2000] for l in (f.get("labels") or [])).most_common())
if zp.endswith(".zip"):
    n = zipfile.ZipFile(zp).namelist(); print("zip 항목 수:", len(n), "| 예:", n[:3])
else:
    n = [str(p.relative_to(zp)) for p in list(Path(zp).rglob("*.jpg"))[:3]]; print("폴더 예:", n)

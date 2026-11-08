"""seq.csv 를 읽어 환경별 시퀀스 목록(data/lists/*.txt)과 스트림 정의(configs/streams.json)를 만든다.
  1) python scripts/select_seqs.py --seq-csv <train의 seq.csv> --inspect     ← 열 이름과 값 종류를 출력(먼저 실행)
  2) python scripts/select_seqs.py --train-csv <train seq.csv> --val-csv <val seq.csv> --seed 0
SHIFT 의 열 이름이 아래 후보와 다르면 --inspect 출력을 보고 상단 WEATHER_KEYS/TIME_KEYS 만 고친다.
스트림은 모델 학습·검증에 쓴 시퀀스를 제외하고 뽑는다(누수 방지)."""
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT

WEATHER_KEYS = ["weather_coarse", "weather"]; TIME_KEYS = ["timeofday_coarse", "timeofday", "time_of_day"]; ID_KEYS = ["video", "videoName", "seq", "name", "id"]
WEATHER_MAP = {"clear": ["clear"], "rain": ["rain"], "fog": ["fog"]}          # 값에 이 부분문자열이 들어 있으면 해당 환경
TIME_MAP = {"day": ["day"], "night": ["night"]}


def pick(df, keys):
    for k in keys:
        if k in df.columns: return k
    raise SystemExit(f"열을 찾지 못함: {keys}. --inspect 로 열 이름을 확인하고 이 스크립트 상단의 *_KEYS 를 수정하세요. 현재 열: {list(df.columns)}")


def tag(df):
    w, t, i = pick(df, WEATHER_KEYS), pick(df, TIME_KEYS), pick(df, ID_KEYS)

    def m(v, mp):
        v = str(v).lower()
        for k, subs in mp.items():
            if any(s in v for s in subs): return k
        return "other"
    return pd.DataFrame({"seq": df[i].astype(str), "w": df[w].map(lambda v: m(v, WEATHER_MAP)), "t": df[t].map(lambda v: m(v, TIME_MAP))})


def env_of(r):
    if r.t == "night": return "night"
    if r.t == "day" and r.w in ("clear", "rain", "fog"): return {"clear": "clear_day", "rain": "rain_day", "fog": "fog_day"}[r.w]
    return "other"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seq-csv"); ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--train-csv"); ap.add_argument("--val-csv"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-student", type=int, default=300); ap.add_argument("--n-teacher", type=int, default=750)
    ap.add_argument("--seg-frames", type=int, default=500)
    a = ap.parse_args()
    if a.inspect:
        df = pd.read_csv(a.seq_csv); print(df.head(3).to_string()); print("열:", list(df.columns))
        for c in df.columns:
            if df[c].nunique() <= 25: print(f"\n[{c}]\n{df[c].value_counts().to_string()}")
        sys.exit(0)
    lists = ROOT / "data/lists"; lists.mkdir(parents=True, exist_ok=True); (ROOT / "configs").mkdir(exist_ok=True)
    tr = tag(pd.read_csv(a.train_csv)); tr["env"] = tr.apply(env_of, axis=1); va = tag(pd.read_csv(a.val_csv)); va["env"] = va.apply(env_of, axis=1)
    print("train 환경별 시퀀스 수:\n", tr.env.value_counts().to_string(), "\nval:\n", va.env.value_counts().to_string())
    st = tr[tr.env == "clear_day"].sample(min(a.n_student, (tr.env == "clear_day").sum()), random_state=a.seed)
    te = tr[tr.env.isin(["clear_day", "rain_day", "night"])]                          # 안개 제외
    te = te.groupby("env", group_keys=False).apply(lambda g: g.sample(min(len(g), a.n_teacher // 3), random_state=a.seed))
    (lists / "student_train.txt").write_text("\n".join(st.seq)); (lists / "teacher_train.txt").write_text("\n".join(te.seq))
    (lists / "student_val.txt").write_text("\n".join(va[va.env == "clear_day"].seq.head(40)))
    (lists / "teacher_val.txt").write_text("\n".join(va[va.env.isin(["clear_day", "rain_day", "night"])].groupby("env").head(15).seq))
    used_tr = set(st.seq) | set(te.seq); used_va = set((lists / "student_val.txt").read_text().split()) | set((lists / "teacher_val.txt").read_text().split())
    tr_pool, va_pool = tr[~tr.seq.isin(used_tr)], va[~va.seq.isin(used_va)]

    def seg(df, env, k):
        c = df[df.env == env]
        if len(c) < k: raise SystemExit(f"스트림용 '{env}' 시퀀스가 {len(c)}개뿐(필요 {k}). --n-student/--n-teacher 를 줄이세요.")
        return c.sample(k, random_state=a.seed).seq.tolist()

    def build(df, with_fog):                                   # 맑음 → 비 → 맑음 → 밤 → 맑음 (+ 평가 스트림은 안개 → 맑음)
        c, r, n = seg(df, "clear_day", 4 if with_fog else 3), seg(df, "rain_day", 1), seg(df, "night", 1)
        order = [(c[0], "clear"), (r[0], "rain"), (c[1], "clear"), (n[0], "night"), (c[2], "clear")]
        if with_fog: order += [(seg(df, "fog_day", 1)[0], "fog"), (c[3], "clear")]
        return [[s_, e, a.seg_frames] for s_, e in order]
    js = {"detector_train": build(tr_pool, False), "eval": build(va_pool, True)}
    json.dump(js, open(ROOT / "configs/streams.json", "w"), indent=1, ensure_ascii=False)
    (lists / "stream_train.txt").write_text("\n".join(x[0] for x in js["detector_train"])); (lists / "stream_eval.txt").write_text("\n".join(x[0] for x in js["eval"]))
    print("저장: data/lists/*.txt, configs/streams.json")

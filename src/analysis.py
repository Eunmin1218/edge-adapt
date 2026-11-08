"""결과 분석 공용 함수 (analyze.py, check_goals.py 가 공유)."""
import numpy as np


def segments(env):
    """연속된 같은 환경 태그 구간 [(시작, 끝, 태그)] (윈도우 index 기준)."""
    out, s = [], 0
    for i in range(1, len(env) + 1):
        if i == len(env) or env[i] != env[s]: out.append((s, i, env[s])); s = i
    return out


def recovery_baselines(T, ratio=0.9):
    """Teacher 단독(T) 결과에서 환경별 회복 기준선 = ratio × 해당 환경 평균 윈도우 mAP50."""
    base = {}
    for s, e, tag in segments(T["win_env"]):
        if tag != "clear": base.setdefault(tag, []).append(np.nanmean(T["win_map"][s:e]))
    return {k: ratio * float(np.mean(v)) for k, v in base.items()}


def recovery_time(r, base, win):
    """환경 구간마다 '윈도우 mAP50 >= 기준선' 에 처음 도달하는 데까지의 프레임 수. 미도달이면 구간 길이(검열값)로 세고 fail 로 센다.
    반환: (구간 평균 회복 프레임 수 또는 None, 미회복 구간 수)"""
    rec, fail = [], 0
    for s, e, tag in segments(r["win_env"]):
        if tag == "clear" or tag not in base: continue
        wm = np.array(r["win_map"][s:e], float); hit = np.where(wm >= base[tag])[0]
        if len(hit): rec.append((hit[0] + 1) * win)
        else: rec.append((e - s) * win); fail += 1
    return (float(np.mean(rec)) if rec else None), fail

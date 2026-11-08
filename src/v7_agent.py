"""제안 방법(V7q)의 엣지 에이전트를 조립한다. scripts/toy_e2e.py 의 `V7q` 실험과 같은 설정이며, 실시간 데모(scripts/demo.py)와 시험이 함께 쓴다.
구성: (1) T0 휘도 CUSUM 으로 켜지는 입력 정합 (2) 엣지 통계 정렬 + 환경 서명 레지스트리 + 10프레임 빠른 점검 (3) 라벨 없는 검증(평균 신뢰도)
      (4) 뒤집기 일관성이 소스 기준보다 0.04 넘게 낮을 때만 서버 정련 요청 (5) 서버 어댑터는 학습에 쓰지 않은 프레임으로 검증 후 적용."""
import numpy as np
from .sim import StreamRunner, Policy
from .server_core import onnx_inits
from .stat_align import StatAligner


def t0_params(frames):
    lum = np.array([f.mean() for f in frames]); return dict(lum_mu=float(lum.mean()), lum_sd=float(lum.std() + 1e-6), mu=np.mean([f.reshape(-1, 3).mean(0) for f in frames], 0).astype(np.float32), sd=np.mean([f.reshape(-1, 3).std(0) for f in frames], 0).astype(np.float32), k=2.0, h=6.0)


def make_v7_runner(det, student_pt, source_frames, server=None, budget=8, win=50, edge_delay=13, codec="pack4", apply_delay=30, seed=0, **extra):
    """source_frames: 맑은 낮 프레임 약 60장(소스 통계). server: 비동기 핸들을 돌려주는 함수(frames, env) 또는 None(서버 없음)."""
    sa = StatAligner(student_pt, list(source_frames)); base = det._base if getattr(det, "_base", None) is not None else None
    B = base if base is not None else onnx_inits(det.path)
    kw = dict(selector="diversity", budget=budget, win=win, apply_delay=apply_delay, use_registry=True, probe_fn=None, seed=seed,
              edge_fn=lambda fr: sa.adapter(fr, B, 1.0)[0], edge_sig=lambda fr: sa.signature(fr), env_thr=0.25, sig_thr=0.25, fast_margin=0.1, fast_n=10, verify_edge=True,
              agree_thr=0.85, confirm=False, min_post=30, gate_margin=9.0,
              t0=t0_params(list(source_frames)), lf_verify=True, fc_margin=0.04, probe_n=6, probe_q=40, codec=codec, heldout=True, edge_delay=edge_delay)
    kw.update(extra)
    return StreamRunner(det, Policy("none"), server, **kw)

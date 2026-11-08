"""전송 예산 B 안에서 서버로 보낼 프레임을 고르는 4가지 전략. 입력은 후보 프레임의 메타데이터 배열.
cands: dict(uncert[N], emb[N,D]). 반환: 선택된 인덱스 (길이 <= B)."""
import numpy as np
import cv2


def random_sel(cands, B, rng):
    n = len(cands["uncert"]); return rng.choice(n, size=min(B, n), replace=False)


def uncertainty_sel(cands, B, rng=None):
    return np.argsort(-cands["uncert"])[:B]                 # 불확실성이 큰 순


def diversity_sel(cands, B, rng):
    """k-center greedy: 이미 고른 프레임들과 가장 먼 프레임을 하나씩 추가."""
    emb = cands["emb"]; n = len(emb)
    first = int(rng.integers(n)); chosen = [first]
    d = np.linalg.norm(emb - emb[first], axis=1)
    while len(chosen) < min(B, n):
        j = int(np.argmax(d)); chosen.append(j)
        d = np.minimum(d, np.linalg.norm(emb - emb[j], axis=1))
    return np.array(chosen)


def hybrid_sel(cands, B, rng):
    """불확실성 상위 2B개로 후보를 줄인 뒤 그 안에서 다양성 선택."""
    top = np.argsort(-cands["uncert"])[:min(2 * B, len(cands["uncert"]))]
    sub = {"uncert": cands["uncert"][top], "emb": cands["emb"][top]}
    return top[diversity_sel(sub, B, rng)]


def edgesync_sel(cands, B, rng=None, alpha=0.5, beta=0.5, tw=50.0):
    """EdgeSync-lite 선별 (arXiv 2510.21781 Eq.1-3 를 옮김): Q = alpha*엔트로피 + beta*최신성, 상위 B개.
    엔트로피는 ln2 로 나눠 [0,1] 로 정규화, 최신성은 2*sigmoid(-age/tw) (가장 최근 프레임 = 1, 오래될수록 감소).
    논문 식 (2)는 부호가 '오래된 샘플 우선'으로 읽히는 모호함이 있어, 논문 본문의 취지(최신 샘플 우선)대로 구현했다."""
    ent = cands["ent"] / np.log(2.0); tim = 2.0 / (1.0 + np.exp(cands["age"] / tw))
    return np.argsort(-(alpha * ent + beta * tim))[:B]


def entropy_sel(cands, B, rng=None):
    """프레임 엔트로피(frame_entropy: 상위 앵커 클래스 확률의 이진 엔트로피) 상위 B개. 최신성은 쓰지 않는다."""
    return np.argsort(-cands["ent"])[:B]


def ent_hybrid_sel(cands, B, rng):
    """엔트로피 상위 2B개로 후보를 줄인 뒤 그 안에서 다양성 선택 (hybrid 의 불확실성 대신 엔트로피)."""
    top = np.argsort(-cands["ent"])[:min(2 * B, len(cands["ent"]))]
    return top[diversity_sel({"uncert": cands["uncert"][top], "emb": cands["emb"][top]}, B, rng)]


STRATEGIES = {"entropy": entropy_sel, "ent_hybrid": ent_hybrid_sel, "random": random_sel, "uncertainty": uncertainty_sel, "diversity": diversity_sel, "hybrid": hybrid_sel, "edgesync": edgesync_sel}


def frame_entropy(det, k=20):
    """엣지 모델 출력 엔트로피: 상위 k개 앵커 점수 s 의 이진 엔트로피 평균 -(s ln s + (1-s) ln(1-s))."""
    s = np.clip(det["topk"][:k].astype(np.float64), 1e-6, 1 - 1e-6)
    return float((-(s * np.log(s) + (1 - s) * np.log(1 - s))).mean())


def frame_uncertainty(det):
    """프레임 불확실성 = 1 - (상위 앵커 점수 평균). 점수가 애매한 프레임일수록 값이 크다."""
    return float(1.0 - det["topk"][:20].mean())


def frame_embedding(bgr):
    """장면 다양성용 임베딩: 16x10 축소 썸네일을 펼친 480차원 벡터. 모델 없이 계산."""
    small = cv2.resize(bgr, (16, 10), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return small.ravel()

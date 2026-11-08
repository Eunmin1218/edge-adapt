"""합성 세계 (SHIFT·GPU 없이 CPU 로 시험하기 위한 것). 회색 배경 위의 색깔 사각형 6종(클래스 = 색).
환경: clear / night(밝기 x0.22 + 파란 기운 + 노이즈: 전역 색조 변환으로 거의 되돌릴 수 있음) /
      fog(대비 감소 + 흐림: 전역 변환 + 공간 필터) / rain(어두워짐 + 빗줄기 + 흐림: 공간 구조 변화) / night_rain(두 가지 결합: 학습 때 한 번도 본 적 없는 조합)
Student 는 clear 만 보고 학습하므로 나머지 환경에서 성능이 떨어진다. 성능 근거로 쓰지 않는다 (방향 확인용)."""
import numpy as np, cv2
from .common import IMG_W, IMG_H

COL = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 220, 220), (220, 0, 220), (255, 160, 0)]      # BGR, 클래스별 색
ENVS = ("clear", "night", "fog", "rain", "night_rain")
ENVS3 = ("clear", "night", "glare", "heavy_rain", "clear_sparse", "clear_dense")      # toy3: 통계 정렬의 가정(전역 통계 변화)을 깨는 환경 — 국소 반사(glare), 빗줄기 과밀(heavy_rain), 영상은 같고 물체 수만 다른 내용 변화(sparse/dense)


def _halos(img, rng, k):
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float32); add = np.zeros_like(img)
    for _ in range(k):
        cx, cy, sg, amp = rng.integers(0, IMG_W), rng.integers(0, IMG_H), rng.uniform(35, 70), rng.uniform(150, 230)
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg ** 2))[..., None] * amp; add += g * np.array([0.8, 0.95, 1.0], np.float32)
    return img + add


def _streaks(img, rng, n, bright, length=(18, 40)):
    out = img.copy()
    for _ in range(n):
        x, y = int(rng.integers(0, IMG_W)), int(rng.integers(0, IMG_H)); L = int(rng.integers(*length)); dx = int(rng.integers(-4, 1))
        cv2.line(out, (x, y), (x + dx, y + L), (bright,) * 3, 1)
    return cv2.addWeighted(img, 0.6, out, 0.4, 0)


def make_img(rng, env):
    base = np.full((IMG_H, IMG_W, 3), rng.integers(100, 140), np.float32) + np.linspace(-15, 15, IMG_W)[None, :, None]
    boxes, cls = [], []
    n_try = {"clear_sparse": rng.integers(1, 3), "clear_dense": rng.integers(10, 17)}.get(env)
    for _ in range(int(n_try) if n_try is not None else rng.integers(3, 7)):
        c = int(rng.integers(0, 6)); w, h = int(rng.integers(40, 130)), int(rng.integers(40, 130)); x, y = int(rng.integers(0, IMG_W - w)), int(rng.integers(0, IMG_H - h))
        if any(not (x + w < b[0] or b[2] < x or y + h < b[1] or b[3] < y) for b in boxes): continue
        cv2.rectangle(base, (x, y), (x + w, y + h), COL[c], -1); cv2.rectangle(base, (x, y), (x + w, y + h), (30, 30, 30), 2); boxes.append([x, y, x + w, y + h]); cls.append(c)
    if env == "night": base = base * 0.22 + np.array([14, 6, 0]) + rng.normal(0, 10, base.shape)
    elif env == "fog": base = cv2.GaussianBlur(base * 0.45 + 135, (0, 0), 2.0) + rng.normal(0, 4, base.shape)
    elif env == "rain": base = cv2.GaussianBlur(_streaks(base * 0.7, rng, 90, 190), (0, 0), 1.0) + rng.normal(0, 8, base.shape)
    elif env == "night_rain": base = cv2.GaussianBlur(_streaks(base * 0.22 + np.array([14, 6, 0]), rng, 90, 90), (0, 0), 1.5) + rng.normal(0, 10, base.shape)
    elif env == "glare": base = _halos(base * 0.22 + np.array([14, 6, 0]), rng, int(rng.integers(2, 4))) + rng.normal(0, 10, base.shape)
    elif env == "heavy_rain": base = cv2.GaussianBlur(_streaks(base * 0.6, rng, 320, 220), (0, 0), 1.5) + rng.normal(0, 8, base.shape)
    else: base = base + rng.normal(0, 3, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8), (np.array(boxes, float).reshape(-1, 4), np.array(cls, int))


def stream(seed, segs):
    rng = np.random.default_rng(seed); out = []
    for env, n in segs:
        for _ in range(n):
            img, gt = make_img(rng, env); out.append({"img": img, "gt": gt, "env": env})
    return out


def write_set(root, name, n, envs, seed):
    from .common import CLASSES
    d = root / "data" / name; (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(exist_ok=True); rng = np.random.default_rng(seed)
    for i in range(n):
        img, (b, c) = make_img(rng, envs[i % len(envs)]); cv2.imwrite(str(d / "images" / f"{i:04d}.jpg"), img)
        (d / "labels" / f"{i:04d}.txt").write_text("\n".join(f"{k} {(x1 + x2) / 2 / IMG_W:.6f} {(y1 + y2) / 2 / IMG_H:.6f} {(x2 - x1) / IMG_W:.6f} {(y2 - y1) / IMG_H:.6f}" for k, (x1, y1, x2, y2) in zip(c, b)))

"""서버 측 적응 학습의 대안: Teacher 의 '조밀한 소프트 출력'을 직접 따라 하도록 Student 를 학습 (dense knowledge distillation).

기존 방식(ultralytics 학습)은 Teacher 의 박스를 신뢰도 tau 로 잘라 '하드 라벨'로 만든 뒤 학습한다. 프레임이 8~16장뿐일 때 두 가지 정보가 버려진다.
  (1) tau 아래의 약한 검출·배경의 상대적 확신도 (예: 어두워서 0.4 로 보이는 차량) → 하드 라벨에서는 '배경'으로 학습됨
  (2) 박스 위치의 분포(DFL 의 4변 x 16구간 확률분포) → 하드 라벨에서는 한 점으로 줄어듦
여기서는 두 정보를 모두 손실에 사용한다.
  분류 손실 : Student 클래스 로짓 vs Teacher 클래스 확률(소프트 타깃)  →  BCEWithLogits, 정규화는 Teacher 확률의 합
  박스 손실 : Teacher 확률 >= tau 인 앵커에서, 박스 분포 간 KL(Teacher || Student), 가중치 = Teacher 확률
증강: Teacher 는 원본(좌우 반전만), Student 는 강한 색·밝기·잡음 증강을 본다 (Unbiased Teacher / Mean Teacher 계열의 weak-to-strong 일관성).
      기하 변환은 좌우 반전만 써서 조밀 출력의 위치가 Teacher 와 1:1 로 대응한다. 에폭마다 새 증강이 나오므로 소수 프레임이 여러 장처럼 쓰인다.
Teacher 와 Student 는 같은 스트라이드·reg_max 구조여야 한다 (YOLO11 계열은 크기만 다르다)."""
import math, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from .onnx_detector import preprocess
from .common import NET_H, NET_W, IMG_H, IMG_W, PAD_Y


def strong_aug(bgr, rng):
    """색·밝기·대비·감마·잡음·흐림. 박스 위치를 바꾸지 않는다."""
    x = bgr.astype(np.float32) / 255.0
    x = np.clip(x, 0, 1) ** float(rng.uniform(0.6, 1.8))                       # 감마(어둡게/밝게)
    x = x * float(rng.uniform(0.5, 1.3)) + float(rng.uniform(-0.08, 0.08))       # 대비·밝기
    hsv = cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
    hsv[..., 1] *= float(rng.uniform(0.6, 1.3)); hsv[..., 0] = (hsv[..., 0] + float(rng.uniform(-10, 10))) % 360
    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if rng.random() < 0.4: x = cv2.GaussianBlur(x, (0, 0), float(rng.uniform(0.5, 1.5)))
    x = x + rng.normal(0, float(rng.uniform(0.0, 0.03)), x.shape).astype(np.float32)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _maps(feats, reg):
    """스케일별 raw 맵 리스트 -> (B, 4*reg_max, A), (B, nc, A)"""
    cat = torch.cat([f.flatten(2) for f in feats], 2)
    return cat[:, :reg], cat[:, reg:]


def distill_loss(s_feats, t_feats, reg, tau=0.5, box_w=1.0, uw=0.0):
    sb, sc = _maps(s_feats, reg); tb, tc = _maps(t_feats, reg); pt = tc.sigmoid()
    if uw > 0:                                                                     # Teacher 가 확실한(이진 엔트로피 낮은) 앵커에 더 큰 가중: w = (1 - H/ln2)^uw 를 앵커별(클래스 평균)로
        H = -(pt * pt.clamp_min(1e-6).log() + (1 - pt) * (1 - pt).clamp_min(1e-6).log()).mean(1, keepdim=True) / math.log(2.0); wt = (1 - H).clamp_min(0.05) ** uw
        l_cls = (F.binary_cross_entropy_with_logits(sc, pt, reduction="none") * wt).sum() / (pt * wt).sum().clamp_min(1.0)
    else:
        l_cls = F.binary_cross_entropy_with_logits(sc, pt, reduction="sum") / pt.sum().clamp_min(1.0)
    conf = pt.max(1).values; w = conf * (conf >= tau); B, _, A = sb.shape
    s_lp = F.log_softmax(sb.view(B, 4, reg // 4, A), 2); t_p = F.softmax(tb.view(B, 4, reg // 4, A), 2)
    kl = (t_p * (t_p.clamp_min(1e-9).log() - s_lp)).sum(2).mean(1)                # (B, A)
    l_box = (kl * w).sum() / w.sum().clamp_min(1.0)
    return l_cls + box_w * l_box, float(l_cls), float(l_box)


@torch.no_grad()
def _teacher_maps(teacher, x):
    teacher.eval(); _, feats = teacher(x); return [f.detach() for f in feats]


def _net_labels(lb, flip):
    """원본 이미지 기준 정규화 xywh (cls, x, y, w, h) -> 네트워크 입력(패딩 포함) 기준 정규화 xywh. 좌우 반전 반영."""
    lb = np.asarray(lb, np.float32).reshape(-1, 5).copy()
    if len(lb) == 0: return lb
    x, y, w, h = lb[:, 1] * IMG_W / NET_W, (lb[:, 2] * IMG_H + PAD_Y) / NET_H, lb[:, 3] * IMG_W / NET_W, lb[:, 4] * IMG_H / NET_H
    lb[:, 1], lb[:, 2], lb[:, 3], lb[:, 4] = (1 - x if flip else x), y, w, h; return lb


def distill_train(student_model, teacher_model, frames, first_train, epochs=30, lr=3e-3, lrf=0.1, batch=8, tau=0.5, device="cpu", seed=0, box_w=1.0, log=None, labels=None, hard_w=0.0, aug=1,
                  mode="head", tone=False, lr_aff=5e-3, lr_tone=1e-2, uw=0.0):
    """student_model: ultralytics DetectionModel (nn.Module, 제자리 수정됨). frames: BGR 이미지 리스트. 반환: (마지막 손실, 톤 커브 매듭 (3,K) 또는 None).
    mode: 'head' = first_train 번호 이상의 레이어 가중치만 학습 (융합 전 모델, 그 앞 레이어의 BN 통계는 고정)
          'affine' = 모든 합성곱 출력 채널의 배율·이동만 학습 (src/adapt_modules.py, 융합 모델, 학습 후 가중치에 접음)
          'affine+head' = 둘 다.   tone=True 이면 입력 톤 커브도 함께 학습.   uw>0 이면 Teacher 엔트로피가 낮은(확실한) 앵커에 더 큰 가중 (Probabilistic Teacher 의 entropy focal 과 같은 취지)"""
    from .adapt_modules import attach_affine, fold_affine, ToneCurve
    rng = np.random.default_rng(seed); torch.manual_seed(seed); dev = torch.device(device if device != "cpu" else "cpu")
    use_aff = mode in ("affine", "affine+head"); use_head = mode in ("head", "affine+head")
    if use_aff: student_model.fuse(verbose=False)                                  # Conv+BN 융합: BN 통계 문제가 없어지고, 접기 쉬운 형태가 된다
    student, teacher = student_model.float().to(dev), teacher_model.float().to(dev)
    for p in student.parameters(): p.requires_grad_(False)
    groups = []
    if use_head:
        hp = []
        for mod in student.model[first_train:]:
            for p in mod.parameters(): p.requires_grad_(True); hp.append(p)
        groups.append({"params": hp, "lr": lr, "base": lr})
    if use_aff: ap_ = attach_affine(student); [p.to(dev) for p in ap_]; groups.append({"params": ap_, "lr": lr_aff, "base": lr_aff})
    tc = ToneCurve().to(dev) if tone else None
    if tc is not None: groups.append({"params": [tc.knots], "lr": lr_tone, "base": lr_tone})
    params = [p for g in groups for p in g["params"]]
    reg = student.model[-1].reg_max * 4; crit = None
    if hard_w > 0:                                                                  # 하드 라벨 손실(ultralytics 의 v8DetectionLoss: CIoU + DFL + BCE)을 증류 손실에 더한다
        from ultralytics.cfg import get_cfg
        student.args = get_cfg(); crit = student.init_criterion(); assert labels is not None
    base = [preprocess(f) for f in frames]; flipped = [np.ascontiguousarray(x[..., ::-1]) for x in base]
    tmaps = []                                                                      # Teacher 출력은 (프레임, 반전 여부)마다 한 번만 계산
    for x, xf in zip(base, flipped):
        tmaps.append([_teacher_maps(teacher, torch.from_numpy(v).to(dev)) for v in (x, xf)])
    opt = torch.optim.AdamW(groups, weight_decay=0.0); n = len(frames); steps_per = math.ceil(n / batch); total = epochs * steps_per; step = 0; last = (0.0, 0.0, 0.0)
    for ep in range(epochs):
        order = rng.permutation(n)
        for b0 in range(0, n, batch):
            ids = order[b0:b0 + batch]; xs, ts = [], [[] for _ in range(3)]; bi, bc, bb = [], [], []
            for j, i in enumerate(ids):
                fl = int(rng.integers(0, 2)); img = np.ascontiguousarray(frames[i][:, ::-1] if fl else frames[i])
                xs.append(preprocess(strong_aug(img, rng) if aug else img))
                if crit is not None:
                    lb = _net_labels(labels[i], fl); bi += [j] * len(lb); bc += lb[:, :1].tolist(); bb += lb[:, 1:].tolist()
                for k, m in enumerate(tmaps[i][fl]): ts[k].append(m)
            x = torch.from_numpy(np.concatenate(xs)).to(dev); t_feats = [torch.cat(t) for t in ts]
            student.train()
            for mod in student.model[:first_train].modules():
                if isinstance(mod, nn.BatchNorm2d): mod.eval()
            cs = 0.5 * (1 + math.cos(math.pi * step / max(total, 1)))
            for g in opt.param_groups: g["lr"] = g["base"] * (lrf + (1 - lrf) * cs)
            s_feats = student(tc(x) if tc is not None else x); loss, lc, lb = distill_loss(s_feats, t_feats, reg, tau, box_w, uw)
            if crit is not None:
                bt = {"batch_idx": torch.tensor(bi, dtype=torch.float32, device=dev), "cls": torch.tensor(bc, dtype=torch.float32, device=dev).reshape(-1, 1), "bboxes": torch.tensor(bb, dtype=torch.float32, device=dev).reshape(-1, 4)}
                hl, _ = crit(s_feats, bt); loss = loss + hard_w * hl.sum() / len(ids)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 10.0); opt.step(); step += 1; last = (float(loss), lc, lb)
            if tc is not None: tc.project()
        if log and (ep + 1) % max(epochs // 5, 1) == 0: log(f"  distill ep {ep + 1}/{epochs} loss={last[0]:.3f} (cls {last[1]:.3f}, box {last[2]:.3f})")
    student.eval()
    if use_aff: fold_affine(student)
    return last, (tc.knots.detach().cpu().numpy().astype(np.float32) if tc is not None else None)

"""어댑터를 '적은 매개변수'로 표현하는 두 모듈.

(1) 채널별 배율·이동 (affine): 융합된 Conv(= Conv+BN 융합 뒤) 하나하나의 출력 채널 c 에 y_c -> (1+ds_c)*y_c + t_c 를 학습한다. 활성함수 앞에서 적용되므로
    act((1+ds)*(W x + b) + t) = act(((1+ds) W) x + ((1+ds) b + t)) 로, 학습이 끝나면 가중치·편향에 그대로 접어 넣을 수 있다(모델 구조 불변, ONNX initializer 이름 불변).
    전송은 접은 가중치 전체가 아니라 (ds, t) 만 보낸다 (src/delta_codec.py 의 'auto' 부호화가 이를 알아본다). 채널 수의 2배만큼의 값이므로 YOLO11n 에서 수십 KB.
    같은 매개변수 집합을 TENT 는 BN γ/β 로, BitFit 은 편향으로 사용한다. 여기서는 Teacher 증류 손실로 학습한다.
(2) 입력 톤 커브: RGB 채널마다 구간 선형 곡선(K 개 매듭, 항등으로 시작) y = f_c(x). 엣지가 전처리 직후 입력에 적용한다. 전역 밝기·대비·색 기운 변화(밤·안개)를 네트워크 앞에서 되돌린다. 3 x K 개 값."""
import numpy as np, torch, torch.nn as nn
from ultralytics.nn.modules import Conv, DWConv

K_KNOTS = 17


def _targets(model):
    """접을 수 있는 합성곱: (모듈, Conv2d, 활성함수 또는 None). 융합 전 모델이면 오류."""
    outs, owned = [], set()
    for m in model.modules():
        if isinstance(m, Conv):
            assert not hasattr(m, "bn"), "affine 어댑터는 융합된(fuse) 모델에서만 쓸 수 있다"
            outs.append((m, m.conv, m.act)); owned.add(id(m.conv))
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and id(m) not in owned and m.bias is not None and m.in_channels > 1 and m.out_channels > 1 and m.weight.requires_grad is not None:
            if type(m.__class__).__name__ != "DFL" and m.weight.shape[1] > 1: outs.append((m, m, None))
    return outs


def attach_affine(model):
    """모든 합성곱에 (ds, t) 를 붙이고 학습 대상 매개변수 리스트를 돌려준다. 원래 forward 는 보관."""
    params, self_state = [], []
    for owner, conv, act in _targets(model):
        c = conv.out_channels; ds = nn.Parameter(torch.zeros(c)); t = nn.Parameter(torch.zeros(c)); owner._ds, owner._t = ds, t; params += [ds, t]
        old = owner.forward

        def fwd(x, owner=owner, conv=conv, act=act, old=old):
            y = (old(x) if owner is conv else conv(x)) * (1 + owner._ds).view(1, -1, 1, 1) + owner._t.view(1, -1, 1, 1)
            return act(y) if act is not None else y
        owner._old_forward = old; owner.forward = fwd; self_state.append(owner)
    model._affine_owners = self_state; return params


@torch.no_grad()
def fold_affine(model):
    """학습한 (ds, t) 를 합성곱 가중치·편향에 접고 모듈에서 제거한다."""
    for owner in model._affine_owners:
        conv = owner.conv if isinstance(owner, Conv) else owner; s = 1 + owner._ds.detach()
        conv.weight.mul_(s.view(-1, 1, 1, 1).to(conv.weight.dtype)); conv.bias.copy_((conv.bias * s + owner._t.detach()).to(conv.bias.dtype))
        owner.forward = owner._old_forward; del owner._old_forward; del owner._ds; del owner._t
        if owner.__dict__.get("forward") is not None and not isinstance(owner, Conv): del owner.__dict__["forward"]
    del model._affine_owners


class ToneCurve(nn.Module):
    """RGB 채널별 구간 선형 곡선. 입력 x: (B,3,H,W) 값 범위 [0,1] (RGB 순서). 항등으로 시작."""
    def __init__(self, k=K_KNOTS):
        super().__init__(); self.k = k; self.knots = nn.Parameter(torch.linspace(0, 1, k).repeat(3, 1))

    def forward(self, x):
        pos = x.clamp(0, 1) * (self.k - 1); i0 = pos.floor().clamp(0, self.k - 2).long(); fr = pos - i0.float(); out = torch.empty_like(x)
        for c in range(3):
            kn = self.knots[c]; out[:, c] = kn[i0[:, c]] * (1 - fr[:, c]) + kn[i0[:, c] + 1] * fr[:, c]
        return out

    @torch.no_grad()
    def project(self):
        """단조 증가 + [0,1] 범위로 사영 (곡선이 뒤집히지 않게)."""
        kn = torch.cummax(self.knots.clamp(0, 1), dim=1).values; self.knots.copy_(kn)


def apply_tone_np(x, knots):
    """x: (1,3,H,W) float32 [0,1] RGB, knots: (3,K). 엣지(ONNX 입력 전)에서 사용."""
    k = knots.shape[1]; xs = np.linspace(0, 1, k); out = np.empty_like(x)
    for c in range(3): out[0, c] = np.interp(x[0, c], xs, knots[c]).astype(np.float32)
    return out

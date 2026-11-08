"""PyTorch 탐지기: Teacher 단독(T)·Student 기준(동등성 확인), BN-Adapt, TENT(기준선 C).
TENT(Wang et al., ICLR'21)는 BN의 scale/shift(γ,β)만 엔트로피 최소화로 갱신한다. 분류용 방법이므로 탐지에는 아래처럼 옮긴다:
  손실 = 상위 K개 앵커의 클래스 시그모이드 확률에 대한 이진 엔트로피 평균 (배경 앵커가 대부분이라 상위 K개만 사용).
예측은 현재 프레임의 BN 통계로 한다(원 논문과 같음). 주의: ONNX export 는 Conv+BN 을 융합하지만, 여기서는 융합 전 PyTorch 모델을 그대로 쓴다(BN 파라미터가 살아 있어야 TENT 가 가능)."""
import numpy as np, torch
from ultralytics import YOLO
from .onnx_detector import preprocess, decode


class TorchDetector:
    def __init__(self, pt, mode="none", lr=1e-3, topk=100, conf=0.25, device="cuda:0"):
        assert mode in ("none", "bn", "tent")
        self.mode, self.topk, self.conf, self.dev = mode, topk, conf, torch.device(device if torch.cuda.is_available() else "cpu")
        self.m = YOLO(pt).model.float().to(self.dev); self.nc = self.m.model[-1].nc; self.reg = self.m.model[-1].reg_max * 4
        for p in self.m.parameters(): p.requires_grad_(False)
        self.params = []
        if mode == "tent":
            for mod in self.m.modules():
                if isinstance(mod, torch.nn.BatchNorm2d):
                    mod.weight.requires_grad_(True); mod.bias.requires_grad_(True); self.params += [mod.weight, mod.bias]
            self.opt = torch.optim.Adam(self.params, lr=lr)
        self.m.eval()

    def infer(self, bgr):
        """한 번의 순전파: Detect 는 평가 모드(디코딩된 출력 + 스케일별 raw 맵을 함께 반환), BN 만 학습 모드(현재 프레임의 통계 사용).
        TENT/BN-Adapt 원 논문처럼 '현재 입력의 통계로 정규화한 출력'으로 예측하고, TENT 는 같은 순전파의 raw 맵으로 손실을 계산해 BN γ/β 를 갱신한다(다음 프레임부터 반영)."""
        x = torch.from_numpy(preprocess(bgr)).to(self.dev)
        self.m.eval()
        if self.mode in ("bn", "tent"):
            for mod in self.m.modules():
                if isinstance(mod, torch.nn.BatchNorm2d): mod.train()
        with torch.set_grad_enabled(self.mode == "tent"):
            y, feats = self.m(x)                                   # y: (1, 4+nc, A) 디코딩된 출력, feats: [(1, 4*reg_max+nc, h, w), ...]
            if self.mode == "tent":
                logits = torch.cat([f.flatten(2)[:, self.reg:] for f in feats], 2)[0]       # (nc, A)
                p = logits.sigmoid(); top = p.max(0).values.topk(min(self.topk, p.shape[1])).indices; p = p[:, top].clamp(1e-6, 1 - 1e-6)
                loss = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
                self.opt.zero_grad(); loss.backward(); self.opt.step()
        return decode(y[0].detach().float().cpu().numpy(), self.conf)

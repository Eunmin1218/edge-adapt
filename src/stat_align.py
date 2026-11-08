"""엣지에서 라벨 없이 즉시 만드는 어댑터: 채널 통계 정렬 (AdaBN 계열을 '소스 통계'를 목표로 하도록 바꾼 것).

원리: 융합된 각 합성곱 출력 y (활성함수 앞)의 채널별 평균 mu, 표준편차 sd 는 입력 영상의 분포(밝기·대비·잡음)에 따라 변한다.
      학습 데이터(소스, 맑은 날)에서 잰 평균·표준편차 (mu_s, sd_s) 를 저장해 두고, 새 환경 프레임 몇 장에서 잰 (mu_b, sd_b) 를 이 값에 맞추는
      채널별 배율·이동 s = (sd_s/sd_b)**lam, t = m* - s*mu_b 를 구한다 (lam=1 이면 완전 정렬, lam<1 이면 부분 정렬).
      앞 층을 이미 보정한 뒤의 활성을 다음 층에서 재므로 한 번의 순전파 안에서 층 순서대로 계산된다 (훅이 출력을 바꿔 돌려준다).
      결과는 가중치·편향에 접어 넣어 기존 어댑터(ONNX initializer 차이)와 같은 형식이 된다.
비용: 라벨·Teacher·서버·학습이 필요 없다. 프레임 몇 장의 순전파 한 번. 전송 바이트 0 (소스 통계는 배포 때 한 번 저장).
주의: 마지막 검출 분기의 1x1 합성곱(클래스·박스 출력)은 정렬하지 않는다 (물체 내용에 따라 통계가 달라지는 층).
"""
import copy, numpy as np, torch
from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from .adapt_modules import _targets
from .onnx_detector import preprocess


class StatAligner:
    def __init__(self, student_pt, src_frames, device="cpu"):
        self.pt = str(student_pt); self.yolo = YOLO(self.pt); self.yolo.model.float().eval().fuse(verbose=False); self.model = self.yolo.model
        tg = _targets(self.model); self.layers = [(o, c) for o, c, a in tg if isinstance(o, Conv)]            # Conv 블록의 합성곱만 (최종 1x1 출력층 제외)
        self.src = self._moments(src_frames)

    @torch.no_grad()
    def _moments(self, frames):
        S = [[0.0, 0.0, 0] for _ in self.layers]; hs = []
        for i, (o, c) in enumerate(self.layers):
            def hook(m, inp, out, i=i):
                y = out.double(); S[i][0] = S[i][0] + y.sum((0, 2, 3)); S[i][1] = S[i][1] + (y * y).sum((0, 2, 3)); S[i][2] += y.shape[0] * y.shape[2] * y.shape[3]
            hs.append(c.register_forward_hook(hook))
        for f in frames: self.model(torch.from_numpy(preprocess(f)))
        for h in hs: h.remove()
        out = []
        for s1, s2, n in S:
            mu = s1 / n; sd = (s2 / n - mu * mu).clamp_min(0).sqrt(); out.append((mu.float(), sd.float()))
        return out

    @torch.no_grad()
    def fit(self, frames, lam=1.0, s_min=0.25, s_max=4.0):
        """frames 의 통계를 소스 통계에 정렬하는 (s, t) 리스트와 서명 벡터를 돌려준다. 모델은 변하지 않는다(훅만 일시 설치)."""
        x = torch.from_numpy(np.concatenate([preprocess(f) for f in frames])); st, hs = [None] * len(self.layers), []
        for i, (o, c) in enumerate(self.layers):
            def hook(m, inp, out, i=i):
                mu = out.mean((0, 2, 3)); sd = out.var((0, 2, 3), unbiased=False).sqrt(); ms, ss = self.src[i]
                ok = (sd > 1e-3) & (ss > 1e-3); s = torch.where(ok, (ss / sd.clamp_min(1e-6)) ** lam, torch.ones_like(sd)).clamp(s_min, s_max)
                t = torch.where(ok, mu + lam * (ms - mu) - s * mu, torch.zeros_like(mu)); st[i] = (s, t, mu, sd)
                return out * s.view(1, -1, 1, 1) + t.view(1, -1, 1, 1)
            hs.append(c.register_forward_hook(hook))
        self.model(x)
        for h in hs: h.remove()
        sig = np.concatenate([[float(torch.log(s).mean()), float(((mu - ms) / ss.clamp_min(1e-3)).mean())] for (s, t, mu, sd), (ms, ss) in zip(st, self.src)])
        return [(s, t) for s, t, _, _ in st], sig.astype(np.float32)

    @torch.no_grad()
    def signature(self, frames):
        """정렬 없이 환경 서명만 (프레임 통계의 소스 대비 로그 배율·이동). fit 과 같은 값이지만 보정 전 활성 기준이 아니라 순차 보정 기준임에 유의."""
        return self.fit(frames)[1]

    def adapter(self, frames, base_inits, lam=1.0, tmp_dir=None):
        """접은 가중치를 ONNX 로 내보내 base 와 다른 initializer 만 담은 어댑터 dict 로 돌려준다 (서버 어댑터와 같은 형식)."""
        import tempfile
        from pathlib import Path
        from .server_core import save_pt, export_onnx, onnx_inits
        st, sig = self.fit(frames, lam); y = YOLO(self.pt); y.model.float().eval().fuse(verbose=False)
        lay = [(o, c) for o, c, a in _targets(y.model) if isinstance(o, Conv)]; assert len(lay) == len(st)
        for (o, c), (s, t) in zip(lay, st):
            c.weight.mul_(s.view(-1, 1, 1, 1)); c.bias.copy_(c.bias * s + t)
        d = Path(tmp_dir or tempfile.mkdtemp()); d.mkdir(parents=True, exist_ok=True); save_pt(y, d / "sa.pt"); new = onnx_inits(export_onnx(d / "sa.pt", d / "sa.onnx"))
        ad = {k: v for k, v in new.items() if k in base_inits and v.shape == base_inits[k].shape and not np.array_equal(v, base_inits[k])}
        return ad, sig

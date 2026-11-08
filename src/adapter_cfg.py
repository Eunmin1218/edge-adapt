"""서버 적응 설정 읽기 (torch 불필요)."""
from pathlib import Path


def load_adapter_cfg(path=None):
    """configs/adapter.json (scripts/tune_adapter.py 의 결과)이 있으면 읽고, 없으면 기본값. 반환: dict(epochs, lr, train_layers, tau, batch, lrf, trainer, selector, mode, tone, alpha, lr_aff, lr_tone, aug)
    기본값은 합성 시험(scripts/ablate_adapter.py, GUIDE 0.1.1~0.1.2절)에서 증류 학습이 하드 라벨보다 높았고, 모든 합성곱 채널별 배율·이동(mode=affine, 약 70KB)을 가중치 보간 계수 α=1.25 로 적용한 것이 같은 조건의 헤드 어댑터(약 360KB)보다 높았다. 실제 데이터의 값은 tune_adapter.py 로 정한다."""
    import json
    from .common import ROOT
    d = {"epochs": 60, "lr": 1e-2, "train_layers": 0, "tau": 0.5, "batch": 0, "lrf": 0.1, "trainer": "distill", "selector": "diversity",
         "mode": "affine", "tone": 0, "alpha": 1.25, "lr_aff": 0.03, "lr_tone": 0.01, "aug": 0}      # batch 0 = 자동(증류: 프레임 수의 절반 → epoch 당 2스텝, 총 60스텝)
    f = Path(path) if path else ROOT / "configs/adapter.json"
    if f.exists(): d.update({k: v for k, v in json.load(open(f)).items() if k in d})
    return d


def server_kwargs(cfg):
    """load_adapter_cfg 의 결과 -> AdapterServer 생성 인자 (selector 는 서버가 아니라 엣지 쪽 선별에 쓰이므로 제외)."""
    keys = ("epochs", "lr", "train_layers", "tau", "batch", "lrf", "trainer", "mode", "tone", "alpha", "lr_aff", "lr_tone", "aug")
    kw = {k: cfg[k] for k in keys}; kw["tone"] = bool(kw["tone"]); return kw

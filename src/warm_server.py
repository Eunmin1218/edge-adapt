"""워밍업 부분 서버: 서버가 받은 프레임으로 엣지와 같은 통계 정렬 어댑터를 먼저 Student 에 접어 넣고, 레이어 번호 >= K 인 합성곱의 채널별 (ds, t) 만 증류로 학습한다.
엣지도 같은 프레임으로 통계 정렬 어댑터를 직접 만들 수 있으므로(전송 0), 전송하는 것은 학습한 층의 잔차 (ds, t) 뿐이다 (bits 비트 대칭 양자화 + zlib).
__call__ 은 엣지가 최종 적용할 전체 어댑터 dict(통계 정렬 + 양자화된 잔차)를 돌려주고, 전송 바이트는 last_bytes 에 남긴다. (sim.py 가 last_bytes 가 있으면 ship 대신 이 값을 쓴다)
scripts/codec_warm.py 에서 먼저 오프라인 시험한 절차를 그대로 옮긴 것이다."""
import copy, time, zlib
import numpy as np, torch
from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from . import adapt_modules as AM
from .adapt_modules import _targets
from .distill import distill_train
from .server_core import save_pt, export_onnx, onnx_inits


class WarmServer:
    def __init__(self, inner, aligner, K=23, bits=4, alpha=1.0):
        self.inner, self.sa, self.K, self.bits, self.alpha = inner, aligner, K, bits, alpha; self.last_bytes = None; self.last_info = {}

    def __getattr__(self, name): return getattr(self.__dict__["inner"], name)

    @staticmethod
    def _layer_of(model):
        m = {}
        for i, mod in enumerate(model.model):
            for s in mod.modules(): m[id(s)] = i
        return m

    def __call__(self, frames, env_id):
        t0 = time.time(); S = self.inner; S.n_jobs += 1; K, bits = self.K, self.bits
        y = YOLO(str(S.student_pt)); y.model.float().eval().fuse(verbose=False); model = y.model
        st, _ = self.sa.fit(frames); lay = [(o, c) for o, c, a in _targets(model) if isinstance(o, Conv)]
        with torch.no_grad():
            for (o, c), (s, t) in zip(lay, st): c.weight.mul_(s.view(-1, 1, 1, 1)); c.bias.copy_(c.bias * s + t)
        lm = self._layer_of(model); vecs = []; orig_att, orig_fold = AM.attach_affine, AM.fold_affine
        buf = S.replay.setdefault(env_id, []); buf += [(f, None) for f in frames]; buf[:] = buf[-S.replay_max:]

        def att(m):
            params, owners = [], []
            for owner, conv, act in _targets(m):
                if lm.get(id(owner), 0) < K: continue
                c = conv.out_channels; ds = torch.nn.Parameter(torch.zeros(c)); t = torch.nn.Parameter(torch.zeros(c)); owner._ds, owner._t = ds, t; params += [ds, t]; old = owner.forward

                def fwd(x, owner=owner, conv=conv, act=act, old=old):
                    yv = (old(x) if owner is conv else conv(x)) * (1 + owner._ds).view(1, -1, 1, 1) + owner._t.view(1, -1, 1, 1)
                    return act(yv) if act is not None else yv
                owner._old_forward = old; owner.forward = fwd; owners.append(owner)
            m._affine_owners = owners; return params

        def fold(m):
            L = 2 ** (bits - 1) - 1
            with torch.no_grad():
                for owner in m._affine_owners:                                            # 전송될 값 그대로(양자화 후)를 접는다
                    v = (self.alpha * torch.cat([owner._ds.detach(), owner._t.detach()])).numpy(); C = len(v) // 2; parts = []
                    for sl in (slice(0, C), slice(C, 2 * C)):
                        x = v[sl]; mx = float(np.abs(x).max()); sc = np.float16(mx / L if mx > 0 else 1.0); q = np.clip(np.rint(x / np.float32(sc)), -L, L).astype(np.int8); parts.append(sc.tobytes() + q.tobytes())
                        (owner._ds if sl.start == 0 else owner._t).data[:] = torch.from_numpy(q.astype(np.float32) * np.float32(sc))
                    vecs.append(b"".join(parts))
            orig_fold(m)
        AM.attach_affine, AM.fold_affine = att, fold
        try:
            teacher = copy.deepcopy(S.teacher.model)
            distill_train(model, teacher, [f for f, _ in buf], S.first_train, epochs=S.epochs, lr=S.lr, lrf=S.lrf, batch=S.batch or max(1, -(-len(buf) // 2)), mode="affine", tone=False, lr_aff=S.lr_aff, aug=S.aug)
        finally: AM.attach_affine, AM.fold_affine = orig_att, orig_fold
        job = S.work / f"warm{S.n_jobs:04d}"; job.mkdir(parents=True, exist_ok=True); save_pt(y, job / "m.pt"); new = onnx_inits(export_onnx(job / "m.pt", job / "m.onnx"))
        ad = {k: v for k, v in new.items() if k in S.base and v.shape == S.base[k].shape and not np.array_equal(v, S.base[k])}
        self.last_bytes = len(zlib.compress(b"".join(vecs), 9)) + 2 * len(vecs)
        self.last_info = {"sec": round(time.time() - t0, 1), "n_train": len(buf), "bytes": self.last_bytes, "n_tensors": len(ad)}; print("[warm]", self.last_info, flush=True); return ad

"""서버 측 적응: Teacher 의사 라벨 → 신뢰도 필터 → Detect 헤드만 학습 → 어댑터(= base ONNX 대비 값이 바뀐 initializer 묶음) 추출."""
import shutil, tempfile, time, numpy as np, cv2, onnx, torch, torch.nn as nn
from pathlib import Path
from onnx import numpy_helper
from ultralytics import YOLO
from .common import CLASSES, NET_H, NET_W
from .adapter_cfg import load_adapter_cfg  # noqa: F401  (torch 없이도 설정을 읽을 수 있도록 별도 모듈)


def save_pt(yolo, path):
    """현재 yolo.model 의 가중치를 .pt 로 저장한다. YOLO.save() 는 원래 체크포인트(ckpt)를 저장하는 구현일 수 있어 쓰지 않는다."""
    ck = dict(getattr(yolo, "ckpt", None) or {})
    ck["model"] = yolo.model; ck["ema"] = None; ck.setdefault("train_args", {})
    torch.save(ck, str(path))


def export_onnx(pt, out_onnx):
    """pt 를 ONNX 로 내보내 out_onnx 에 저장. 임시 폴더에서 내보내 기존 파일을 덮어쓰거나 옮기지 않는다.
    simplify=False 로 두어 export 마다 initializer 이름이 같게 유지한다."""
    tmp = Path(tempfile.mkdtemp()); shutil.copy(str(pt), tmp / "m.pt")
    p = YOLO(str(tmp / "m.pt")).export(format="onnx", imgsz=[NET_H, NET_W], simplify=False, dynamic=False, half=False, opset=12)
    Path(out_onnx).parent.mkdir(parents=True, exist_ok=True); shutil.move(p, out_onnx); shutil.rmtree(tmp, ignore_errors=True)
    return out_onnx


def onnx_inits(path):
    m = onnx.load(str(path)); return {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}


from . import slog


class AdapterServer:
    def __init__(self, student_pt, base_onnx, teacher_pt, work, tau=0.5, epochs=10, lr=1e-3, device="auto", replay_max=300, imgsz=640, train_layers=0, batch=8, lrf=0.1, trainer="ultra", box_w=1.0, hard_w=0.0, aug=1, mode="head", tone=False, lr_aff=5e-3, lr_tone=1e-2, alpha=1.0, uw=0.0):
        self.student_pt, self.work, self.tau, self.epochs, self.lr, self.device, self.replay_max, self.imgsz = Path(student_pt), Path(work), tau, epochs, lr, device, replay_max, imgsz
        if device == "auto":                                                  # GPU(NVIDIA)가 있으면 사용, 없으면 CPU
            import torch; device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device; print(f"[server] device={device}")
        self.work.mkdir(parents=True, exist_ok=True); self.teacher = YOLO(str(teacher_pt))
        self.base = onnx_inits(base_onnx); self.head_idx = len(YOLO(str(student_pt)).model.model) - 1     # 마지막 레이어 = Detect
        self.batch, self.lrf = batch, lrf                                     # 소수 프레임이므로 배치를 작게 해 epoch 당 학습 스텝을 늘린다 (스텝 수 = epochs x ceil(프레임 수 / batch))
        self.train_layers = train_layers; self.first_train = max(self.head_idx - train_layers, 0)         # 학습 시작 레이어 번호 (0 = 헤드만이 아니라 헤드 앞 train_layers 개 레이어도 학습)
        assert trainer in ("ultra", "distill"); self.trainer, self.box_w, self.hard_w, self.aug = trainer, box_w, hard_w, aug; self.mode, self.tone, self.lr_aff, self.lr_tone, self.alpha, self.uw = mode, tone, lr_aff, lr_tone, alpha, uw     # ultra: Teacher 하드 라벨 + ultralytics 학습 / distill: Teacher 조밀 소프트 출력 증류 (src/distill.py)
        self.replay = {}; self.n_jobs = 0; self.last_info = {}

    def _teach_stats(self, frames):
        """로그용: Teacher 가 이번 프레임에 낸 의사 라벨 요약 (박스 수, 평균 신뢰도, 클래스 분포). 학습에 쓰는 라벨과 같은 tau 를 쓴다."""
        if slog.quiet: return "(quiet)"
        res = self.teacher.predict(frames, imgsz=self.imgsz, conf=self.tau, iou=0.5, device=self.device, verbose=False)
        n = [len(r.boxes) for r in res]; cf = np.concatenate([r.boxes.conf.cpu().numpy() for r in res]) if sum(n) else np.zeros(0)
        cl = np.concatenate([r.boxes.cls.cpu().numpy() for r in res]).astype(int) if sum(n) else np.zeros(0, int)
        from .common import CLASSES
        hist = ", ".join(f"{CLASSES[c]}:{int((cl == c).sum())}" for c in sorted(set(cl.tolist())))
        return f"의사 라벨 tau={self.tau}: {len(frames)}프레임 → 박스 {sum(n)}개 (프레임당 {np.mean(n):.1f}, 0개인 프레임 {sum(1 for x in n if x == 0)}) 평균 신뢰도 {cf.mean() if len(cf) else float('nan'):.3f} 클래스 {{{hist}}}"

    def _delta_stats(self, ad):
        """로그용: 어댑터가 base 에서 얼마나 바뀌었나 (텐서별 상대 L2 변화 ‖Δ‖/‖base‖)."""
        rel = []
        for k, v in ad.items():
            if k == "__tone__" or k not in self.base: continue
            b = np.asarray(self.base[k], np.float32); d = np.asarray(v, np.float32) - b; rel.append((float(np.linalg.norm(d) / (np.linalg.norm(b) + 1e-9)), k))
        rel.sort(reverse=True); top = ", ".join(f"{k.replace('.weight', '.w').replace('.bias', '.b')}:{r:.3f}" for r, k in rel[:3])
        return f"base 대비 변화 텐서 {len(rel)}개 · 평균 상대변화 {np.mean([r for r, _ in rel]):.4f} · 최대 {rel[0][0]:.3f} · 상위3 {{{top}}}" if rel else "변화 없음"

    def pseudo_label(self, frames):
        res = self.teacher.predict(frames, imgsz=self.imgsz, conf=self.tau, iou=0.5, device=self.device, verbose=False)
        return [np.c_[r.boxes.cls.cpu().numpy(), r.boxes.xywhn.cpu().numpy()] if len(r.boxes) else np.zeros((0, 5)) for r in res]

    def agreement(self, frames, dets, iou_thr=0.5):
        """EdgeSync-lite 프로브: Teacher 박스(신뢰도 >= tau) 중 엣지 박스(같은 클래스, IoU >= iou_thr)가 맞춘 비율의 프레임 평균. Teacher 박스가 없는 프레임은 제외, 전부 없으면 1.0."""
        from .mapeval import iou_matrix
        res = self.teacher.predict(frames, imgsz=self.imgsz, conf=self.tau, iou=0.5, device=self.device, verbose=False); vals = []
        for r, (eb, es, ec) in zip(res, dets):
            if len(r.boxes) == 0: continue
            tb, tc = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
            if len(eb) == 0: vals.append(0.0); continue
            m = iou_matrix(tb, eb) * (tc[:, None] == ec[None, :]); vals.append(float((m.max(1) >= iou_thr).mean()))
        return float(np.mean(vals)) if vals else 1.0

    def _train_ultra(self, job, frames, env_id):
        (job / "images").mkdir(parents=True, exist_ok=True); (job / "labels").mkdir(exist_ok=True)
        labs = self.pseudo_label(frames); buf = self.replay.setdefault(env_id, []); buf += list(zip(frames, labs)); buf[:] = buf[-self.replay_max:]
        for i, (img, lb) in enumerate(buf):
            cv2.imwrite(str(job / "images" / f"{i:05d}.jpg"), img)
            (job / "labels" / f"{i:05d}.txt").write_text("\n".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in lb))
        (job / "data.yaml").write_text(f"path: {job.resolve()}\ntrain: images\nval: images\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
        model = YOLO(str(self.student_pt)); n = self.head_idx

        def bn_eval(tr):                                                     # 동결 구간의 BN 통계가 바뀌지 않도록 매 배치 eval 고정
            for mod in tr.model.model[:n].modules():
                if isinstance(mod, nn.BatchNorm2d): mod.eval()
        model.add_callback("on_train_batch_start", bn_eval)
        model.train(data=str(job / "data.yaml"), epochs=self.epochs, imgsz=self.imgsz, batch=self.batch or 8, device=self.device, freeze=self.first_train, optimizer="AdamW", lr0=self.lr, lrf=self.lrf,
                    warmup_epochs=0, mosaic=0.0, val=False, plots=False, workers=2, project=str(job), name="train", exist_ok=True, verbose=False, amp=False)
        return YOLO(str(job / "train" / "weights" / "last.pt"))

    def __call__(self, frames, env_id):
        t0 = time.time(); self.n_jobs += 1; job = self.work / f"job{self.n_jobs:04d}"; J = f"job{self.n_jobs:04d}"; L = lambda st, m: slog.log(st, m, J)
        L("TEACH", self._teach_stats(frames)); tt = time.time()
        shutil.rmtree(job, ignore_errors=True)                               # 이전 실행의 잔여물이 있으면 비운다
        (job / "images").mkdir(parents=True); (job / "labels").mkdir()
        base = YOLO(str(self.student_pt)); n = self.head_idx
        if self.trainer == "distill":
            from .distill import distill_train
            buf = self.replay.setdefault(env_id, []); buf += list(zip(frames, self.pseudo_label(frames) if self.hard_w > 0 else [None] * len(frames))); buf[:] = buf[-self.replay_max:]
            L("REPLAY", f"env={env_id!s} 버퍼 {len(buf)}/{self.replay_max}프레임 (이번 업로드 {len(frames)}장 포함)")
            L("TRAIN", f"mode={self.mode} 학습 레이어 {self.first_train}~{self.head_idx} epochs={self.epochs} lr={self.lr} lrf={self.lrf} batch={self.batch or max(1, -(-len(buf) // 2))} tau={self.tau} aug={self.aug} device={self.device}"); tt = time.time()
            tuned = YOLO(str(self.student_pt)); _, knots = distill_train(tuned.model, self.teacher.model, [f for f, _ in buf], self.first_train, labels=[l for _, l in buf], hard_w=self.hard_w, aug=self.aug, epochs=self.epochs, lr=self.lr, lrf=self.lrf, batch=self.batch or max(1, -(-len(buf) // 2)), tau=self.tau, device=self.device, box_w=self.box_w,
                                      mode=self.mode, tone=self.tone, lr_aff=self.lr_aff, lr_tone=self.lr_tone, uw=self.uw, log=lambda m: L("TRAIN", m.strip()))
            L("TRAIN", f"완료 {time.time() - tt:.1f}s"); shutil.rmtree(job, ignore_errors=True)
        else:
            tuned = self._train_ultra(job, frames, env_id); buf = self.replay[env_id]
        job.mkdir(parents=True, exist_ok=True); knots = knots if self.trainer == "distill" else None
        if self.trainer == "distill" and self.mode != "head": save_pt(tuned, job / "merged.pt")      # 융합 모델(affine 을 접은 것)을 그대로 저장
        else:
            sd, bsd = tuned.model.state_dict(), base.model.state_dict()
            keep = tuple(f"model.{i}." for i in range(self.first_train, n + 1))
            for k in sd:                                                         # 학습 대상 레이어 밖의 모든 값(가중치+BN 통계)을 원래대로 되돌린다
                if not k.startswith(keep): sd[k] = bsd[k]
            base.model.load_state_dict(sd); save_pt(base, job / "merged.pt")
        te = time.time(); new = onnx_inits(export_onnx(job / "merged.pt", job / "merged.onnx")); L("EXPORT", f"merged.pt → ONNX {time.time() - te:.1f}s")
        ad = {k: v for k, v in new.items() if k in self.base and v.shape == self.base[k].shape and not np.array_equal(v, self.base[k])}
        if knots is not None: ad["__tone__"] = knots
        if self.alpha != 1.0:
            from .delta_codec import scale_adapter; ad = scale_adapter(ad, self.base, self.alpha)
        assert ad, "어댑터가 비어 있음: 학습이 헤드를 바꾸지 않았거나 ONNX initializer 이름이 base 와 다름 (scripts/check_adapter_pipeline.py 실행)"
        self.last_info = {"sec": round(time.time() - t0, 1), "n_train": len(buf), "adapter_MB": round(sum(np.asarray(v).nbytes for v in ad.values()) / 2**20, 3), "n_tensors": len(ad)}
        L("DELTA", self._delta_stats(ad)); L("DONE", f"총 {self.last_info['sec']}s · 학습 프레임 {len(buf)} · 어댑터 텐서 {len(ad)}개 {self.last_info['adapter_MB']}MB(FP32 원본)")
        shutil.rmtree(job / "images", ignore_errors=True); return ad


"""엣지 모사 스트림 실행기. 한 프레임씩 step() 으로 진행하므로 (1) 오프라인 실험(run_stream)과 (2) 실시간 데모(여러 에이전트를 같은 프레임에 동시 실행)가 같은 코드를 쓴다.
정책(Policy)은 '언제 서버에 요청하는가'만 다르고, 선별·서버 학습·적용 지연·롤백은 모든 정책이 동일하다.
server(frames_bgr, env_id) 의 반환: dict(어댑터) → apply_delay 프레임 뒤 적용(시뮬레이션) / 핸들(ready(), result(), status) → 준비되는 즉시 적용(실시간 데모)."""
import time
import numpy as np, cv2
from .signals import FrameSignal, window_features, ALL_COLS
from .selectors import STRATEGIES, frame_uncertainty, frame_embedding, frame_entropy
from .env import env_signature, EnvRegistry
from .mapeval import windowed_map
from .delta_codec import encode as codec_encode, decode as codec_decode


def ship(ad, base, mode="auto"):
    """서버가 만든 어댑터를 실제 전송 형식으로 부호화했다가 엣지에서 복원한다. 반환: (엣지가 적용할 어댑터, 전송 바이트). 양자화 오차가 성능에 반영된다. base 가 없으면 FP16 크기만 계산."""
    if base is None or ad is None or len(ad) == 0: return ad, (adapter_bytes(ad) if ad else 0)
    raw = codec_encode(ad, base, mode); return codec_decode(raw, base), len(raw)


def adapter_bytes(ad):
    return int(sum(np.asarray(v).astype(np.float16).nbytes for v in ad.values()))     # 전송은 FP16 기준


def wire(frames, q=80, scale=1.0):
    """전송 시뮬레이션: (축소 후) JPEG 부호화 -> 복호 -> 원래 크기로 복원. 반환 (서버가 받는 프레임, 바이트)"""
    out, nb = [], 0
    for f in frames:
        g = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else f
        ok, buf = cv2.imencode('.jpg', g, [cv2.IMWRITE_JPEG_QUALITY, int(q)]); nb += len(buf); d = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        out.append(cv2.resize(d, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else d)
    return out, nb


def jpeg_bytes(bgr, q=80):
    return len(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])[1])


class Policy:
    """should_trigger(t, wf) -> bool.  kind: none | periodic | ams | drift | edgesync
    edgesync(EdgeSync-lite): wf = [Teacher 일치도]. 최근 10개 윈도우 최대 일치도 - 현재 일치도 = 긴급도 d, d >= theta 이면 갱신 요청 (단일 엣지에서는 논문의 '엣지 간 긴급도 순위'가 임계값 판정으로 축소됨)."""
    def __init__(self, kind, period=100, drift_model=None, consec=2, cooldown=100, cols=ALL_COLS, gain=20.0, probe_n=4, theta=0.15, cp=True):
        self.gain = gain; self.probe_n, self.theta = probe_n, theta; self.hist = []; self.prob = 0.0; self.urgency = 0.0
        self.kind, self.period, self.model, self.consec, self.cooldown, self.cols = kind, period, drift_model, consec, cooldown, cols
        self.hits, self.last = 0, -10**9
        self.cp, self.first_hit = cp, None          # cp: 선별 후보를 "처음 드리프트로 판정된 윈도우 이후" 프레임으로 제한(변화 시점 이전의 정상 프레임이 예산을 쓰지 않게 함)

    def should_trigger(self, t, wf):
        if self.kind == "none":
            return False
        if self.kind == "periodic":
            return (t + 1) % self.period == 0
        if self.kind == "ams":     # AMS-lite: 장면 변화가 클수록 갱신 간격을 줄인다 (wf[0] = 윈도우 평균 프레임 변화량)
            gap = self.period / (1.0 + self.gain * wf[0])
            if t - self.last >= gap:
                self.last = t; return True
            return False
        if self.kind == "edgesync":
            ref = max(self.hist[-10:]) if self.hist else wf[0]; self.hist.append(float(wf[0])); self.urgency = float(ref - wf[0])
            if self.urgency >= self.theta and t - self.last >= self.cooldown:
                self.last = t; self.hist = []; return True
            return False
        if self.kind == "drift":
            self.prob = float(self.model.predict_proba(wf[None])[0, 1]) if hasattr(self.model, "predict_proba") else float(self.model.predict(wf[None])[0])
            hit = self.model.predict(wf[None])[0] == 1
            if hit and self.hits == 0: self.first_hit = t
            self.hits = self.hits + 1 if hit else 0
            if self.hits >= self.consec and t - self.last >= self.cooldown:
                self.last = t; self.hits = 0; return True
        return False


def _apply(detector, ad):
    """{} 는 기본(적응 전) 가중치로 복귀."""
    if ad is None or len(ad) == 0:
        detector.reset_adapter()
    else:
        detector.apply_adapter(ad)


class StreamRunner:
    """한 엣지 에이전트. step(img, gt, env) 를 프레임마다 호출한다. state 는 UI 표시용(영문 키):
    static | normal | drift | adapting | offline | recovered | switched | rollback | error"""
    def __init__(self, detector, policy, server, selector="hybrid", budget=16, win=50, pool_size=100, apply_delay=50, rollback_margin=0.05,
                 use_registry=True, seed=0, n_cls=6, env_thr=0.08, probe_fn=None, verify_margin=None, edge_fn=None, edge_sig=None, sig_thr=0.25, gate_margin=0.08, edge_n=12, fast_n=0, fast_margin=0.1, verify_edge=False, agree_thr=None, confirm=False, min_post=0, probe_n=6, probe_q=80, probe_scale=1.0, train_q=80, codec="auto", heldout=False, edge_delay=0, conf_tie=0.0, t0=None, lf_verify=False, lf_margin=0.0, gate_since=False, fc_margin=None, srv_trust=False, srv_all=False, lf_or=False):
        self.fc_margin, self.ref_fc = fc_margin, None
        self.srv_trust, self.srv_all, self.lf_or = srv_trust, srv_all, lf_or; self.t0, self.lf_verify, self.lf_margin, self.gate_since = t0, lf_verify, lf_margin, gate_since; self.t0_S, self.t0_on, self.t0_calm = 0.0, False, 0
        self.conf_tie = conf_tie; self.edge_delay, self.edge_ready = edge_delay, None
        self.heldout, self.train_t = heldout, set()
        self.probe_n, self.probe_q, self.probe_scale, self.train_q, self.codec = probe_n, probe_q, probe_scale, train_q, codec
        self.min_post, self.want = min_post, None
        self.confirm, self.confirm_at = confirm, None
        self.agree_thr = agree_thr; self.verify_edge = verify_edge; self.checked = set()
        self.edge_fn, self.edge_sig, self.sig_thr, self.gate_margin, self.edge_n = edge_fn, edge_sig, sig_thr, gate_margin, edge_n
        self.ref_conf = None; self.last_sig = None; self.fast_n, self.fast_margin, self.fast_block = fast_n, fast_margin, 0; self.refined = set(); self.edge_frames = 0
        self.det, self.policy, self.server, self.budget, self.win, self.pool_size = detector, policy, server, budget, win, pool_size
        self.sel = STRATEGIES[selector]; self.apply_delay, self.rollback_margin, self.use_registry, self.n_cls, self.probe_fn = apply_delay, rollback_margin, use_registry, n_cls, probe_fn
        self.rng = np.random.default_rng(seed); self.sig = FrameSignal(); self.reg = EnvRegistry(env_thr)
        self.cur_env, self.cur_adapter, self.prev_adapter = None, {}, {}
        self.pool, self.fbuf, self.win_frames, self.preds, self.gts, self.events = [], [], [], [], [], []
        self.up = self.down = 0; self.pending = None; self.conf_hist = []; self.wff, self.wenv, self.lat = [], [], []
        self.verify_margin = verify_margin; self.last_change = -1; self.t = 0; self.state = "static" if policy.kind == "none" else "normal"; self.last_det = None; self.has_gt = True; self._envs = []

    def _verify_candidate(self, ad):
        if self.lf_verify:
            pp = [q for q in self.pool if q["t"] not in self.train_t] if self.heldout else self.pool; fr = [q["img"] for q in pp[-self.probe_n:]]
            def cmean_(adapter):
                _apply(self.det, adapter); ds = list(map(self.det.infer, fr)); return float(np.mean([np.mean(d["scores"]) if len(d["scores"]) else 0.0 for d in ds]))
            cur, cand = cmean_(self.cur_adapter), cmean_(ad)
            if __import__("os").environ.get("EDGE_DIAG"):
                _apply(self.det, ad); fc_c = self._flipcons(fr[:4]); _apply(self.det, self.cur_adapter); fc_0 = self._flipcons(fr[:4]); self.events.append(("verify", self.t, round(cur, 3), round(cand, 3), round(fc_0, 3), round(fc_c, 3)))
            _apply(self.det, self.cur_adapter); ok = cand >= cur - self.lf_margin
            if self.lf_or and not ok:                                                   # 신뢰도 기준이 거절해도 뒤집기 일관성이 유지되면(두 약한 신호 중 하나라도 괜찮으면) 받아들인다
                _apply(self.det, ad); fc_c = self._flipcons(fr[:4]); _apply(self.det, self.cur_adapter); fc_0 = self._flipcons(fr[:4]); ok = fc_c >= fc_0 - 0.02
            return ok
        """서버가 보낸 후보 어댑터를 현재 어댑터와 비교 (EdgeSync-lite 의 프로브 재사용): 최근 프레임 몇 장을 서버 Teacher 에 보내 일치도를 받아, 후보가 확실히 나을 때만 받아들인다. 업링크 바이트는 계산에 넣는다."""
        pp = [q for q in self.pool if q["t"] not in self.train_t] if self.heldout else self.pool         # 서버가 학습에 쓴 프레임으로 평가하면 후보가 부풀려져 잘못 받아들여진다
        fr = [q["img"] for q in pp[-self.probe_n:]]; sv, nb = wire(fr, self.probe_q, self.probe_scale); self.up += nb
        def score(adapter):
            _apply(self.det, adapter); return self.probe_fn(sv, [(d["boxes"], d["scores"], d["cls"]) for d in map(self.det.infer, fr)])
        cur, cand = score(self.cur_adapter), score(ad); _apply(self.det, self.cur_adapter); return cand >= cur + 0.02

    def _finish_pending(self, t):
        if self.edge_sig is not None and (self.probe_fn is not None or self.lf_verify) and not self.srv_trust and not self._verify_candidate(self.pending["ad"]):
            self.refined.add(self.pending["env"]); self.events.append(("reject", t)); self.pending = None; self.state = "recovered"; return
        p = self.pending; self.prev_adapter = self.cur_adapter; _apply(self.det, p["ad"]); self.cur_adapter = p["ad"]; self.cur_env = p["env"]
        if self.use_registry and p["env"] < len(self.reg.adapters): self.reg.adapters[p["env"]] = p["ad"]
        self.last_change = t; self.refined.add(p["env"]); self.events.append(("apply", t)); self.pending = None; self.state = "recovered"
        self.conf_hist.append((t, float(np.mean([f[1] for f in self.fbuf[-self.win:]]))))

    def _flipcons(self, frames):
        """BoS 류 대용 지표(간소화): 원본과 좌우 뒤집은 입력의 예측 상자가 서로 얼마나 겹치는지(양방향 최대 IoU 평균). 정답·Teacher 불필요."""
        from .mapeval import iou_matrix
        out = []
        for f in frames:
            d1 = self.det.infer(f); d2 = self.det.infer(f[:, ::-1].copy()); b2 = d2["boxes"].copy()
            if len(b2): b2[:, [0, 2]] = f.shape[1] - b2[:, [2, 0]]
            if len(d1["boxes"]) == 0 and len(b2) == 0: out.append(1.0)
            elif len(d1["boxes"]) == 0 or len(b2) == 0: out.append(0.0)
            else: M = iou_matrix(d1["boxes"], b2); out.append(float(0.5 * (M.max(1).mean() + M.max(0).mean())))
        self.edge_frames += 2 * len(frames); return float(np.mean(out))

    def _t0_apply(self, img, t):
        """T0: 프레임 휘도 평균의 CUSUM(Page) 검출로 켜지고, 켜진 동안 프레임마다 채널 평균·표준편차를 소스 값에 맞춘다(입력 단계 AdaBN). 학습·서버·전송 0.
        z=(휘도평균-소스평균)/소스표준편차, S=max(0,S+|z|-k), S>h 이면 켬, |z|<2 가 3프레임 연속이면 끔. 켠 프레임부터(검출한 그 프레임 포함) 보정한다."""
        T = self.t0; z = (float(img.mean()) - T["lum_mu"]) / T["lum_sd"]   # 채널 평균의 평균 = 휘도 근사(정확한 회색 변환 불필요)
        self.t0_S = max(0.0, self.t0_S + abs(z) - T.get("k", 2.0))
        if not self.t0_on and self.t0_S > T.get("h", 6.0): self.t0_on = True; self.t0_calm = 0; self.events.append(("t0on", t))
        elif self.t0_on:
            self.t0_calm = self.t0_calm + 1 if abs(z) < 2.0 else 0
            if self.t0_calm >= 3: self.t0_on = False; self.t0_S = 0.0; self.events.append(("t0off", t))
        if not self.t0_on: return img
        x = img.reshape(-1, 3).astype(np.float32); m, sd = x.mean(0), x.std(0)
        return np.clip((img.astype(np.float32) - m) / np.maximum(sd, 1e-3) * T["sd"] + T["mu"], 0, 255).astype(np.uint8)

    def step(self, img, gt=None, env=None):
        t, win = self.t, self.win
        if self.t0 is not None: img = self._t0_apply(img, t)
        t0 = time.perf_counter(); det = self.det.infer(img); self.lat.append(time.perf_counter() - t0); self.last_det = det
        self.preds.append((det["boxes"], det["scores"], det["cls"])); self.gts.append(gt); self.has_gt = self.has_gt and gt is not None; self._envs.append(env)
        self.fbuf.append(self.sig.update(det)); self.win_frames.append(img)
        self.pool.append({"t": t, "img": img, "u": frame_uncertainty(det), "e": frame_embedding(img), "ent": frame_entropy(det)}); self.pool = self.pool[-self.pool_size:]
        if self.pending:
            h = self.pending["handle"]
            if h is None:
                if t >= self.pending["ready_t"]: self._finish_pending(t)
            else:
                st = getattr(h, "status", "running")
                if st == "error": self.pending = None; self.state = "error"
                elif h.ready():
                    ad = h.result(); ad, nb = ship(ad, getattr(self.det, "_base", None)); self.down += nb; self.pending["ad"] = ad; self._finish_pending(t)
                else: self.state = "offline" if st == "queued" else "adapting"
        if (t + 1) % win == 0: self._window_end(t)
        elif self.edge_sig is not None and self.fast_n and self.ref_conf is not None and (t + 1) % self.fast_n == 0: self._fast_check(t)
        if self.edge_ready is not None and t >= self.edge_ready[0]:
            _, i_, sg_ = self.edge_ready; self.edge_ready = None; self._edge_commit(t, i_, sg_); self.conf_hist.clear()
        if self.want is not None and self.pending is None and sum(p["t"] > self.want[1] for p in self.pool) >= self.min_post:
            i_, w_ = self.want; self.want = None; self._submit(t, i_, do_edge=False, since=w_)
        if self.confirm_at is not None and t >= self.confirm_at[1] + 10 and len(self.win_frames) >= 6: self._confirm(t)
        self.t += 1

    def _submit(self, t, env_id, do_edge=True, since=None):
        """선별 -> 업링크 -> 서버 요청. 반환 후 state 는 adapting."""
        win, pol = self.win, self.policy
        if self.edge_fn is not None and do_edge:                                  # 엣지 즉시 단계: 최근 윈도우 프레임으로 라벨 없는 통계 정렬 어댑터를 만들어 바로 적용 (서버 왕복·학습 없음, 전송 0)
            fr = self.win_frames[-win:]; fr = fr[::max(1, len(fr) // 12)][:12]; ad_e = self.edge_fn(fr)
            self.prev_adapter = self.cur_adapter; _apply(self.det, ad_e); self.cur_adapter = ad_e; self.cur_env = env_id; self.last_change = t
            if self.use_registry and env_id < len(self.reg.adapters): self.reg.adapters[env_id] = ad_e
            self.events.append(("edge", t, env_id)); self.state = "recovered"; self.conf_hist.append((t, float(np.mean([f[1] for f in self.fbuf[-win:]]))))
            if self.server is None: return
        P = self.pool
        if since is not None:                                                    # 엣지 추적 모드: 환경이 바뀐 시점 이후의 프레임만 서버 학습 후보로 (바뀌기 전 프레임이 섞이면 어댑터가 나빠진다)
            P2 = [p for p in P if p["t"] > since]
            if len(P2) >= self.budget: P = P2
        if pol.kind == "drift" and pol.cp and pol.first_hit is not None and since is None:
            P2 = [p for p in P if p["t"] > pol.first_hit - win]               # 처음 판정된 윈도우의 시작 이후 프레임만
            if len(P2) >= self.budget: P = P2
        cands = {"uncert": np.array([p["u"] for p in P]), "emb": np.stack([p["e"] for p in P]), "ent": np.array([p["ent"] for p in P]), "age": np.array([t - p["t"] for p in P], float)}
        _sel = self.sel(cands, self.budget, self.rng); self.train_t = {P[i]["t"] for i in _sel}
        frames, nb = wire([P[i]["img"] for i in _sel], self.train_q); self.up += nb
        out = self.server(frames, env_id); env_after = env_id if self.use_registry else 0
        if isinstance(out, dict):
            out, nb = ((out, self.server.last_bytes) if getattr(self.server, "last_bytes", None) is not None else ship(out, getattr(self.det, "_base", None), self.codec)); self.down += nb; self.pending = {"handle": None, "ready_t": t + self.apply_delay, "ad": out, "env": env_after}
        else: self.pending = {"handle": out, "ready_t": None, "ad": None, "env": env_after}
        self.events.append(("trigger", t, env_id, len(frames))); self.state = "adapting"

    def _edge_apply(self, t, sig, fr):
        if self.srv_all and self.pending is not None: self.events.append(("cancel", t)); self.pending = None; self.want = None      # 서버 결과를 기다리는 중에 환경이 또 바뀌면 지난 환경용 결과는 버린다
        pol = self.policy; i = self.reg.match(sig)
        if i is None: i = self.reg.add(sig)
        if self.reg.adapters[i] is None: self.reg.adapters[i] = self.edge_fn(fr)
        if self.verify_edge and self.lf_verify and i not in self.checked:        # 라벨 없는 검증(BoS 류 대용 지표): 같은 프레임에서 후보를 적용했을 때 평균 신뢰도가 가장 높은 쪽을 택한다. 전송 0, Teacher 불필요
            self.checked.add(i); pf = fr[:self.probe_n]; cands = [self.reg.adapters[i], {}, self.cur_adapter]; cm = []
            for ad in cands:
                _apply(self.det, ad); ds = list(map(self.det.infer, pf)); cm.append(float(np.mean([np.mean(d["scores"]) if len(d["scores"]) else 0.0 for d in ds])))
            _apply(self.det, self.cur_adapter); self.edge_frames += 2 * len(pf)
            best = 0 if cm[0] >= max(cm[1], cm[2]) - self.lf_margin else (1 if cm[1] >= cm[2] else 2)
            if best != 0: self.reg.adapters[i] = cands[best]; self.events.append(("reject_edge", t, i))
            self._agree = 1.0
            if self.fc_margin is not None and self.ref_fc is not None:                # 적용할 최선 후보의 일관성이 소스 기준보다 fc_margin 넘게 낮으면 서버 정련 요청(agree_thr 경로)
                _apply(self.det, cands[best]); fcv = self._flipcons(pf[:4]); _apply(self.det, self.cur_adapter)
                self._agree = 1.0 if fcv >= self.ref_fc - self.fc_margin else 0.0; self.events.append(("fc", t, round(fcv, 2), round(self.ref_fc, 2)))
            if self.srv_all and i != 0: self._agree = 0.0                                # 새 환경 서명마다 서버 정련을 한 번 요청 (헤드만 4비트 워밍업 서버라 다운링크가 거의 없고, 업링크는 적은 프레임으로 줄인다)
            self.events.append(("probe", t, i, [round(float(x), 2) for x in cm]))
        elif self.verify_edge and self.probe_fn is not None and i not in self.checked:        # 통계 정렬이 해로운 경우(내용 통계만 바뀐 환경 등)를 거른다: 같은 프레임을 Teacher 일치도로 평가 (EdgeSync-lite 의 프로브 재사용, 업링크 계산)
            self.checked.add(i); pf = fr[:self.probe_n]; sv, nb = wire(pf, self.probe_q, self.probe_scale); self.up += nb; cands = [self.reg.adapters[i], {}, self.cur_adapter]
            cm = []
            def score(ad):
                _apply(self.det, ad); ds = list(map(self.det.infer, pf)); cm.append(float(np.mean([np.mean(d["scores"]) if len(d["scores"]) else 0.0 for d in ds])))
                return self.probe_fn(sv, [(d["boxes"], d["scores"], d["cls"]) for d in ds])
            sc = [score(a_) for a_ in cands]; _apply(self.det, self.cur_adapter)
            best = 0 if sc[0] >= max(sc[1], sc[2]) + 0.02 else (1 if sc[1] >= sc[2] else 2)
            if best != 0 and self.conf_tie and sc[0] >= max(sc[1], sc[2]) - 0.02 and cm[0] >= max(cm[1], cm[2]) + self.conf_tie: best = 0   # 일치도 동률(재현율만 보는 측정은 오검출 증가를 못 본다)이면 라벨 없는 평균 신뢰도가 확실히 높은 후보를 채택
            if best != 0: self.reg.adapters[i] = cands[best]; self.events.append(("reject_edge", t, i))
            self._agree = sc[best]; self.events.append(("probe", t, i, [round(float(x), 2) for x in sc]))
            if self.confirm and best == 0: self.confirm_at = (i, t)
        if self.edge_delay > 0: self.edge_ready = (t + self.edge_delay, i, sig); self.last_sig = sig; self.last_change = t      # 엣지 계산 시간(순전파·접기)을 모사: 결정 후 edge_delay 프레임 뒤에 적용
        else: self._edge_commit(t, i, sig)
        self.conf_hist.clear()                                                   # 이전 서버 적용 기록이 남아 있으면 정당한 환경 변화 때 엉뚱한 어댑터로 롤백된다 (시드 66 에서 확인)
        self.events.append(("edge", t, i)); self.state = "recovered"; pol.hits = 0; pol.last = t
        if self.agree_thr is not None and self.server is not None and self.pending is None and i not in self.refined and getattr(self, "_agree", 1.0) < self.agree_thr:   # 검증된 최선 후보의 Teacher 일치도가 아직 낮으면(통계 정렬로 부족한 국소 변화 등) 그때만 서버 정련 요청
            self.refined.add(i); self._agree = 1.0
            if self.min_post: self.want = (i, t - self.fast_n)
            else: self._submit(t, i, do_edge=False)

    def _edge_commit(self, t, i, sig):
        self.prev_adapter = self.cur_adapter; _apply(self.det, self.reg.adapters[i]); self.cur_adapter = self.reg.adapters[i]; self.cur_env = i; self.last_sig = sig; self.last_change = t

    def _confirm(self, t):
        """히스테리시스: 후보를 적용한 뒤 새로 들어온 프레임 6장으로 (적용 중인 어댑터 vs 어댑터 없음)을 다시 비교해, 어댑터 없음이 같거나 낫다면 되돌리고 그 서명은 어댑터 없음으로 확정한다. 처음 6장 판정의 우연을 걸러낸다(적응 불필요 환경의 해악 방지)."""
        i, _ = self.confirm_at; self.confirm_at = None; pf = self.win_frames[-6:]; self.up += sum(jpeg_bytes(f) for f in pf)
        def score(ad):
            _apply(self.det, ad); return self.probe_fn(pf, [(d["boxes"], d["scores"], d["cls"]) for d in map(self.det.infer, pf)])
        cur = self.cur_adapter; sc_cur, sc_none = score(cur), score({})
        if sc_none >= sc_cur: self.reg.adapters[i] = {}; self.cur_adapter = {}; self.refined.add(i); self.events.append(("revert", t, i)); _apply(self.det, {})
        else: _apply(self.det, cur)

    def _fast_check(self, t):
        """윈도우 중간의 빠른 점검: 최근 fast_n 프레임의 평균 신뢰도가 기준보다 fast_margin 넘게 낮을 때만 서명을 계산한다 (신뢰도 모니터는 추가 연산이 없고, 서명은 그때만 몇 장으로 계산).
        서명이 바뀌지 않았으면 한 윈도우 동안 다시 점검하지 않는다."""
        if t < self.fast_block or len(self.win_frames) < self.fast_n or (self.pending is not None and not self.srv_all): return
        if float(np.mean([f[1] for f in self.fbuf[-self.fast_n:]])) >= self.ref_conf - self.fast_margin: return
        fr = self.win_frames[-self.fast_n::2]; self.edge_frames += len(fr); sig = self.edge_sig(fr)
        if float(np.linalg.norm(sig - self.last_sig)) > self.sig_thr: self._edge_apply(t, sig, fr)
        else: self.fast_block = t + self.win

    def _edge_track(self, t, cmean, trig):
        """엣지 추적 모드: 윈도우마다 채널 통계 서명(StatAligner.signature)을 구해 (1) 서명이 바뀌었거나 드리프트 판정이 나면 레지스트리에서 같은 환경을 찾아 그 어댑터를,
        없으면 라벨 없는 통계 정렬 어댑터(edge_fn)를 즉시 적용한다. (2) 적용 후 온전한 윈도우의 평균 신뢰도가 기준(첫 윈도우=소스 환경)보다 gate_margin 넘게 낮을 때만 서버 정련을 요청한다."""
        win, pol = self.win, self.policy; fr = self.win_frames[-win:]; fr = fr[::max(1, len(fr) // self.edge_n)][:self.edge_n]; self.edge_frames += len(fr); sig = self.edge_sig(fr)
        if self.ref_conf is None:
            self.ref_conf = cmean; self.last_sig = sig; i = self.reg.add(sig)
            if self.fc_margin is not None: self.ref_fc = self._flipcons(fr[:6]); self.reg.adapters[i] = {}; self.cur_env = i; return
        changed = float(np.linalg.norm(sig - self.last_sig)) > self.sig_thr; drift = trig and self.pending is None and t - self.last_change >= 2 * win
        if (changed or drift) and (self.pending is None or self.srv_all): self._edge_apply(t, sig, fr)
        elif self.server is not None and self.pending is None and t - win + 1 > self.last_change and self.cur_env not in self.refined and self.cur_env != 0 and cmean < self.ref_conf - self.gate_margin:
            self._submit(t, self.cur_env, do_edge=False, since=(self.last_change if self.gate_since else None))

    def _window_end(self, t):
        win, pol = self.win, self.policy
        self.wenv.append(self._envs[t - win + 1]); self.wff.append(np.stack(self.fbuf[-win:]))
        wf = window_features(np.stack(self.fbuf[-win:]), pol.cols)
        if pol.kind == "ams":
            th = [cv2.resize(f, (32, 20)).astype(np.float32) for f in self.win_frames[::5]]
            wf = np.array([np.mean([np.abs(a - b).mean() / 255.0 for a, b in zip(th[:-1], th[1:])])])
        if pol.kind == "edgesync":                                              # 프로브: 최근 윈도우에서 몇 장을 서버 Teacher 에 보내 일치도를 받는다 (업링크로 계산)
            ids = self.rng.choice(len(self.win_frames), min(pol.probe_n, len(self.win_frames)), replace=False)
            fr, nb = wire([self.win_frames[i] for i in ids], self.probe_q, self.probe_scale); self.up += nb
            wf = np.array([self.probe_fn(fr, [self.preds[t - win + 1 + i] for i in ids])])
        cmean = float(np.mean([f[1] for f in self.fbuf[-win:]]))
        es = env_signature(self.win_frames[-win:]) if (self.use_registry and self.edge_sig is None) else None
        if self.use_registry and self.edge_sig is None and not self.reg.sigs:                              # 시작 환경을 '기본 가중치'로 등록
            self.reg.add(es); self.reg.adapters[0] = {}; self.cur_env = 0
        known = self.reg.match(es, cmean) if (self.use_registry and self.edge_sig is None) else None
        verify = False                                                           # 재사용·적용한 어댑터의 라벨 없는 검증: 전환/적용 후 첫 온전한 윈도우의 신뢰도를 기대값으로 저장, 이후 그보다 verify_margin 넘게 낮으면 '다른 환경'으로 보고 새로 적응
        if self.verify_margin is not None and self.edge_sig is None and self.use_registry and self.cur_env is not None and self.pending is None and t - win + 1 > self.last_change:
            if self.cur_env not in self.reg.exp: self.reg.exp[self.cur_env] = cmean
            elif cmean < self.reg.exp[self.cur_env] - self.verify_margin: verify = True
        if self.conf_hist and t - self.conf_hist[-1][0] < 2 * win and cmean < self.conf_hist[-1][1] - self.rollback_margin and self.prev_adapter is not None:
            _apply(self.det, self.prev_adapter); self.cur_adapter = self.prev_adapter; self.events.append(("rollback", t)); self.conf_hist.clear(); self.state = "rollback"
        trig = pol.should_trigger(t, wf)
        if self.edge_sig is not None:
            self._edge_track(t, cmean, trig); self.win_frames = []; return
        if verify and not trig: trig = True; known = None; pol.hits = max(pol.hits, 1)
        if pol.kind == "drift" and pol.hits > 0 and self.pending is None and self.state in ("normal", "recovered", "switched", "rollback"): self.state = "drift"
        if self.use_registry and known is not None and known != self.cur_env and self.reg.adapters[known] is not None and self.pending is None:
            self.prev_adapter = self.cur_adapter                                 # 이미 본 환경으로 복귀: 재학습 없이 전환
            _apply(self.det, self.reg.adapters[known]); self.cur_adapter = self.reg.adapters[known]; self.cur_env = known
            self.last_change = t; self.events.append(("switch", t, known)); pol.hits = 0; pol.last = t - pol.cooldown + win; self.state = "switched"      # 재사용한 어댑터가 실제로 맞는지 다음 윈도우에서 다시 판정한다 (유사 환경 오인 시 새로 적응). 쿨다운은 한 윈도우만 적용
        elif trig and self.pending is None:
            env_id = (known if known is not None else self.reg.add(es)) if self.use_registry else len(self.events)
            self._submit(t, env_id)
        elif self.state == "normal" or self.state == "static": pass
        self.win_frames = []

    def result(self):
        gts = self.gts if self.has_gt else None
        return {"preds": self.preds, "gts": self.gts, "events": self.events, "up_bytes": self.up, "down_bytes": self.down,
                "win_map": windowed_map(self.preds, self.gts, self.n_cls, self.win) if self.has_gt else np.array([]),
                "win_frame_feats": np.stack(self.wff) if self.wff else None, "win_env": self.wenv, "lat_ms": float(np.mean(self.lat) * 1000), "edge_extra": self.edge_frames / max(len(self.preds), 1)}


def run_stream(stream, detector, policy, server, selector="hybrid", budget=16, win=50, pool_size=100, apply_delay=50, rollback_margin=0.05,
               use_registry=True, seed=0, n_cls=6, env_thr=0.08, probe_fn=None, edge_fn=None, edge_sig=None, sig_thr=0.25, gate_margin=0.08, fast_n=0, fast_margin=0.1, verify_edge=False, agree_thr=None, confirm=False, min_post=0, probe_n=6, probe_q=80, probe_scale=1.0, train_q=80, codec="auto", heldout=False, edge_delay=0, conf_tie=0.0, **extra):
    """stream: 항목마다 item["img"], item["gt"]=(boxes,cls), item["env"].  server(frames_bgr, env_id) -> adapter dict."""
    r = StreamRunner(detector, policy, server, selector, budget, win, pool_size, apply_delay, rollback_margin, use_registry, seed, n_cls, env_thr, probe_fn, edge_fn=edge_fn, edge_sig=edge_sig, sig_thr=sig_thr, gate_margin=gate_margin, fast_n=fast_n, fast_margin=fast_margin, verify_edge=verify_edge, agree_thr=agree_thr, confirm=confirm, min_post=min_post, probe_n=probe_n, probe_q=probe_q, probe_scale=probe_scale, train_q=train_q, codec=codec, heldout=heldout, edge_delay=edge_delay, conf_tie=conf_tie, **extra)
    for item in stream: r.step(item["img"], item["gt"], item["env"])
    return r.result()

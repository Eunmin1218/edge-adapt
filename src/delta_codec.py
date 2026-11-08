"""어댑터 전송 부호화. 서버·엣지는 모두 같은 base ONNX 의 initializer 를 가지고 있으므로 '새 값' 대신 '변화량(delta = 새 값 - base)'만 보내면 된다.
변화량은 값의 범위가 작아 같은 비트 수로 훨씬 정밀하게 표현된다 (FP16 으로 새 값을 보내면 가중치 크기에 비례한 오차가 생긴다).
  fp16   : 새 값을 FP16 으로 (기존 방식, 기준선)
  d8     : delta 를 채널별(0번 축) 대칭 int8 로 양자화
  auto   : (권장) 텐서마다 가장 작은 정확한 표현을 고른다.
             - 합성곱 가중치가 base 의 채널별 상수배(= affine 어댑터를 접은 결과)이면 채널별 (s-1) 만 FP16 으로, 짝이 되는 편향은 t = b_new - s*b_base 만 FP16 으로 보낸다
             - 그 밖의 텐서는 d8 (채널별 int8 변화량)
             - 이름이 "__" 로 시작하는 텐서(입력 톤 커브)는 FP16 그대로
  sparse : 텐서별 |delta| 상위 keep 비율만 남기고(나머지 0) 남긴 값은 int8 로, 위치는 앞 위치와의 차이로 저장 (ComPEFT·Deep Gradient Compression 계열의 희소화)
모든 방식은 np.savez_compressed 로 담아, len(raw) 가 실제 전송 바이트다. decode 결과는 엣지가 apply_adapter 에 넣을 '새 값' dict 이다."""
import io, struct, zlib, numpy as np


def _q8(d):
    ax = tuple(range(1, d.ndim)) if d.ndim >= 2 else None
    m = np.abs(d).max(axis=ax, keepdims=True) if ax else np.abs(d).max(keepdims=True)
    sc = np.where(m > 0, m / 127.0, 1.0).astype(np.float32)
    return np.clip(np.rint(d / sc), -127, 127).astype(np.int8), sc


def _affine_fit(wn, wb, tol=2e-3):
    """wn ≈ s[:,None,None,None] * wb 인 채널별 s 를 찾는다. 오차가 tol(상대) 이내면 s, 아니면 None."""
    ax = tuple(range(1, wb.ndim)); den = (wb * wb).sum(axis=ax)
    if wb.ndim < 2 or np.any(den <= 0): return None
    sc = (wn * wb).sum(axis=ax) / den; res = np.linalg.norm(wn - sc.reshape((-1,) + (1,) * (wb.ndim - 1)) * wb); nrm = np.linalg.norm(wn - wb)
    return sc.astype(np.float32) if res <= tol * max(np.linalg.norm(wn), 1e-12) and nrm > 0 else None


def _q8t(x, bits=8):
    """텐서 전체에 하나의 배율을 쓰는 대칭 정수(기본 int8, bits<8 이면 값 범위만 좁히고 zlib 이 엔트로피를 줄인다). 반환: (int8 배열, FP16 배율, 복원값)"""
    L = 2 ** (bits - 1) - 1; m = float(np.abs(x).max()); sc = np.float16(m / L if m > 0 else 1.0); q = np.clip(np.rint(x / np.float32(sc)), -L, L).astype(np.int8); return q, sc, q.astype(np.float32) * np.float32(sc)


def _encode_auto(ad, base, out, q8aff=False):
    done = set()
    for k, v in ad.items():
        v = np.asarray(v, np.float32)
        if k.startswith("__"): out[k + "|r"] = v.astype(np.float16); done.add(k); continue
        if k.endswith(".weight") and v.ndim == 4 and k in base:
            sc = _affine_fit(v, np.asarray(base[k], np.float32))
            if sc is not None:
                bn = k[:-len("weight")] + "bias"
                if bn in base:
                    bb = np.asarray(base[bn], np.float32); bnew = np.asarray(ad.get(bn, bb), np.float32)
                    if q8aff:                                                                              # auto8: (s-1) 과 t 를 텐서별 int8 로 (FP16 의 절반)
                        qa, sa, ds = _q8t(sc - 1); out[k + "|a8"], out[k + "|as"] = qa, sa
                        qt, st, _ = _q8t(bnew - (1 + ds) * bb); out[bn + "|t8"], out[bn + "|ts"] = qt, st
                        done.update((k, bn)); continue
                    s16 = (sc - 1).astype(np.float16); out[k + "|a"] = s16
                    out[bn + "|t"] = (bnew - (1 + s16.astype(np.float32)) * bb).astype(np.float16)       # 디코더가 복원하는 s(FP16 반올림 포함)에 맞춰 t 를 계산
                    done.update((k, bn)); continue
    for k, v in ad.items():
        if k in done: continue
        v = np.asarray(v, np.float32); d = v - np.asarray(base[k], np.float32); q, sc = _q8(d); out[k + "|q"], out[k + "|s"] = q, sc.astype(np.float16)


# ---- pack / pack16: 단일 바이너리 + zlib. np.savez 는 텐서마다 zip 항목(약 130B 머리)이 붙어 텐서 수백 개면 머리만 수십 KB 가 된다 (162 텐서 어댑터: 내용 15KB 인데 파일 70~92KB).
#      기준 initializer 이름은 양쪽이 같으므로 이름 대신 정렬된 base 키의 번호(uint16)만 보내고, 모양도 base 에서 얻는다.
_MAGIC = 0xA7


def _pack(ad, base, q8aff, bits=8):
    keys = sorted(base.keys()); ix = {k: i for i, k in enumerate(keys)}; items, done = [], set()
    for k, v in ad.items():
        v = np.asarray(v, np.float32)
        if k.startswith("__"):
            nb = k.encode(); items.append(struct.pack("<BB", 2, len(nb)) + nb + struct.pack("<B", v.ndim) + struct.pack("<%dI" % v.ndim, *v.shape) + v.astype(np.float16).tobytes()); done.add(k); continue
        if k.endswith(".weight") and v.ndim == 4 and k in base:
            sc = _affine_fit(v, np.asarray(base[k], np.float32)); bn = k[:-len("weight")] + "bias"
            if sc is not None and bn in base:
                bb = np.asarray(base[bn], np.float32); bnew = np.asarray(ad.get(bn, bb), np.float32)
                if q8aff:
                    qa, sa, ds = _q8t(sc - 1, bits); qt, st, _ = _q8t(bnew - (1 + ds) * bb, bits)
                    items.append(struct.pack("<BH", 0, ix[k]) + sa.tobytes() + qa.tobytes() + st.tobytes() + qt.tobytes())
                else:
                    s16 = (sc - 1).astype(np.float16); t16 = (bnew - (1 + s16.astype(np.float32)) * bb).astype(np.float16)
                    items.append(struct.pack("<BH", 3, ix[k]) + s16.tobytes() + t16.tobytes())
                done.update((k, bn))
    for k, v in ad.items():
        if k in done: continue
        d = np.asarray(v, np.float32) - np.asarray(base[k], np.float32); q, sc = _q8(d); items.append(struct.pack("<BH", 1, ix[k]) + sc.astype(np.float16).ravel().tobytes() + q.tobytes())
    return bytes([_MAGIC, 1 if q8aff else 0]) + zlib.compress(struct.pack("<I", len(items)) + b"".join(items), 9)


def _unpack(raw, base):
    keys = sorted(base.keys()); body = zlib.decompress(raw[2:]); n = struct.unpack_from("<I", body, 0)[0]; o = 4; ad = {}
    def take(dt, cnt):
        nonlocal o
        a = np.frombuffer(body, dtype=dt, count=cnt, offset=o); o += a.nbytes; return a
    for _ in range(n):
        kind = body[o]; o += 1
        if kind == 2:
            ln = body[o]; o += 1; name = body[o:o + ln].decode(); o += ln; nd = body[o]; o += 1; shp = struct.unpack_from("<%dI" % nd, body, o); o += 4 * nd
            ad[name] = take(np.float16, int(np.prod(shp))).astype(np.float32).reshape(shp); continue
        k = keys[struct.unpack_from("<H", body, o)[0]]; o += 2; wb = np.asarray(base[k], np.float32)
        if kind in (0, 3):
            C = wb.shape[0]; bn = k[:-len("weight")] + "bias"; bb = np.asarray(base[bn], np.float32)
            if kind == 0:
                sa = np.float32(take(np.float16, 1)[0]); ds = take(np.int8, C).astype(np.float32) * sa; st = np.float32(take(np.float16, 1)[0]); t = take(np.int8, C).astype(np.float32) * st
            else: ds = take(np.float16, C).astype(np.float32); t = take(np.float16, C).astype(np.float32)
            s1 = 1 + ds; ad[k] = wb * s1.reshape((-1,) + (1,) * (wb.ndim - 1)); ad[bn] = s1 * bb + t
        else:
            ns = wb.shape[0] if wb.ndim >= 2 else 1; sc = take(np.float16, ns).astype(np.float32); q = take(np.int8, wb.size).astype(np.float32).reshape(wb.shape)
            ad[k] = wb + q * (sc.reshape((-1,) + (1,) * (wb.ndim - 1)) if wb.ndim >= 2 else sc)
    return ad


def encode(ad, base, mode="fp16", keep=0.1):
    if mode in ("pack", "pack16", "pack4"): return _pack(ad, base, mode != "pack16", 4 if mode == "pack4" else 8)
    out = {"_mode": np.array(mode)}
    if mode in ("auto", "auto8"): _encode_auto(ad, base, out, q8aff=(mode == "auto8")); b = io.BytesIO(); np.savez_compressed(b, **out); return b.getvalue()
    for k, v in ad.items():
        v = np.asarray(v, np.float32)
        if mode == "fp16": out[k] = v.astype(np.float16); continue
        d = v - np.asarray(base[k], np.float32)
        if mode == "d8":
            q, sc = _q8(d); out[k + "|q"], out[k + "|s"] = q, sc.astype(np.float16)
        elif mode == "sparse":
            f = d.ravel(); n = max(1, int(round(keep * f.size))); idx = np.sort(np.argpartition(np.abs(f), -n)[-n:]) if n < f.size else np.arange(f.size)
            q, sc = _q8(f[idx]); gaps = np.diff(idx, prepend=-1)
            out[k + "|g"] = gaps.astype(np.uint8) if gaps.max() < 256 else gaps.astype(np.uint32); out[k + "|q"], out[k + "|s"], out[k + "|n"] = q, sc.astype(np.float16), np.array(v.shape)
        else: raise ValueError(mode)
    b = io.BytesIO(); np.savez_compressed(b, **out); return b.getvalue()


def decode(raw, base):
    if raw[0] == _MAGIC: return _unpack(raw, base)
    z = np.load(io.BytesIO(raw)); mode = str(z["_mode"]); ad = {}
    if mode == "fp16":
        return {k: z[k].astype(np.float32) for k in z.files if k != "_mode"}
    if mode in ("auto", "auto8"):
        for f in z.files:
            if f.endswith("|a8"):
                k = f[:-3]; wb = np.asarray(base[k], np.float32); s1 = 1 + z[f].astype(np.float32) * np.float32(z[k + "|as"]); ad[k] = wb * s1.reshape((-1,) + (1,) * (wb.ndim - 1))
                bn = k[:-len("weight")] + "bias"; ad[bn] = s1 * np.asarray(base[bn], np.float32) + z[bn + "|t8"].astype(np.float32) * np.float32(z[bn + "|ts"])
            elif f.endswith("|r"): ad[f[:-2]] = z[f].astype(np.float32)
            elif f.endswith("|a"):
                k = f[:-2]; wb = np.asarray(base[k], np.float32); s1 = 1 + z[f].astype(np.float32); ad[k] = wb * s1.reshape((-1,) + (1,) * (wb.ndim - 1))
                bn = k[:-len("weight")] + "bias"; ad[bn] = s1 * np.asarray(base[bn], np.float32) + z[bn + "|t"].astype(np.float32)
            elif f.endswith("|q"):
                k = f[:-2]; ad[k] = np.asarray(base[k], np.float32) + z[k + "|q"].astype(np.float32) * z[k + "|s"].astype(np.float32)
        return ad
    for k in [f[:-2] for f in z.files if f.endswith("|q")]:
        b = np.asarray(base[k], np.float32)
        if mode == "d8": d = z[k + "|q"].astype(np.float32) * z[k + "|s"].astype(np.float32)
        else:
            d = np.zeros(b.size, np.float32); idx = np.cumsum(z[k + "|g"].astype(np.int64)) - 1; d[idx] = z[k + "|q"].astype(np.float32) * float(z[k + "|s"].reshape(-1)[0])
            d = d.reshape(tuple(z[k + "|n"]))
        ad[k] = b + d
    return ad


def scale_adapter(ad, base, alpha):
    """가중치 보간(WiSE-FT): base + alpha*(adapter - base). alpha<1 은 적응을 약하게(과적합 완화), alpha>1 은 외삽. 톤 커브는 항등과 보간한다."""
    out = {}
    for k, v in ad.items():
        if k == "__tone__":
            idt = np.tile(np.linspace(0, 1, np.asarray(v).shape[1], dtype=np.float32), (3, 1)); out[k] = idt + alpha * (np.asarray(v, np.float32) - idt)
        else: b = np.asarray(base[k], np.float32); out[k] = b + alpha * (np.asarray(v, np.float32) - b)
    return out

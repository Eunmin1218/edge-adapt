"""데모 화면 렌더러 (순수 함수: 입력 데이터 → BGR 이미지). 화면 구성:
  상단  : 제목 + 현재 환경(정답) 배지
  중단  : 시스템별 패널(영상+탐지 박스 / 상태 칩 / 지표 줄)
  하단  : 윈도우별 성능 곡선(시스템별 색, 환경 구간을 배경 띠로 표시, 적응 요청 ▲ / 적용 ● 표시)
한글은 cv2.putText 로 그릴 수 없으므로 PIL 로 그린다. 한글 폰트를 못 찾으면 영문 라벨로 자동 대체한다."""
import os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

BG, FG, DIM = (28, 24, 24), (240, 240, 240), (150, 150, 150)           # BGR
FONTS = ["C:/Windows/Fonts/malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
         "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]
STATE = {  # 상태 키 → (한글, 영문, 칩 색 BGR)
    "static": ("적응 없음", "NO ADAPTATION", (110, 110, 110)), "normal": ("정상", "NORMAL", (80, 170, 80)), "drift": ("성능 저하 감지!", "DRIFT DETECTED", (30, 140, 240)),
    "adapting": ("서버 적응 중...", "ADAPTING...", (220, 140, 40)), "offline": ("네트워크 단절·큐 보관", "OFFLINE (QUEUED)", (60, 60, 210)), "recovered": ("회복됨", "RECOVERED", (60, 200, 90)),
    "switched": ("저장 어댑터 전환", "ADAPTER SWITCHED", (180, 190, 40)), "rollback": ("롤백", "ROLLBACK", (200, 80, 160)), "error": ("서버 오류", "SERVER ERROR", (40, 40, 200))}
ENV_COLOR = {"clear": (70, 90, 70), "rain": (130, 90, 40), "night": (90, 40, 40), "fog": (110, 110, 110)}
ENV_KO = {"clear": "맑음(낮)", "rain": "비", "night": "밤", "fog": "안개", "glare": "야간 눈부심", "heavy_rain": "폭우", "clear_sparse": "맑음(차량 적음)", "clear_dense": "맑음(차량 많음)", "night_rain": "밤+비"}
_cache = {}


def get_font(size):
    if size in _cache: return _cache[size]
    f = None
    for p in ([os.environ["EDGE_ADAPT_FONT"]] if os.environ.get("EDGE_ADAPT_FONT") else []) + FONTS:
        if os.path.exists(p):
            try: f = ImageFont.truetype(p, size); break
            except OSError: pass
    _cache[size] = f; return f


def korean_ok(): return get_font(12) is not None


def _txt(draw, xy, s, size, color=FG, anchor="la"):
    f = get_font(size) or ImageFont.load_default()
    draw.text(xy, s, font=f, fill=(color[2], color[1], color[0]), anchor=anchor)


def render(panels, curves, bands, n_win, title, env_label, markers=(), pw=416, ph=260, ylabel="mAP50", ko=None):
    """panels: [dict(name, color(BGR), frame(BGR 400x640), state, lines[list[str]])]
    curves: [(이름, 색, [윈도우별 값(NaN 허용)])]   bands: [(시작윈도우, 끝윈도우, 환경)]   markers: [(윈도우, 색, 'trigger'|'apply')]"""
    ko = korean_ok() if ko is None else ko
    gap, top, hdr, foot, ch = 12, 56, 34, 78, 250
    n = len(panels); W = n * pw + (n + 1) * gap; H = top + hdr + ph + foot + ch + 3 * gap
    cv = np.full((H, W, 3), BG, np.uint8); chips = []
    for i, p in enumerate(panels):
        x0 = gap + i * (pw + gap); y0 = top
        cv2.rectangle(cv, (x0, y0), (x0 + pw, y0 + hdr), tuple(int(c * 0.45) for c in p["color"]), -1)
        cv[y0 + hdr:y0 + hdr + ph, x0:x0 + pw] = cv2.resize(p["frame"], (pw, ph), interpolation=cv2.INTER_AREA)
        k, e, col = STATE[p["state"]]; cv2.rectangle(cv, (x0, y0 + hdr + ph - 30), (x0 + pw, y0 + hdr + ph), col, -1)
        cv2.rectangle(cv, (x0, y0), (x0 + pw, y0 + hdr + ph), p["color"], 2); chips.append((x0, y0, k if ko else e))
    cx0, cy0, cx1, cy1 = gap + 46, top + hdr + ph + foot + 2 * gap, W - gap - 10, top + hdr + ph + foot + 2 * gap + ch - 28
    cv2.rectangle(cv, (cx0, cy0), (cx1, cy1), (60, 56, 56), 1); nw = max(n_win, 1); X = lambda w: int(cx0 + (cx1 - cx0) * w / nw); Y = lambda v: int(cy1 - (cy1 - cy0) * float(np.clip(v, 0, 1)))
    for s, e, env in bands:
        ov = cv[cy0 + 1:cy1, X(s):X(e)]; cv[cy0 + 1:cy1, X(s):X(e)] = (ov * 0.5 + np.array(ENV_COLOR.get(env, (80, 80, 80))) * 0.5).astype(np.uint8)
    for v in (0.25, 0.5, 0.75): cv2.line(cv, (cx0, Y(v)), (cx1, Y(v)), (60, 56, 56), 1)
    for name, col, ys in curves:
        pts = [(X(i + 0.5), Y(v)) for i, v in enumerate(ys) if not np.isnan(v)]
        for a, b in zip(pts[:-1], pts[1:]): cv2.line(cv, a, b, col, 3, cv2.LINE_AA)
        if pts: cv2.circle(cv, pts[-1], 5, col, -1, cv2.LINE_AA)
    for w, col, kind in markers:
        x = X(w + 1); y = cy0 + 6 if kind == "trigger" else cy0 + 18
        (cv2.drawMarker(cv, (x, y), col, cv2.MARKER_TRIANGLE_DOWN, 10, 2) if kind == "trigger" else cv2.circle(cv, (x, y), 4, col, -1))
    im = Image.fromarray(cv[:, :, ::-1]); d = ImageDraw.Draw(im)
    _txt(d, (gap, 10), title, 24, FG); _txt(d, (W - gap, 10), env_label, 26, (60, 220, 255), "ra")
    for p, (x0, y0, chip) in zip(panels, chips):
        _txt(d, (x0 + 10, y0 + 6), p["name"], 18, FG); _txt(d, (x0 + 10, y0 + hdr + ph - 27), chip, 18, (255, 255, 255))
        for j, l in enumerate(p["lines"]): _txt(d, (x0 + 4, y0 + hdr + ph + 6 + 22 * j), l, 15, FG if j == 0 else DIM)
    _txt(d, (cx0, cy0 - 20), ylabel, 14, DIM); _txt(d, (cx0 - 6, cy1 - 8), "0", 12, DIM, "ra"); _txt(d, (cx0 - 6, cy0 + 2), "1", 12, DIM, "ra")
    lx = cx0 + 8
    for name, col, _ in curves:
        d.rectangle([lx, cy1 + 8, lx + 14, cy1 + 18], fill=(col[2], col[1], col[0])); _txt(d, (lx + 20, cy1 + 4), name, 15, FG); lx += 40 + 9 * len(name)
    _txt(d, (cx1, cy1 + 4), ("▲ 적응 요청   ● 어댑터 적용   (배경: 환경)" if ko else "v request   o applied   (background: environment)"), 14, DIM, "ra")
    _txt(d, (cx0, cy1 + 26), f"window (1 window = 50 frames)  {int(np.sum(~np.isnan(curves[0][2]))) if curves else 0}/{n_win}", 12, DIM)
    return np.asarray(im)[:, :, ::-1].copy()

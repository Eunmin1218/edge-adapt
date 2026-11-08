"""엣지 쪽 통신. 네트워크가 끊기면 선별 프레임을 큐에 보관하고, 재연결 시 이어서 전송한다."""
import time, collections
import cv2, requests
from .api import unpack


class EdgeClient:
    def __init__(self, base, timeout=5.0, tag="", base_inits=None):
        self.base, self.timeout, self.tag, self.base_inits = base.rstrip("/"), timeout, tag, base_inits; self.queue = collections.deque(); self.up_bytes = self.down_bytes = 0

    def alive(self):
        try: return requests.get(self.base + "/v1/health", timeout=self.timeout).ok
        except requests.RequestException: return False

    def submit(self, frames_bgr, env):
        """프레임 업로드. 실패하면 큐에 넣고 None."""
        self.queue.append((frames_bgr, env)); return self.flush()

    def flush(self):
        job = None
        while self.queue:
            frames, env = self.queue[0]
            try:
                files = [("files", (f"{i}.jpg", cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), "image/jpeg")) for i, f in enumerate(frames)]
                r = requests.post(self.base + "/v1/frames", data={"env": env, "client": self.tag}, files=files, timeout=self.timeout); r.raise_for_status()
                self.up_bytes += sum(len(x[1][1]) for x in files); job = r.json()["job"]; self.queue.popleft()
            except requests.RequestException:
                return None                                     # 다음 flush()에서 재시도
        return job

    def poll(self, job, interval=0.5, max_wait=600):
        t0 = time.time()
        while time.time() - t0 < max_wait:
            try:
                s = requests.get(f"{self.base}/v1/jobs/{job}", timeout=self.timeout).json()
                if s["state"] != "running": return s
            except requests.RequestException: pass
            time.sleep(interval)
        return {"state": "timeout"}

    def download(self, name):
        r = requests.get(f"{self.base}/v1/adapters/{name}", timeout=self.timeout); r.raise_for_status()
        self.down_bytes += len(r.content); return unpack(r.content, self.base_inits)

    def probe(self, frames_bgr, dets):
        """EdgeSync-lite: 프레임 + 엣지 탐지를 보내 Teacher 일치도를 받는다. 끊기면 1.0(긴급도 0)으로 처리한다."""
        import json
        files = [("files", (f"{i}.jpg", cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), "image/jpeg")) for i, f in enumerate(frames_bgr)]
        d = json.dumps([{"boxes": b.tolist(), "scores": s.tolist(), "cls": c.tolist()} for b, s, c in dets])
        try:
            r = requests.post(self.base + "/v1/probe", data={"dets": d}, files=files, timeout=self.timeout); r.raise_for_status()
            self.up_bytes += sum(len(x[1][1]) for x in files); return float(r.json()["agreement"])
        except requests.RequestException: return 1.0

    def remote(self, frames_bgr, env):
        return RemoteJob(self, frames_bgr, env)


class RemoteJob:
    """StreamRunner 가 쓰는 비차단 핸들. status: queued(네트워크 단절, 큐 보관) | running | done | error.  ready() 는 호출마다 짧게 한 번만 조회한다."""
    def __init__(self, cli, frames, env):
        self.cli, self.job, self.status, self._ad = cli, cli.submit(frames, env), "running", None
        if self.job is None: self.status = "queued"

    def ready(self):
        if self.job is None:
            self.job = self.cli.flush()
            if self.job is None: return False
            self.status = "running"
        try:
            s = requests.get(f"{self.cli.base}/v1/jobs/{self.job}", timeout=0.3).json()
        except requests.RequestException: return False
        if s["state"] == "error": self.status = "error"; return False
        if s["state"] != "done": return False
        self._ad = self.cli.download(s["adapter"]); self.status = "done"; return True

    def result(self): return self._ad

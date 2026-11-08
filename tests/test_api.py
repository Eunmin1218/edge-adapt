import threading, time, socket
import numpy as np, uvicorn
from src.api import create_app
from src.edge_client import EdgeClient


def test_api_roundtrip_and_queue():
    got = {}

    def adapt(frames, env):
        got["n"] = len(frames); got["shape"] = frames[0].shape; time.sleep(0.3)
        return {"model.23.cv.bias": np.arange(10, dtype=np.float32), "model.23.cv.weight": np.ones((10, 3, 2, 2), np.float32)}
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    srv = uvicorn.Server(uvicorn.Config(create_app(adapt), host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    c = EdgeClient(f"http://127.0.0.1:{port}")
    for _ in range(50):
        if c.alive(): break
        time.sleep(0.1)
    frames = [np.random.default_rng(i).integers(0, 255, (400, 640, 3), dtype=np.uint8) for i in range(4)]
    job = c.submit(frames, env=3); assert job
    st = c.poll(job); assert st["state"] == "done", st
    ad = c.download(st["adapter"]); assert got["n"] == 4 and got["shape"] == (400, 640, 3)
    assert set(ad) == {"model.23.cv.bias", "model.23.cv.weight"} and np.allclose(ad["model.23.cv.bias"], np.arange(10))
    c2 = EdgeClient("http://127.0.0.1:9", timeout=0.5)               # 네트워크 단절: 큐에 보관
    assert c2.submit(frames, 0) is None and len(c2.queue) == 1
    srv.should_exit = True


def test_probe_and_remote_job_and_client_tag():
    seen = {}

    def adapt(frames, env): seen["env"] = env; return {"model.23.cv.bias": np.arange(10, dtype=np.float32)}
    def agree(frames, dets): seen["dets"] = dets; return 0.75
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    srv = uvicorn.Server(uvicorn.Config(create_app(adapt, agree), host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    c = EdgeClient(f"http://127.0.0.1:{port}", tag="ours")
    for _ in range(50):
        if c.alive(): break
        time.sleep(0.1)
    fr = [np.random.default_rng(i).integers(0, 255, (400, 640, 3), dtype=np.uint8) for i in range(2)]
    dets = [(np.array([[1., 2, 30, 40]]), np.array([0.9]), np.array([1])), (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))]
    assert abs(c.probe(fr, dets) - 0.75) < 1e-9 and seen["dets"][0][0].shape == (1, 4) and seen["dets"][1][0].shape == (0, 4)
    h = c.remote(fr, 2); assert h.status == "running"
    for _ in range(50):
        if h.ready(): break
        time.sleep(0.1)
    assert h.status == "done" and seen["env"] == "ours:2" and np.allclose(h.result()["model.23.cv.bias"], np.arange(10))
    off = EdgeClient("http://127.0.0.1:9", timeout=0.3).remote(fr, 0); assert off.status == "queued" and not off.ready()
    assert EdgeClient("http://127.0.0.1:9", timeout=0.3).probe(fr, dets) == 1.0
    srv.should_exit = True

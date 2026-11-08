import subprocess, sys, tarfile, tempfile, io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_extract_seqs():
    d = Path(tempfile.mkdtemp())
    with tarfile.open(d / "img.tar", "w") as tf:
        for seq in ["s1", "s2", "s3"]:
            for i in range(3):
                data = f"{seq}-{i}".encode(); ti = tarfile.TarInfo(f"img/{seq}/{i:08d}_img_front.jpg"); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
    (d / "want.txt").write_text("s1\ns3\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/extract_seqs.py"), "--tar", str(d / "img.tar"), "--seqs", str(d / "want.txt"), "--out", str(d / "out")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (d / "out").iterdir()) == ["s1", "s3"]
    assert (d / "out/s1/00000002_img_front.jpg").read_text() == "s1-2"


def test_extract_every():
    d = Path(tempfile.mkdtemp())
    with tarfile.open(d / "img.tar", "w") as tf:
        for i in range(0, 30):
            data = b"x"; ti = tarfile.TarInfo(f"s1/{i:08d}_img_front.jpg"); ti.size = 1; tf.addfile(ti, io.BytesIO(data))
    (d / "want.txt").write_text("s1\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/extract_seqs.py"), "--tar", str(d / "img.tar"), "--seqs", str(d / "want.txt"), "--out", str(d / "out"), "--every", "10"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (d / "out/s1").iterdir()) == ["00000000_img_front.jpg", "00000010_img_front.jpg", "00000020_img_front.jpg"]

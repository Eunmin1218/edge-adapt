import json, tempfile, os
from src.adapter_cfg import load_adapter_cfg


def test_defaults_and_override():
    d = load_adapter_cfg("/nonexistent/adapter.json"); assert d["epochs"] == 60 and d["mode"] == "affine" and d["alpha"] == 1.25 and d["train_layers"] == 0 and d["batch"] == 0 and d["trainer"] == "distill"
    p = os.path.join(tempfile.mkdtemp(), "a.json"); json.dump({"epochs": 60, "train_layers": 3, "lr": 0.01, "all": [1, 2], "selected_on": "x"}, open(p, "w"))
    c = load_adapter_cfg(p); assert c["epochs"] == 60 and c["train_layers"] == 3 and c["lr"] == 0.01 and c["tau"] == 0.5 and "all" not in c

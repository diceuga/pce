# util/config.py
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    path = os.path.join(BASE_DIR, "etc", "config.json")
    with open(path) as f:
        return json.load(f)

def load_bgpls_peers():
    cfg = load_config()
    return cfg.get("bgpls", {}).get("peers", [])

def load_bgpls():
    cfg = load_config()
    return cfg.get("bgpls", {})

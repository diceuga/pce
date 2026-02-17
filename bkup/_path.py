# util/node.py
import json
import os
import time
import threading

from .diff import diff_dict, DiffType

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_PATH = os.path.join(BASE_DIR, "etc", "path.json")

def load_config():
    mtime = os.path.getmtime(NODE_PATH)
    with open(NODE_PATH) as f:
        return json.load(f), mtime

def load_path():
    cfg, mtime = load_config()
    return cfg.get("path", {}), mtime

def get_path_mtime():
    return os.path.getmtime(NODE_PATH)

def path_config_watcher(ev_queue, last_nodes, last_mtime, interval=1.0):
    """
    node.json の変更を監視し、差分イベントを ev_queue に流す
    """
    while True:
            mtime = os.path.getmtime(NODE_PATH)
            if mtime != last_mtime:
                nodes, _ = load_path()

                diffs = diff_dict(last_nodes, nodes)
                for d in diffs:
                    ev_queue.put({
                        "type": "PATH_CONFIG",
                        "diff": d,
                    })

                last_nodes = nodes
                last_mtime = mtime

            time.sleep(interval)


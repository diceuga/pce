import json
import os
import time
import threading

from .diff      import diff_dict, DiffType
#from .loadclass import PathDef

from dataclasses import dataclass
from typing import List, Literal

Mode = Literal["ecmp", "k"]
PathType = Literal["p2p", "p2mp"]

# PathDef
@dataclass(frozen=True)
class PathDef:
  name: str
  type: PathType
  underlay: str
  src: List[str]
  dst: List[str]
  mode: Mode
  K: int
  delta: int
  bw: int
  pri:int
  optm:int

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_PATH   = os.path.join(BASE_DIR, "etc", "node.json")
CONST_PATH  = os.path.join(BASE_DIR, "etc", "const.json")
PATH_PATH   = os.path.join(BASE_DIR, "etc", "path.json")
CONFIG_PATH = os.path.join(BASE_DIR, "etc", "config.json")

NORMALIZED_P = {}

class ConfigManager:
  def __init__(self, log):
    self.log = log

    # DIR
    self.BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    self.NODE_PATH   = os.path.join(BASE_DIR, "etc", "node.json")
    self.CONST_PATH  = os.path.join(BASE_DIR, "etc", "const.json")
    self.PATH_PATH   = os.path.join(BASE_DIR, "etc", "path.json")
    self.CONFIG_PATH = os.path.join(BASE_DIR, "etc", "config.json")
    #print(str(BASE_DIR))

    # Q
    self.G_queue    = None
    self.C_queue    = None
    self.C_cnt      = None

    # info
    self.NODEINFO   = {}
    self.NODETIME   = 0
    self.PATHINFO   = {}
    self.N_PATHINFO = {}
    self.PATHTIME   = 0
    self.CONSTINFO   = {}
    self.CONSTTIME   = 0
    self.BGPLSINFO  = {}

    # init load
    self.NODEINFO,  self.NODETIME   = self.load_node_config()
    self.PATHINFO,  self.PATHTIME   = self.load_path_config()
    self.CONSTINFO, self.CONSTTIME  = self.load_const_config()
    self.BGPLSINFO                  = self.load_bgpls_config()

    # PATH NORM
    #print(self.PATHINFO)
    for name, raw in self.PATHINFO.items():
      self.N_PATHINFO[name] = self.normalize_pathdef(name, raw)
    #print(self.N_PATHINFO)

    # thread
    self.wp = threading.Thread(
      target= self.path_file_watcher, daemon=True,
      #args=(G_C_Queue, G_PATHINFO,  pathmtime,  "path",  G_PATHINFO ), daemon=True,
    )
    self.wp.start()
    self.wn = threading.Thread(
      target= self.node_file_watcher, daemon=True,
      #args=(G_C_Queue, G_PATHINFO,  pathmtime,  "path",  G_PATHINFO ), daemon=True,
    )
    self.wn.start()
    self.wc = threading.Thread(
      target= self.const_file_watcher, daemon=True,
      #args=(G_C_Queue, G_PATHINFO,  pathmtime,  "path",  G_PATHINFO ), daemon=True,
    )
    self.wc.start()

  def attach_g_q(self, Q):
    self.G_queue    = Q

  def attach_c_q(self, Q):
    self.C_queue    = Q

  def attach_c_cnt(self, cb):
    self.C_cnt      = cb

  # bgpls
  def load_bgpls_config(self):
    #mtime = os.path.getmtime(self.CONFIG_PATH)
    with open(self.CONFIG_PATH) as f:
      wk = json.load(f)
      return wk.get("bgpls",{})

  # const
  def load_const_config(self):
    mtime = os.path.getmtime(self.CONST_PATH)
    with open(self.CONST_PATH) as f:
      wk = json.load(f)
      return wk.get("const",{}), mtime

  def get_one_const(self, key):
    if ( key in self.CONSTINFO.keys() ):
      return self.CONSTINFO[key]
    else:
      return None

  def get_all_const(self):
    #wk = []
    #for key in self.N_PATHINFO.keys():
    #  wk.append(self.N_PATHINFO[key])
    return self.CONSTINFO

  def const_file_watcher(self):

    while True:

      wktime = os.path.getmtime(self.CONST_PATH)

      if wktime != self.CONSTTIME:
        wkdata, _ = self.load_const_config()
        diffs = diff_dict(self.CONSTINFO, wkdata)

        self.CONSTINFO = wkdata
        self.CONSTTIME = wktime
        #for name, raw in self.PATHINFO.items():
        #  self.N_PATHINFO[name] = self.normalize_pathdef(name, raw)

        for d in diffs:
          #pri = (100, self.C_cnt())
          self.G_queue.put({
            "type": "CONST_CONFIG",
            "diff": d,
          })
          #self.C_queue.put((pri,{
          #    "type": "CONST_CONFIG",
          #    "diff": d,
          #}))

      time.sleep(1)

  # node
  def load_node_config(self):
    mtime = os.path.getmtime(self.NODE_PATH)
    with open(self.NODE_PATH) as f:
      wk = json.load(f)
      return wk.get("node",{}), mtime

  def node_file_watcher(self):
    while True:
      wktime = os.path.getmtime(self.NODE_PATH)
      if wktime != self.NODETIME:
        wkdata, _ = self.load_node_config()
        diffs = diff_dict(self.NODEINFO, wkdata)

        self.NODEINFO = wkdata
        self.NODETIME = wktime

      time.sleep(1)

  # path
  def load_path_config(self):
    mtime = os.path.getmtime(self.PATH_PATH)
    with open(self.PATH_PATH) as f:
      wk = json.load(f)
      return wk.get("path",{}), mtime

  def get_one_pathdef(self, key):
    if ( key in self.N_PATHINFO.keys() ):
      return self.N_PATHINFO[key]
    else:
      return None

  def get_all_pathdef(self):
    wk = []
    for key in self.N_PATHINFO.keys():
      wk.append(self.N_PATHINFO[key])
    return wk

  def path_file_watcher(self):

    while True:

      wktime = os.path.getmtime(self.PATH_PATH)

      if wktime != self.PATHTIME:
        wkdata, _ = self.load_path_config()
        diffs = diff_dict(self.PATHINFO, wkdata)

        self.PATHINFO = wkdata
        self.PATHTIME = wktime
        for name, raw in self.PATHINFO.items():
          self.N_PATHINFO[name] = self.normalize_pathdef(name, raw)

        for d in diffs:
          pri = (100, self.C_cnt())
          self.C_queue.put((pri,{
              "type": "PATH_CONFIG",
              "diff": d,
          }))

      time.sleep(1)
  
  def normalize_pathdef(self, name, raw):

    src_set = []
    dst_set = []
    nodes   = self.NODEINFO

    bw   = raw.get("bw",  0)
    pri  = raw.get("pri", 0)
    optm = raw.get("optm", 0)
    mode = raw.get("mode", "ecmp")

    if mode == "ecmp":
      K = raw.get("K", 128)
      delta = 0
    else:
      K = raw.get("K", 1)
      delta = raw.get("delta", 0)

    if raw["type"] == "p2mp":
      K = 1
      delta = 0

    if raw["type"] == "p2p":
      if raw["src"] == "role:PE":
        for n in nodes.keys():
          if nodes[n]["role"] == "PE":
            #src_set.append(n)A
            rid = nodes[n].get("rid")
            if rid:
              src_set.append(rid)
      else:
        for n in raw["src"]:
          if n in nodes.keys():
            rid = nodes[n].get("rid")
            if rid:
              src_set.append(nodes[n]["rid"])
     # src_set = raw["src"]

      if raw["dst"] == "role:PE":
        for n in nodes.keys():
          if nodes[n]["role"] == "PE":
            #dst_set.append(n)
            rid = nodes[n].get("rid")
            if rid:
              dst_set.append(nodes[n]["rid"])
      else:
        for n in raw["dst"]:
          if n in nodes.keys():
            rid = nodes[n].get("rid")
            if rid:
              dst_set.append(nodes[n]["rid"])
        #dst_set = raw["dst"]

    # p2mp
    else:
      for n in raw["src"]:
        if n in nodes.keys():
          src_set.append(nodes[n]["rid"])
      if (len(src_set) == 1):
        for n in raw["dst"]:
          if n in nodes.keys():
            rid = nodes[n].get("rid")
            if rid:
              dst_set.append(nodes[n]["rid"])

    return PathDef(
      name=name,
      type=raw["type"],
      underlay=raw["underlay"],
      src=src_set,
      dst=dst_set,
      mode=mode,
      K=K,
      delta=delta,
      bw=bw,
      pri=pri,
      optm=optm
    )

def load_config(key):
    if key == "bgpls":
      path = CONFIG_PATH
    elif key == "node":
      path = NODE_PATH
    elif key == "const":
      path = CONST_PATH
    elif key == "path":
      path = PATH_PATH

    mtime = os.path.getmtime(path)
    with open(path) as f:
        return json.load(f), mtime

#def load_node():
#    cfg, mtime = load_config()
#    return cfg.get("node", {}), mtime

def load_defs(key):
    cfg, mtime = load_config(key)
    return cfg.get(key, {}), mtime

def load_p_defs(key):
    cfg, mtime = load_config(key)
    normalized = [
     pd
     for name, raw in pathinfo.items()
     if (pd := normalize_pathdef(name, raw, "ALL")) is not None
    ]
    return cfg.get(key, {}), mtime

#def get_node_mtime():
#    return os.path.getmtime(NODE_PATH)

#def file_watcher(ev_queue, last_data, last_mtime, key, interval=1):
def file_watcher(ev_queue, last_data, last_mtime, key, gval, interval=1):
    """
    変更を監視し、差分イベントを ev_queue に流す
    """
    if key == "bgpls":
      path = CONFIG_PATH
    elif key == "node":
      path   = NODE_PATH
      etype  = "NODE_CONFIG"
    elif key == "const":
      path = CONST_PATH
      etype  = "CONST_CONFIG"
    elif key == "path":
      path = PATH_PATH
      etype  = "PATH_CONFIG"

    while True:
      mtime = os.path.getmtime(path)
      if mtime != last_mtime:
          data, _ = load_defs(key)
          #last_data  = data
          #last_mtime = mtime

          diffs = diff_dict(last_data, data)
          gval.clear()
          gval.update(data)
          last_data  = data
          last_mtime = mtime
          for d in diffs:
              if key == "const":
                ev_queue.put({
                    "type": etype,
                    "diff": d,
                })
              else:
                pri = (100, int(time.time() * 1000))
                ev_queue.put((pri,{
                    "type": etype,
                    "diff": d,
                }))


          #last_data  = data
          #last_mtime = mtime

      time.sleep(interval)


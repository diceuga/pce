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

class ConfigManager:
  def __init__(self, log):
    self.log = log

    # DIR
    self.BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    self.NODE_PATH   = os.path.join(self.BASE_DIR, "etc", "node.json")
    self.CONST_PATH  = os.path.join(self.BASE_DIR, "etc", "const.json")
    self.PATH_PATH   = os.path.join(self.BASE_DIR, "etc", "path.json")
    self.CONFIG_PATH = os.path.join(self.BASE_DIR, "etc", "config.json")
    #print(str(BASE_DIR))

    # Q
    #self.G_queue    = None
    #self.C_queue    = None
    #self.C_cnt      = None

    # M
    self.PM         = None
    self.GM         = None
    self.BGPLSM     = None
    self.PCEPM      = None

    # info
    self.NODEINFO   = {}
    self.NODETIME   = 0
    self.PATHINFO   = {}
    self.N_PATHINFO = {}
    self.PATHTIME   = 0
    self.CONSTINFO  = {}
    self.CONSTTIME  = 0
    self.BGPLSINFO  = {}
    self.PCEPINFO   = {}
    self.CONFIGTIME = 0

    self.pcepserver = None

    # init load
    self.NODEINFO,  self.NODETIME   = self.load_node_config()
    self.PATHINFO,  self.PATHTIME   = self.load_path_config()
    self.CONSTINFO, self.CONSTTIME  = self.load_const_config()
    self.BGPLSINFO, self.CONFIGTIME = self.load_bgpls_config()
    self.PCEPINFO,  _               = self.load_pcep_config()

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

    self.wco = threading.Thread(
      target= self.config_file_watcher, daemon=True,
      #args=(G_C_Queue, G_PATHINFO,  pathmtime,  "path",  G_PATHINFO ), daemon=True,
    )
    self.wco.start()

  #def attach_g_q(self, Q):
  #  self.G_queue    = Q

  #def attach_c_q(self, Q):
  #  self.C_queue    = Q

  #def attach_c_cnt(self, cb):
  #  self.C_cnt      = cb

  def attach_PM(self, PM):
    self.PM = PM

  def attach_GM(self, GM):
    self.GM = GM

  # bgpls
  def load_bgpls_config(self):
    mtime = os.path.getmtime(self.CONFIG_PATH)
    with open(self.CONFIG_PATH) as f:
      wk = json.load(f)
      return wk.get("bgpls",{}), mtime

  # bgpls
  def load_pcep_config(self):
    mtime = os.path.getmtime(self.CONFIG_PATH)
    with open(self.CONFIG_PATH) as f:
      wk = json.load(f)
      return wk.get("pcep",{}),  mtime

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

  def attach_pcepserver(self,pcep):
    self.pcepserver = pcep
  def attach_bgpserver(self, bgp):
    self.bgpserver = bgp

  def config_file_watcher(self):

    while True:
      wktime = os.path.getmtime(self.CONFIG_PATH)
      if wktime != self.CONFIGTIME:
        wkdata1, _ = self.load_bgpls_config()
        wkdata2, _ = self.load_pcep_config()

        #diffs1 = diff_dict(self.BGPLSINFO["peers"], wkdata1["peers"])
        diffs1 = []
        diffs2 = diff_dict(self.PCEPINFO["pccs"], wkdata2["pccs"])
        self.BGPLSINFO = wkdata1
        self.PCEPINFO  = wkdata2
        self.COFIGTIME = wktime
        for d in diffs1:
          self.bgpserver.update_peers({
            "type": "BGPLS_CONFIG",
            "diff": d,
          })
        for d in diffs2:
          self.pcepserver.update_pccs({
            "type": "PCEP_CONFIG",
            "diff": d,
          })

      time.sleep(1)

  def const_file_watcher(self):

    while True:

      wktime = os.path.getmtime(self.CONST_PATH)

      if wktime != self.CONSTTIME:
        wkdata, _ = self.load_const_config()
        diffs = diff_dict(self.CONSTINFO, wkdata)

        self.CONSTINFO = wkdata
        self.CONSTTIME = wktime

        for d in diffs:
          self.GM.G_Queue.put({
            "type": "CONST_CONFIG",
            "diff": d,
          })
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
        wkpathinfo = {}
        for name, raw in self.PATHINFO.items():
          wkpathinfo[name] = self.normalize_pathdef(name, raw)
        
        self.N_PATHINFO = wkpathinfo

        #print(self.PATHINFO)
        #print(self.PATHINFO)

        for d in diffs:
          #pri = (100, self.C_cnt())
          #self.C_queue.put((pri,{
          pri = (100, self.PM.get_C_cnt())
          self.PM.C_queue.put((pri,{
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
    #sts  = raw.get("status", "up")

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
      optm=optm,
    )


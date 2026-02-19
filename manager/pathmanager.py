import queue
import time
from itertools import count
from utils.diff    import DiffType


class PathManager:
  def __init__(self, log):
    self.log = log
    self.C_Queue      = queue.PriorityQueue()
    self.C_Cnt        = count()

    self.computeX     = None

  def attach_X(self,X):
    self.computeX = X

  def attach_BM(self,BM):
    self.BM = BM

  def attach_CM(self,CM):
    self.CM = CM

  def attach_GM(self,GM):
    self.GM = GM

  def get_C_cnt(self):
    return next(self.C_Cnt)

  def remove_link(self, G, bw1, bw2, link_set):
    # bw check
    bwdiff = max(bw1 - bw2, 0)

    wkG = G
    for _, _, k in list(wkG.edges(keys=True)):
      bwflg = False
      if ( k in link_set ):
        bwflg = self.BM.chkbw(k,bwdiff)
      else:
        bwflg = self.BM.chkbw(k,bw1)
      if bwflg == False:
        wkG.remove_edge(k[0],k[1],k)

    return wkG

  def recompute_path_for_change(self):

    self.log.info("[COMPUTE] recompute path check start")

    normalized = G_CM.get_all_pathdef()
    paths_sorted = sorted(normalized, key=lambda p: (-p.pri, -p.bw))

    wk_G = {}

    # Link down check
    for pd in paths_sorted:
      wk_skip_flg = False

      self.log.info("[COMPUTE] compute check " + str(pd.name))

      if pd.name in G_PATH.keys():
        link_of_p     = G_PATH[pd.name]["link_set"]
        under_of_p    = G_PATH[pd.name]["underlay"]
        if under_of_p not in wk_G.keys():
          wk_G[under_of_p] = G_GM.get_one_graph(under_of_p)
        wk_del_flg = False
        for lkey in link_of_p:
          if not wk_G[under_of_p].has_edge(lkey[0],lkey[1],lkey):
            wk_del_flg = True
            break
        if wk_del_flg == False:
          wk_skip_flg = True

      if wk_skip_flg:
        self.log.info("[COMPUTE] compute skip")
        if pd.name in G_PATH.keys():
          G_PATH[pd.name]["opttime"] = int(time.time() * 1000)
        continue

      else:
        self.log.info("[COMPUTE] compute exec")
        qpri = ( 50, self.get_C_cnt() )
        self.C_Queue.put((qpri,{
          "type": "RECOMPUTEX",
          "comptype": pd.name
        }))

  def handle_event(self,ev):

    self.log.debug("[COMPUTE] HANDLE EVENT")

    ev_time = int(time.time() * 1000)

    ev_t = ev["type"]
    self.log.debug("[COMPUTE] event start "  + ev_t)

    if ev_t == "PATH_CONFIG":
      cid   = ev["diff"]["id"]
      difft = ev["diff"]["type"]
      if ( difft == DiffType.DEL ):
        pass
      elif ( difft == DiffType.ADD ):
        pass
      elif ( difft == DiffType.MOD ):
        qpri = ( 100, G_PM.C_cnt())
        self.C_Queue.put((qpri,{
          #"type": "RECOMPUTE4",
          "type": "RECOMPUTEX",
          "comptype": cid,
        }))
  
    elif ev_t == "RECOMPUTEX":
      #compute_pathsX(ev["comptype"])
      self.computeX(ev["comptype"])

  #-----------------------------------------------
  # compute
  def ecmp_paths(self, results, k):
    if not results:
        return []
    best = results[0]["cost"]
    ecmp = [r for r in results if r["cost"] == best]
    return ecmp[:k]

  def k_paths(results, k):
    return results[:k]


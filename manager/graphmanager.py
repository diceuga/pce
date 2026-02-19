# import 
import networkx as nx
import time
import copy
import threading
from utils.diff    import DiffType
import queue

class GraphManager:
  def __init__(self, log):
    self.log = log

    # GRAPH RELATED
    self.G_base   = nx.MultiDiGraph()
    self.G_base_t = 0
    self.Gc   = {}
    self.Gc_t = {}

    self.changed     = set()          # done
    self.linkdownflg = False          # done
    self.maxtime     = 9999999999999  # done
    self.changed_t   = self.maxtime   # done

    # snapshot
    self.snap_G_base_t = 0          # done
    self.snap_Gc       = {}         # done
    self.snap_Gc_t     = {}         # done

    # state
    self.bgpls_active = False       # done

    # queue
    self.G_Queue      = queue.Queue()

    self.BM           = None
    self.CM           = None
    self.PM           = None

    # start Q mon
    self.wq = threading.Thread( target=self.watch_graph_q, daemon=True )
    self.wq.start()

    # start nw change mon
    self.wn = threading.Thread( target=self.check_nw_change,  daemon=True )
    self.wn.start()

  def get_bgpls_active(self):
    return self.bgpls_active

  def attach_BM(self,BM):
    self.BM = BM

  def attach_CM(self,CM):
    self.CM = CM

  def attach_PM(self,PM):
    self.PM = PM

  def check_link(self, linkdata, constinfo):

    # if not applicable, return None
    new_d = {}
    cost = 65535

    # check if link is good for constraints
    # for example, FA128
    # access to G_base_node is needed

    # cost set
    if constinfo["metric"]  == "TE":
      cost = linkdata.get("te_metric", 65535)
    else:
      cost = linkdata.get("igp_metric", 65535)

    new_d["cost"] = cost

    return new_d 

  #----------------------------------------
  def make_G(self, base_G, const, constinfo):

    G_tmp = nx.MultiDiGraph()
    # node is node needed
    #G_tmp.add_nodes_from(base_G.nodes(data=True))

    # link check
    for u, v, k, d in base_G.edges(keys=True, data=True):
      new_d = self.check_link(d,constinfo)
      if new_d != None:
        G_tmp.add_edge(u, v, key=k, **new_d)

    return G_tmp

  #----------------------------------------
  def handle_graph_event(self,ev):

    ev_time = int(time.time() * 1000)

    t = ev["type"]
    self.log.info(f"[GRAPH] event start {t}")

    if t == "BGPLS_STATE":

      ev_state = ev["state"] 

      self.log.info(f"[GRAPH] status {ev_state}")

      if ev_state == "SYNCING":
        self.bgpls_active = False

      elif ev_state == "ACTIVE":
        if self.bgpls_active == False:
          # make G
          self.G_base_t = ev_time
          constinfo = self.CM.get_all_const()

          for const in constinfo.keys():
            self.Gc[const]  = self.make_G(self.G_base, const, constinfo[const])
            self.Gc_t[const]= ev_time

          #G_PATH = {}
          self.G_Queue.put({
            "type": "NWCHANGE"
          })

        self.bgpls_active = True

    elif t.startswith("LS_"):
      self.handle_ls(ev,ev_time)

    elif t == "CONST_CONFIG":
      cid              = ev["diff"]["id"]
      if ( ev["diff"]["type"] == DiffType.DEL ): # Just Del
        if cid in self.Gc.keys():
          self.Gc.pop(cid)

      elif ( ev["diff"]["type"] == DiffType.ADD ):
        if self.bgpls_active == True:
          self.Gc[cid]   = self.make_G(self.G_base, cid, ev["diff"]["new"])
          self.Gc_t[cid] = ev_time

    elif t == "NWCHANGE":
      self.snap_G_base_t  = self.G_base_t
      self.snap_Gc        = copy.deepcopy(self.Gc)
      self.snap_Gc_t      = self.Gc_t.copy()
 
      if self.linkdownflg == True:
        self.PM.recompute_path_for_change() 

      self.changed     = set()
      self.changed_t   = self.maxtime
      self.linkdownflg = False
 
  #----------------------------------------
  def handle_ls(self, ev, ev_time):
    nlri = ev["nlri"]
    nlri_type = nlri["nlri_type"]
    if nlri_type == 1:
      self.handle_node(ev,ev_time)
    elif nlri_type == 2:
      self.handle_link(ev,ev_time)

  #----------------------------------------
  # node update/withdraw handle
  def handle_node(self, ev, ev_time):
    nlri    = ev["nlri"]
    d       = nlri["detail"]
    node_id = d["local_node"]["ipv4_router_id"]

    # withdraw
    if ev["type"] == "LS_WITHDRAW":

      if self.G_base.has_node(node_id):
        self.G_base.remove_node(node_id)
      #if self.bgpls_active == True:
      #  for const in self.Gc.keys():
      #    if self.Gc[const].has_node(node_id):
      #      self.Gc[const].remove_node(node_id)
      return

    # update/add
    self.G_base.add_node(node_id,   nlri=d)

    #if self.bgpls_active == True:
    #  for const in self.Gc.keys():
    #    self.Gc[const].add_node(node_id, nlri=d)


  #----------------------------------------

  def ckeck_topo_change(old_attr: dict, new_attr: dict) -> bool:
    excludek = {"max_link_bw", "max_reservable_bw", "unreserved_bw"}
    for k in old_attr.keys() | new_attr.keys():
        if k not in excludek:
            if old_attr.get(k) != new_attr.get(k):
                return True
    return False

    return True
  # link update/withdraw handle
  def handle_link(self, ev, ev_time):
    nlri   = ev["nlri"]
    key    = ev["key"]
    d      = nlri["detail"]
    lsattr = ev["ls_attrs"]

    self.log.info(f"[GRAPH] LS Link")
    self.log.info(f"[GRAPH] " + str(ev))

    # None is ignore
    if None not in key:

      src = d["local_node"]["ipv4_router_id"]
      dst = d["remote_node"]["ipv4_router_id"]

      if ev["type"] == "LS_WITHDRAW":

        if self.G_base.has_edge(src, dst, key):
          self.G_base.remove_edge(src, dst, key)

        self.G_base_t = ev_time

        self.BM.delbw(key)

        if self.bgpls_active == True:
          for const in self.Gc.keys():
            if self.Gc[const].has_edge(src,dst,key):
              self.Gc[const].remove_edge(src,dst,key)
              self.Gc_t[const] = ev_time
             
          self.linkdownflg = True
          self.changed.add(key)
          self.changed_t=min(self.changed_t, ev_time)

        return

      # ADD / UPDATE
      if "igp_metric" in lsattr:
        oldlsattr = self.G_base.get_edge_data(src, dst, key)

        wk_topo_change = False

        if oldlsattr != None:
          wk_topo_change = ckeck_topo_change(oldlsattr,lsattr)

        self.G_base.add_edge(src, dst, key=key, **lsattr)

        if wk_topo_change:
          self.G_base_t = ev_time

        wk_maxrsvbw = 0
        wk_unrsvbw  = 0

        if "unreserved_bw" in lsattr.keys():
          wk_unrsvbw = lsattr["unreserved_bw"][0]

        if "max_reservable_bw" in lsattr.keys():
          wk_maxrsvbw = lsattr["max_reservable_bw"]
          
        self.BM.updbw(key, wk_unrsvbw, wk_maxrsvbw)

        if self.bgpls_active == True:
          for const in self.Gc.keys():
            constinfo = self.CM.get_one_const(const)
            new_d = self.check_link(lsattr, constinfo)

            oldflg= self.Gc[const].has_edge(src,dst,key)

            # check add/mod/del
            if new_d != None:
              if oldflg == True:
                self.Gc[const].add_edge(src, dst, key=key, **new_d)
                if wk_topo_change:
                  self.Gc_t[const] = ev_time
              else:
                self.Gc[const].add_edge(src, dst, key=key, **new_d)
                self.Gc_t[const] = ev_time
            else:
              if oldflg == True:
                self.Gc[const].remove_edge(src, dst, key)
                self.Gc_t[const] = ev_time

          self.changed.add(key)
          self.changed_t=min(self.changed_t, ev_time)


  #----------------------------------------
  def check_nw_change(self):

    while True:
      if len(self.changed) >0 :
        if int(time.time() * 1000) - self.changed_t > 200:
          self.G_Queue.put({
            "type": "NWCHANGE"
          })
      time.sleep(0.1)

  #----------------------------------------
  def watch_graph_q(self):
    while True:
      ev = self.G_Queue.get()
      self.handle_graph_event(ev)

  #----------------------------------------
  def on_bgpls_event(self, ev):
    self.G_Queue.put(ev)

  #----------------------------------------
  def get_one_graph_infos(self,gid):
    return (
      copy.deepcopy(self.snap_Gc[gid]),
      self.snap_Gc_t[gid],
    )

  #----------------------------------------
  def get_all_graphs(self):
    return copy.deepcopy(self.snap_Gc)

  def get_one_graph(self,gid):
    return copy.deepcopy(self.snap_Gc[gid])

  #----------------------------------------
  def get_last_g_time(self):
    return self.G_base_t


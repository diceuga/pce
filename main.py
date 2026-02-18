# import 
import networkx as nx
import heapq
import json
import os
import time
import copy
from collections import defaultdict
from datetime import datetime
import threading
from protocols.bgpserver import BgpServer
from utils.logging import setup_logging
from utils.config  import ConfigManager
from utils.diff    import DiffType
import logging
import queue
from itertools import count
from manager.bwmanager import BWManager

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
    #self.bwdownflg   = False          # done
    self.maxtime     = 9999999999999  # done
    self.changed_t   = self.maxtime   # done
    #self.bwchanged   = set()          # done
    #self.bwchanged_t = self.maxtime   # done

    # snapshot
    #self.snap_G_base   = None       # done
    self.snap_G_base_t = 0          # done
    self.snap_Gc       = {}         # done
    self.snap_Gc_t     = {}         # done
    #self.snap_changed  = set()

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

  #----------------------------------------
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
    G_LOG.info(f"[GRAPH] event start {t}")

    if t == "BGPLS_STATE":

      ev_state = ev["state"] 

      G_LOG.info(f"[GRAPH] status {ev_state}")

      if ev_state == "SYNCING":
        self.bgpls_active = False

      elif ev_state == "ACTIVE":
        if self.bgpls_active == False:
          # make G
          self.G_base_t = ev_time
          #constinfo = get_G_CONSTINFO()
          constinfo = self.CM.get_all_const()

          for const in constinfo.keys():
            self.Gc[const]  = self.make_G(self.G_base, const, constinfo[const])
            self.Gc_t[const]= ev_time

          #G_PATH = {}
          self.G_Queue.put({
            #"type": "RECOMPUTE4",
            "type": "NWCHANGE"
            #"comptype"  : "NWCHANGE",
            #"subcomptype": "INIT",
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
    #elif t == "RECOMPUTE4":
      #ctype  = ev["comptype"]
      #cstype = ev["subcomptype"]
      #self.snap_G_base    = copy.deepcopy(self.G_base)
      # copy to snapshot
      self.snap_G_base_t  = self.G_base_t
      self.snap_Gc        = copy.deepcopy(self.Gc)
      self.snap_Gc_t      = self.Gc_t.copy()
 
      if self.linkdownflg == True:
        recompute_path_for_change()
      #recompute_path_for_change()

      self.changed     = set()
      self.changed_t   = self.maxtime
      self.linkdownflg = False
 
      #qpri = ( 50, ev_time )
      #qpri = ( 50, get_G_C_cnt())
      #G_C_Queue.put((qpri,{
      #  "type": "RECOMPUTE4",
      #  "comptype": ctype,
      #  "comptype": ctype,
      #}))
      #recompute_path_for_change()

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

    G_LOG.info(f"[GRAPH] LS Link")
    G_LOG.info(f"[GRAPH] " + str(ev))

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
        #print("new-lsattr")
        #print(lsattr)
        #print("old-lsattr")
        #print(oldlsattr)

        wk_topo_change = False

        if oldlsattr != None:
          wk_topo_change = ckeck_topo_change(oldlsattr,lsattr)

        self.G_base.add_edge(src, dst, key=key, **lsattr)

        if wk_topo_change:
          self.G_base_t = ev_time

        wk_maxrsvbw = 0
        wk_unrsvbw  = 0
        #wkbwdownflg   = False

        if "unreserved_bw" in lsattr.keys():
          wk_unrsvbw = lsattr["unreserved_bw"][0]

        if "max_reservable_bw" in lsattr.keys():
          wk_maxrsvbw = lsattr["max_reservable_bw"]
          
        self.BM.updbw(key, wk_unrsvbw, wk_maxrsvbw)

        if self.bgpls_active == True:
          #constinfo = get_G_CONSTINFO()
          for const in self.Gc.keys():
            #constinfo = get_one_G_CONSTINFO(const)
            constinfo = self.CM.get_one_const(const)
            #new_d = self.check_link(lsattr, constinfo[const])
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
            #"type": "RECOMPUTE4",
            "type": "NWCHANGE"
            #"comptype"   : "NWCHANGE",
            #"subcomptype": "UPDATE",
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
    #return self.snap_G_base_t
    return self.G_base_t


class PathManager:
  def __init__(self, log):
    self.log = log
    self.C_Queue      = queue.PriorityQueue()
    self.C_Cnt        = count()

  def get_C_cnt(self):
    return next(self.C_Cnt)

  


    
  


#------------------------------------------
# PATH RELATED
#------------------------------------------
def ecmp_paths(results, k):
    if not results:
        return []
    best = results[0]["cost"]
    ecmp = [r for r in results if r["cost"] == best]
    return ecmp[:k]

def k_paths(results, k):
    return results[:k]

def k_physical_paths_visited(DG, src, dst, K, delta, dist_dst, best, mode):
    results   = []
    sum_links = []

    def dfs(u, cost, path, links, visited):
        # prune
        if cost + dist_dst.get(u, float("inf")) > best + delta:
            return

        if u == dst:
            results.append({
                "cost": cost,
                "path": path + [u],
                #"links": links.copy()
                "links": list(links)
            })
            return

        for _, v, k, data in DG.out_edges(u, keys=True, data=True):
            if v in visited:
                continue

            visited.add(v)
            path.append(u)
            links.append(k)

            dfs(
                v,
                cost + data["cost"],
                path,
                links,
                visited
            )

            links.pop()
            path.pop()
            visited.remove(v)

    dfs(src, 0, [], [], {src})

    results.sort(key=lambda x: (x["cost"], len(x["path"])))

    if ( mode == "ecmp" ):
      R = ecmp_paths(results, K)
    else:
      R = k_paths(results,K)

    return R


#def build_paths(pd: PathDef, Gc, Gc_rev):
def build_paths(pd, Gc, Gc_rev):

    results={}
    link_set=set()
    src_set = []
    dst_set = []

    if ( pd.type == "p2p" ): 

      for src in pd.src:
        results[src] = {}

        dist_src = nx.single_source_dijkstra_path_length( Gc, src, weight="cost" )

        for dst in pd.dst:
          if src == dst:
            continue

          best = dist_src.get(dst)
          if best is None:
            continue

          dist_dst = nx.single_source_dijkstra_path_length( Gc_rev, dst, weight="cost" )
          paths = k_physical_paths_visited(
            Gc, src, dst, pd.K, pd.delta, dist_dst, best, pd.mode
          )

          results[src][dst]   = paths
          for r in paths:
            for r2 in r["links"]:
              link_set.add(r2)
    else:
    # p2mp

      src = pd.src[0]

      (dest_dist, parent0) = dijkstra_to_dests(Gc, src, pd.dst)
      o_dests = sorted(dest_dist, key=lambda d: dest_dist[d], reverse=True)

      if len(o_dests) == 0:
        return {}

      tree_edges = set()
      tree_nodes = set()

      first = o_dests[0]
      path = build_path(parent0, src, first)

      for u, v, key in path:
        tree_edges.add((u, v, key))
        tree_nodes.add(u)
        tree_nodes.add(v)


      for d in o_dests[1:]:
        if d in tree_nodes:
          continue

        dist2, parent2 = dijkstra_from_tree(Gc, tree_nodes, tree_edges)

        # back to tree from d
        u = d
        path = []

        #if u in tree_nodes:
        #  continue

        while u not in tree_nodes:
          pi, key = parent2[u]
          path.append((pi, u, key))
          u = pi

        path.reverse()

        for e in path:
          tree_edges.add(e)
          tree_nodes.add(e[0])
          tree_nodes.add(e[1])

      for e in tree_edges:
        link_set.add(e[2])
      

      #tree = tree_edges_to_json(tree_edges, src, Gc)
      tree = tree_edges_to_json(tree_edges, src)

      results = tree

    return results, link_set


#def tree_edges_to_json(tree_edges, src, G):
def tree_edges_to_json(tree_edges, src):
    #adj = build_adj(tree_edges, G)
    adj = build_adj(tree_edges)
    return {src: build_tree_json(adj, src)}

def build_tree_json(adj, u, parent=None):
    node  = {}
    node["children"]  = []

    #for v, key, cost in adj[u]:
    for v, key in adj[u]:
        if v == parent:
            continue

        a = build_tree_json(adj, v, u)

        if a == None:
          node["children"].append({
              "node" : v,
              "link" : key,
          })
        else:
          node["children"].append({
              "node" : v,
              "link" : key,
              **a
          })
    
    if node["children"] == []:
      return None

    return node

def build_adj(tree_edges):
    adj = defaultdict(list)
    for u, v, key in tree_edges:
        adj[u].append((v, key))
        adj[v].append((u, key))
    return adj

def dijkstra_from_tree(G, tree_nodes, tree_edges):

    dist = {}
    parent = {}
    pq = []

    # tree 上のノードは距離 0
    for n in tree_nodes:
        dist[n] = 0
        pq.append((0, n))

    heapq.heapify(pq)

    while pq:
        cost_u, u = heapq.heappop(pq)
        if cost_u > dist[u]:
            continue

        for v, keydict in G[u].items():
            for key, attr in keydict.items():
                w = attr["cost"]

                # 既存 tree を優遇（branch 抑制）
                if (u, v, key) in tree_edges: # 
                    w *= 0.5

                new = cost_u + w
                if v not in dist or new < dist[v]:
                    dist[v] = new
                    parent[v] = (u, key)
                    heapq.heappush(pq, (new, v))

    return dist, parent


def build_path(parent, src, dest):
    path = []
    u = dest
    while u != src:
        p, key = parent[u]
        path.append((p, u, key))
        u = p
    return list(reversed(path))


def dijkstra_to_dests(G, src, dests):
    dests = set(dests)
    found = {}
    parent = {}
    dist = {src: 0}
    pq = [(0, src)]

    while pq and dests:
      cost_u, u = heapq.heappop(pq)

      if cost_u > dist[u]:
        continue

      if u in dests:
        found[u] = cost_u
        dests.remove(u)
        # continue since there are another dests

      for v, keydict in G[u].items():
        for key, attr in keydict.items():
          w = attr["cost"]   
          new = cost_u + w
          if v not in dist or new < dist[v]:
              dist[v] = new
              parent[v] = (u, key)
              heapq.heappush(pq, (new, v))

    return found, parent

def compute_pathsX(pid):
  G_LOG.info("[COMPUTE] computeX rtn start for " + pid)

  if G_PATH.get(pid) != None:
    G_LOG.info("[COMPUTE] skip there is other candidate")
    return

  #cpathinfo  = G_PATH.get(pid)
  cpathinfo  = G_PATH_d.get(pid)
  pd         = G_CM.get_one_pathdef(pid)

  G_LOG.debug("[COMPUTE] path name: "  + str(pid))
  G_LOG.debug("[COMPUTE] path def:  "  + str(pd))

  wkG         = None
  wkG_t       = None
  wk_skip_flg = False

  #bef_g_time  = G_GM.get_last_g_time()

  #graph / linkinfo:
  ( wkG, wkG_t ) = G_GM.get_one_graph_infos(pd.underlay)
  bef_g_time  = wkG_t


  #bwdiff = 0
  if cpathinfo != None:
    #bwdiff  = pathinfo["bw"] - cpathinfo["bw"]
    removed_wkG  = remove_link2(
                     wkG, pd.bw, cpathinfo["bw"], cpathinfo["link_set"]
                   )
  else:
    #new
    #bwdiff  = pathinfo["bw"]
    removed_wkG  = remove_link2(
                     wkG, pd.bw, 0, set()
                   )

  removed_rwkG = removed_wkG.reverse(copy=False)
  wk_results = {}

  #G_LOG.info("[COMPUTE] path name:" + str(pd.name))
  G_LOG.info("[COMPUTE] compute start for " + pd.name)

  p, link_set  = build_paths(pd, removed_wkG, removed_rwkG)

  G_LOG.info("[COMPUTE] compute end for" + pd.name)
  #G_LOG.info("[COMPUTE] calc results")
  #G_LOG.info("[COMPUTE] " + str(p))

  wk_results["time"]      = wkG_t
  wk_results["initiate"]  = True
  wk_results["opttime"]   = int(time.time() * 1000)
  wk_results["underlay"]  = pd.underlay
  wk_results["bw"]        = pd.bw
  wk_results["detail"]    = p
  wk_results["link_set"]  = link_set

  #G_LOG.info("[COMPUTE] " + str(wk_results))

  aft_g_time  = G_GM.get_last_g_time()

  if bef_g_time != aft_g_time:
    G_LOG.info("[COMPUTE] NW change happend during compute")
    return
  else:
    if pd.name in G_PATH_d.keys():
      if ( wk_results["detail"] == G_PATH_d[pd.name]["detail"] ):
        G_LOG.info("[COMPUTE] calc results same as bef")
        G_PATH_d[pd.name]["opttime"] = int(time.time() * 1000)
        return

    G_LOG.info("[COMPUTE] calc results")
    G_LOG.info("[COMPUTE] new path" + str(wk_results))

  # delete path ---> need to change
  #if pd.name in G_PATH.keys():
  #  delete_path(pd.name)

  # add
  for wk_l in wk_results["link_set"]:
    if cpathinfo != None:
      if wk_l in cpathinfo["link_set"]:
        bwdiff = max(0, wk_results["bw"] - cpathinfo["bw"] )
        G_BM.addwkpbw(wk_l, bwdiff,pd.name)
      else:
        G_BM.addwkpbw(wk_l, wk_results["bw"],pd.name)
    else:
      G_BM.addwkpbw(wk_l, wk_results["bw"],pd.name)

  # wkbw

  if pd.name not in G_PATH.keys():
    G_PATH[pd.name] = {}

  G_PATH[pd.name] = wk_results

  return True


def recompute_path_for_change():

  G_LOG.info("[COMPUTE] recompute path check start")

  normalized = G_CM.get_all_pathdef()

  paths_sorted = sorted(
    normalized,
    key=lambda p: (-p.pri, -p.bw)
  ) 

  wk_G = {}
  #wk_skip_flg = False

  # Link down check 
  for pd in paths_sorted:
    wk_skip_flg = False

    G_LOG.info("[COMPUTE] compute check " + str(pd.name))

    if pd.name in G_PATH.keys():
      link_of_p     = G_PATH[pd.name]["link_set"]
      under_of_p    = G_PATH[pd.name]["underlay"]
      if under_of_p not in wk_G.keys():
        wk_G[under_of_p] = G_GM.get_one_graph(under_of_p)
      wk_del_flg = False
      for lkey in link_of_p:
        #if not wk_graphs[under_of_p].has_edge(lkey[0],lkey[1],lkey):
        if not wk_G[under_of_p].has_edge(lkey[0],lkey[1],lkey):
          wk_del_flg = True
          break
      if wk_del_flg == False:
        #G_LOG.info("[COMPUTE] skip no link deleted")
        wk_skip_flg = True

    if wk_skip_flg:
      G_LOG.info("[COMPUTE] compute skip")
      if pd.name in G_PATH.keys():
        G_PATH[pd.name]["opttime"] = int(time.time() * 1000)
      continue

    else:
      G_LOG.info("[COMPUTE] compute exec")
      qpri = ( 50, self.PM.get_C_cnt() )
      #G_C_Queue.put((qpri,{
      self.PM.C_Queue.put((qpri,{
        "type": "RECOMPUTEX",
        "comptype": pd.name
      }))

def delete_p_from_d(p):
  if p in G_PATH_d.keys():
    for lk in list(G_PATH_d[p]["link_set"]):
      G_BM.delpbw(lk,p)
    G_PATH_d.pop(p)

def check_path_optm():
  # BW check -> if wkbw + reserbable > max change.
  # PATH optimize
  # PATH create
  while True:

    time.sleep(3.0)

    bgpls_status = G_GM.get_bgpls_active()
    if bgpls_status == False:
      continue

    # test move p to _d
    debug_move_p()

    # all path def
    allpathdef = G_CM.get_all_pathdef()
    sorted_pathdef1 = sorted( allpathdef, key=lambda p: (-p.pri, -p.bw) ) # for new
    sorted_pathdef2 = sorted( allpathdef, key=lambda p: (p.pri, -p.bw)  ) # for bw

    #print(G_PATH_d)

    #-----------------------------------------
    # del not sync pcep
    #-----------------------------------------
    # del from both G_PATH/G_PATH_d
    for pd in sorted_pathdef1:
      pass

    #-----------------------------------------
    # del not in path def
    #-----------------------------------------
    for k in list(G_PATH_d.keys()):
      if G_PATH_d[k]["initiate"] == True:
      # check path is initiated lsp
        wk = G_CM.get_one_pathdef(k)
        #print(wk)
        if wk == None:
          if k not in G_PATH.keys():
            #print("delete")
            delete_p_from_d(k)

    #print(G_PATH_d)
    #-----------------------------------------
    # BW check
    #-----------------------------------------
    allbwinfo   = G_BM.getallbw()
    cpath       = set()
    for lk in list(allbwinfo.keys()):
      #print(lk)
      v = allbwinfo.get(lk)
      if v!= None:
        #print(v)
        reducebw = v["sumpbw"] - v["maxbw"]
        #print(reducebw)
        #reducebw = 5 # debug
        if ( reducebw > 0  ): 
          #print("why")
          wkbw = 0
          for pd in sorted_pathdef2:
            if pd.name in G_PATH_d.keys():
              if G_PATH_d[pd.name]["bw"] > 0:
                if lk in G_PATH_d[pd.name]["link_set"]:
                  cpath.add(pd.name)
                  wkbw += G_PATH_d[pd.name]["bw"]
                  if wkbw > reducebw:
                    break
    for p in cpath:
      #qpri = ( 100, get_G_C_cnt()) 
      qpri = ( 100, G_PM.get_C_cnt()) 
      #G_C_Queue.put((qpri,{
      G_PM.C_Queue.put((qpri,{
          "type": "RECOMPUTEX",
          "comptype": p,
      }))

    #-----------------------------------------
    # optm
    #-----------------------------------------
    for k in G_PATH_d.keys():
      pathinfo = G_CM.get_one_pathdef(k)
      now      = int(time.time() * 1000)
      calctime = int(G_PATH_d[k]["opttime"])
      #optmtime = pathinfo[k].get("optm")
      optmtime = pathinfo.optm
      #if optmtime != None:
      if optmtime != 0:
        optmtime = max(optmtime * 1000, 10000)
        if now - calctime > optmtime:
          #qpri = ( 100, now )
          #qpri = ( 100, get_G_C_cnt())
          qpri = ( 100, G_PM.get_C_cnt())
          #print("qpri")
          #print(qpri)
          #G_C_Queue.put((qpri,{
          G_PM.C_Queue.put((qpri,{
            #"type": "RECOMPUTE4",
            "type": "RECOMPUTEX",
            "comptype": k,
          }))
      #for wk_l in G_PATH[pathid]["link_set"]:
    
    #-----------------------------------------
    # New
    #-----------------------------------------
    # check PCEP sattus if sync calc
    for pd in sorted_pathdef1:
      # if PCEP, currently skip
      if pd.name not in G_PATH_d.keys():
        #qpri = ( 100, get_G_C_cnt())
        #G_C_Queue.put((qpri,{
        qpri = ( 100, G_PM.get_C_cnt())
        G_PM.C_Queue.put((qpri,{
          #"type": "RECOMPUTE4",
          "type": "RECOMPUTEX",
          "comptype": pd.name,
        }))

def handle_event(ev):

  global G_PATH
  G_LOG.debug("[COMPUTE] HANDLE EVENT")

  ev_time = int(time.time() * 1000)

  ev_t = ev["type"]
  G_LOG.debug("[COMPUTE] event start "  + ev_t)

  if ev_t == "PATH_CONFIG":
    cid   = ev["diff"]["id"]
    difft = ev["diff"]["type"]
    if ( difft == DiffType.DEL ):
      pass
      #qpri = ( 100, get_G_C_cnt())
      #G_C_Queue.put((qpri,{
      #  "type": "PATHDEL",
      #  "pathid": cid,
      #}))
      # delete path from nw(pcep)
    elif ( difft == DiffType.ADD ):
      pass
      #qpri = ( 100, get_G_C_cnt())
      #G_C_Queue.put((qpri,{
      #  #"type": "RECOMPUTE4",
      #  "type": "RECOMPUTEX",
      #  "comptype": cid,
      #}))
    elif ( difft == DiffType.MOD ):
      #qpri = (100, ev_time)
      #qpri = ( 100, get_G_C_cnt())
      #G_C_Queue.put((qpri,{
      qpri = ( 100, G_PM.C_cnt())
      G_PM.C_Queue.put((qpri,{
        #"type": "RECOMPUTE4",
        "type": "RECOMPUTEX",
        "comptype": cid,
      }))

  elif ev_t == "RECOMPUTEX":
    #compute_paths4(comptype=ev["comptype"])
    compute_pathsX(ev["comptype"])

  #elif ev_t == "PATHDEL":
  #  delete_path(ev["pathid"])

def debug_move_p():
  #print("debug move p")
  #print(G_BM.getallbw())
  #print("------------------")
  for p in list(G_PATH.keys()):
    G_PATH_d[p] = G_PATH[p]

    for lk in list(G_PATH_d[p]["link_set"]):
      #print(G_PATH_d[p])
      G_BM.addpbw(lk,G_PATH_d[p]["bw"],p)

    for lk in list(G_PATH[p]["link_set"]):
      G_BM.delwkpbw(lk,p)

    G_PATH.pop(p)


  #print(G_BM.getallbw())
  
#def delete_path(pathid):
#  # PCEP
#  
#  # delete from linktopath and bw
#  if pathid in G_PATH.keys():
#    for wk_l in G_PATH[pathid]["link_set"]:
#      G_BM.delpbw(wk_l, G_PATH[pathid]["bw"], pathid)
#
#    G_PATH.pop(pathid)

def remove_link2(G, bw1, bw2, link_set):
  # bw check
  bwdiff = max(bw1 - bw2, 0)
  #wkG = copy.deepcopy(G)
  wkG = G
  for _, _, k in list(wkG.edges(keys=True)):
    #print(k)
    bwflg = False
    if ( k in link_set ):
      bwflg = G_BM.chkbw(k,bwdiff)
    else:
      bwflg = G_BM.chkbw(k,bw1)
    if bwflg == False:
      wkG.remove_edge(k[0],k[1],k)

  return wkG


#def get_G_C_cnt():
#  return next(G_C_Cnt)

#-------------------------------------------
# Main
# LOG
setup_logging(os.path.dirname(os.path.abspath(__file__)))
G_LOG = logging.getLogger()

# Queue
#G_C_Queue                  = queue.PriorityQueue() # calc
#G_C_Cnt                    = count()

# Path
G_PATH                     = {}
G_PATH_d                   = {}

# Manager
G_BM = BWManager(G_LOG)     # BWManager
G_GM = GraphManager(G_LOG)  # GraphManager
G_CM = ConfigManager(G_LOG) # ConfigManager
G_PM = PathManager(G_LOG)   # PathManager


# Attach
#G_CM.attach_c_q(G_C_Queue)
#G_CM.attach_c_cnt(get_G_C_cnt)
G_CM.attach_g_q(G_GM.G_Queue)
G_CM.attach_PM(G_PM)
G_CM.attach_GM(G_GM)

G_GM.attach_BM(G_BM)
G_GM.attach_CM(G_CM)
G_GM.attach_PM(G_PM)

def main():

  #global G_CONSTINFO, G_C_Queue
  #global G_C_Queue

  G_LOG.info("Main start")

  ############## BGP START
  bgp = BgpServer(G_CM.BGPLSINFO)
  bgp.register_main_callback(G_GM.on_bgpls_event)
  bgp.start()

  ############## PCEP START

  ############## path optm
  po = threading.Thread(
    target=check_path_optm, daemon=True,
  )
  po.start()

  ############## Loop start 
  while True:
    #pri, ev = G_C_Queue.get()
    pri, ev = G_PM.C_Queue.get()
    handle_event(ev)

# start
if __name__ == "__main__":
  main()



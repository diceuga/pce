# import 
import networkx as nx
import heapq
import json
import os
import time
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Literal
from datetime import datetime
import threading
from protocols.bgpserver import BgpServer
from utils.load    import load_defs,  file_watcher
from utils.diff    import DiffType
from utils.logging import setup_logging
import logging
import queue
#from queue import PriorityQueue
from itertools import count

Mode = Literal["ecmp", "k"]
PathType = Literal["p2p", "p2mp"]

# PathDef
@dataclass(frozen=True)
class PathDea:
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

class GraphManager:
  def __init__(self, log):
    self.log = log

    # GRAPH RELATED
    self.G_base   = nx.MultiDiGraph()
    self.G_base_t = 0
    self.Gc   = {}
    self.Gc_t = {}

    self.linkstate = {}             # done
    self.changed   = set()          # done
    self.maxtime   = 9999999999999  # done
    self.changed_t = self.maxtime   # done

    # snapshot
    #self.snap_G_base   = None       # done
    self.snap_G_base_t = 0          # done
    self.snap_Gc       = {}         # done
    self.snap_Gc_t     = {}         # done
    #self.snap_linkstate= {}
    #self.snap_changed  = set()

    # state
    self.bgpls_active = False       # done

    # queue
    self.G_Queue      = queue.Queue()

  #----------------------------------------
  def check_link(self, linkdata, constinfo):

    # if not applicable, return None

    new_d = {}
    cost = 65535

    # check if link is good for constraints
    # access needed G_base_node

    #new_d["te"]   = linkdata.get("te_metric", 65535)
    #new_d["igp"]  = linkdata.get("igp_metric", 65535)

    #if constinfo["metric"]  == "TE":
    #  new_d["cost"] = new_d["te"]
    #else:
    #  new_d["cost"] = new_d["igp"]

    # cost
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

    global G_PATH 
    global G_C_Queue

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
          constinfo = get_G_CONSTINFO()

          for const in constinfo.keys():
            self.Gc[const]  = self.make_G(self.G_base, const, constinfo[const])
            self.Gc_t[const]= ev_time

          G_PATH = {}
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
      self.snap_G_base_t  = self.G_base_t
      self.snap_Gc        = copy.deepcopy(self.Gc)
      self.snap_Gc_t      = self.Gc_t.copy()
  
      self.changed   = set()
      self.changed_t = self.maxtime
 
      #qpri = ( 50, ev_time )
      #qpri = ( 50, get_G_C_cnt())
      #G_C_Queue.put((qpri,{
      #  "type": "RECOMPUTE4",
      #  "comptype": ctype,
      #  "comptype": ctype,
      #}))
      recompute_path_for_change()

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
  # link update/withdraw handle
  def handle_link(self, ev, ev_time):
    nlri   = ev["nlri"]
    key    = ev["key"]
    d      = nlri["detail"]
    lsattr = ev["ls_attrs"]

    # None is ignore
    if None not in key:

      src = d["local_node"]["ipv4_router_id"]
      dst = d["remote_node"]["ipv4_router_id"]

      if ev["type"] == "LS_WITHDRAW":

        if self.G_base.has_edge(src, dst, key):
          self.G_base.remove_edge(src, dst, key)

        self.G_base_t = ev_time

        if key not in self.linkstate:
          self.linkstate[key] = {}
        self.linkstate[key]["state"]  = "down"
        self.linkstate[key]["cstate"] = "down"
        self.linkstate[key]["statetime"] = ev_time

        if self.bgpls_active == True:
          for const in self.Gc.keys():
            if self.Gc[const].has_edge(src,dst,key):
              self.Gc[const].remove_edge(src,dst,key)
              self.Gc_t[const] = ev_time
              
          self.changed.add(key)
          self.changed_t=min(self.changed_t, ev_time)

        return

      # ADD / UPDATE
      if "igp_metric" in lsattr:

        self.G_base.add_edge(src, dst, key=key, **lsattr)
        self.G_base_t = ev_time

        if key not in self.linkstate:
          self.linkstate[key] = {}
        #if key in self.linkstate:
        #  self.linkstate[key]["state"]  = "up"
        #  self.linkstate[key]["statetime"] = ev_time
        #  self.linkstate[key]["cstate"] = "uping"
        #else:
        #  self.linkstate[key] = {}
        #  if ( self.bgpls_active == True ):
        #    self.linkstate[key]["state"]  = "up"
        #    self.linkstate[key]["statetime"] = ev_time
        #    self.linkstate[key]["cstate"] = "uping"
        #  else:
        self.linkstate[key]["state"]  = "up"
        self.linkstate[key]["statetime"] = ev_time
        self.linkstate[key]["cstate"] = "up"

        # bw

        if self.bgpls_active == True:
          #constinfo = get_G_CONSTINFO()
          for const in self.Gc.keys():
            constinfo = get_one_G_CONSTINFO(const)
            #new_d = self.check_link(lsattr, constinfo[const])
            new_d = self.check_link(lsattr, constinfo)
            if new_d != None:
              self.Gc[const].add_edge(src, dst, key=key, **new_d)
              self.Gc_t[const] = ev_time

          self.changed.add(key)
          self.changed_t=min(self.changed_t, ev_time)


  #----------------------------------------
  def check_link_state(self):

    while True:

      now     = int(time.time() * 1000)
      for k in self.linkstate.keys():
        #if self.linkstate[k]["cstate"] == "uping" :
        #  if ( now - self.linkstate[k]["statetime"] ) > 2000:
        #    self.linkstate[k]["cstate"] = "up"
        #    #print(str(k) + ":" + "uping->up")
        #
        if self.linkstate[k]["cstate"] == "down":
          if ( now - self.linkstate[k]["statetime"] ) > 3600000:
            if not self.G_base.has_edge(k[0],k[1],k):
              self.linkstate.pop(k)
            
      time.sleep(1.0)

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
  def get_graph_infos(self):
    return (
      copy.deepcopy(self.snap_Gc),
      self.snap_Gc_t.copy(),
      self.snap_G_base_t,
      copy.deepcopy(self.linkstate)
      )

  def get_one_graph_infos(self,gid):
    return (
      #copy.deepcopy(self.snap_Gc),
      copy.deepcopy(self.snap_Gc[gid]),
      self.snap_Gc_t[gid],
      #self.snap_G_base_t,
      copy.deepcopy(self.linkstate)
    )

  #----------------------------------------
  def get_all_graphs(self):
    return copy.deepcopy(self.snap_Gc)

  def get_one_graph(self,gid):
    return copy.deepcopy(self.snap_Gc[gid])

  #----------------------------------------
  def get_last_g_time(self):
    return self.snap_G_base_t
    #pass

  #----------------------------------------
  #----------------------------------------
  #----------------------------------------
  #----------------------------------------

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


def build_paths(pd: PathDea, Gc, Gc_rev):

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


def normalize_pathdef(name: str, raw: dict) -> PathDea:
  
  src_set = []
  dst_set = []
  nodes   = get_G_NODEINFO()

  # remcomp_check
  #if comptype == "PE": # PE CHANGE
  #  if "role:PE" not in ( raw["src"], raw["dst"] ):
  #    return
  #elif comptype not in ( "ALL", "NWCHANGE"):
  #  if name != comptype:
  #    return

  bw   = raw.get("bw",  0)
  pri  = raw.get("pri", 0)
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

  return PathDea(
    name=name,
    type=raw["type"],
    underlay=raw["underlay"],
    src=src_set,
    dst=dst_set,
    mode=mode,
    K=K,
    delta=delta,
    bw=bw,
    pri=pri
  )

def compute_pathsX(pid):
  G_LOG.info("[COMPUTE] computeX rtn start " + pid)

  pathinfo   = get_one_G_PATHINFO(pid)
  #print("pathinfo")

  #pd         = normalize_pathdef(pid, pathinfo, pid)
  pd         = normalize_pathdef(pid, pathinfo)

  G_LOG.info("[COMPUTE] path name: " + str(pid))
  G_LOG.debug("[COMPUTE] path def:  " + str(pathinfo))

  #  return (
  #    #copy.deepcopy(self.snap_Gc),
  #    copy.deepcopy(self.snap_Gc[gid])
  #    self.snap_Gc_t[gid].copy(),
  #    #self.snap_G_base_t,
  #    copy.deepcopy(self.linkstate)
  #  )

  wkG         = None
  wkG_t       = None
  wk_linkstate= None
  wk_skip_flg = False

  bef_g_time  = G_GM.get_last_g_time()

  # check
  #if pathinfo == None:
  ( wkG, wkG_t, wk_linkstate ) = G_GM.get_one_graph_infos(pd.underlay)

  #if pathinfo != None:
  #  punder = pathinfo["underlay"]
  #  ptime  = pathinfo["time"]
  #  if punder == pd.underlay:
  #    if ptime == wkG_t:
  #      G_LOG.info("[COMPUTE] underlay not changed ")
  #      wk_skip_flg = True

  if pd.name in G_PATH.keys():
     punder = G_PATH[pd.name]["underlay"]
     ptime  = G_PATH[pd.name]["time"]
     if punder == pd.underlay:
  #     ( wkG, wkG_t, wk_linkstate ) = G_GM.get_one_graph_infos(punder)
       if ptime == wkG_t:
         G_LOG.info("[COMPUTE] underlay not changed ")
         wk_skip_flg = True
  #   else:
  #     ( wkG, wkG_t, wk_linkstate ) = G_GM.get_one_graph_infos(pd.underlay)
#
#  else:
#    ( wkG, wkG_t, wk_linkstate ) = G_GM.get_one_graph_infos(pd.underlay)

  # skip
  if wk_skip_flg == True:
    G_LOG.info("[COMPUTE] compute skip")
    if pd.name in G_PATH.keys():
      G_PATH[pd.name]["opttime"] = int(time.time() * 1000)
    return

  removed_wkG  = remove_link2(wkG, wk_linkstate)
  removed_rwkG = removed_wkG.reverse(copy=False)

  wk_results = {}

  #G_LOG.info("[COMPUTE] path name:" + str(pd.name))
  G_LOG.debug("[COMPUTE] path def:" + str(pd))
  G_LOG.info("[COMPUTE] compute start for " + pd.name)

  p, link_set  = build_paths(pd, removed_wkG, removed_rwkG)

  G_LOG.info("[COMPUTE] compute end for" + pd.name)
  #G_LOG.info("[COMPUTE] calc results")
  #G_LOG.info("[COMPUTE] " + str(p))

  wk_results["time"]      = wkG_t
  wk_results["opttime"]   = int(time.time() * 1000)
  wk_results["underlay"]  = pd.underlay
  wk_results["detail"]   = p
  wk_results["link_set"] = link_set

  #G_LOG.info("[COMPUTE] " + str(wk_results))

  aft_g_time  = G_GM.get_last_g_time()

  if bef_g_time != aft_g_time:
    G_LOG.info("[COMPUTE] NW change happend during compute")
    return
  else:
    if pd.name in G_PATH.keys():
      if ( wk_results["detail"] == G_PATH[pd.name]["detail"] ):
        G_LOG.info("[COMPUTE] calc results same as bef")
        G_PATH[pd.name]["opttime"] = int(time.time() * 1000)
        return

    G_LOG.info("[COMPUTE] calc results")
    G_LOG.info("[COMPUTE] " + str(wk_results))


  # delete path
  if pd.name in G_PATH.keys():
    delete_path(pd.name)

  # add
  for wk_l in wk_results["link_set"]:
    if wk_l not in G_LINK_TO_PATH.keys():
      G_LINK_TO_PATH[wk_l] = set()
    G_LINK_TO_PATH[wk_l].add(pd.name)

  if pd.name not in G_PATH.keys():
    G_PATH[pd.name] = {}

  G_PATH[pd.name] = wk_results

  #print("G_PATH")
  #print(G_PATH)
  #print("link_to_X")
  #print(G_LINK_TO_PATH)
  #print("link_state")
  #print(G_GM.linkstate)

  return True


#def compute_paths5(comptype):
#
#  G_LOG.info("[COMPUTE] compute5 rtn start " + comptype)
#
#  ( wk_graphs, wk_graphs_t, 
#    wk_G_base_t, wk_link_state
#   ) = G_GM.get_graph_infos()    
#
#  wkG         = {}
#  rwkG        = {}
#
#  pathinfo = get_G_PATHINFO()
#
#  normalized = []
#
#  #normalized = [
#  #  pd
#  for name, raw in pathinfo.items():
#    normalized.append(normalize_pathdef(name, raw))
#    #if (pd := normalize_pathdef(name, raw, comptype)) is not None
#  #]
#  paths_sorted = sorted(
#    normalized,
#    key=lambda p: (-p.pri, -p.bw)
#  )
#
#  for pd in paths_sorted:
#
#    wk_skip_flg = False
#    #G_LOG.info("[COMPUTE] skipflg:" + str(wk_skip_flg))
#
#    G_LOG.info("[COMPUTE] path name:" + str(pd.name))
#    G_LOG.debug("[COMPUTE] path def:" + str(pd))
#
#    # check if need recomp
#    if ( comptype == "NWCHANGE" ):
#      if pd.name in G_PATH.keys():
#        link_of_p = G_PATH[pd.name]["link_set"]
#        if G_PATH[pd.name]["time"] == wk_graphs_t[pd.underlay]:
#          G_LOG.info("[COMPUTE] skip pathtime")
#          wk_skip_flg = True
#        else:
#          wk_del_flg = False
#          for lkey in link_of_p:
#            if not wk_graphs[pd.underlay].has_edge(lkey[0],lkey[1],lkey):
#              wk_del_flg = True
#              break
#          if wk_del_flg == False:
#            G_LOG.info("[COMPUTE] skip no link deleted")
#            wk_skip_flg = True
#
#    # opt/new
#    else:
#      if pd.name in G_PATH.keys():
#        if G_PATH[pd.name]["time"] == wk_graphs_t[pd.underlay]:
#          wk_skip_flg = True
#
#    # check skip or not
#    if wk_skip_flg == True:
#      G_LOG.info("[COMPUTE] compute skip")
#      if pd.name in G_PATH.keys():
#        G_PATH[pd.name]["opttime"] = int(time.time() * 1000)
#      continue
#
#    if pd.underlay not in wkG.keys():
#      wkG[pd.underlay]   = remove_link2(wk_graphs[pd.underlay], wk_link_state)
#      rwkG[pd.underlay]  = wkG[pd.underlay].reverse(copy=False)
#
#    #print("GRAPH INFO")
#    #for u, v, data in wkG[pd.underlay].edges(data=True):
#    #        print(u, v, data)
    
#    compute_paths6(pd, wkG[pd.underlay], rwkG[pd.underlay], wk_G_base_t, wk_graphs_t[pd.underlay]) 

#def compute_paths6(pd, wkG, rwkG, wk_G_base_t, wk_Gc_t):
#  G_LOG.info("[COMPUTE] compute6 rtn start " + pd.name)
#
#  global G_LINK_TO_PATH, G_PATH
#
#  wk_results = {}
#
#  #G_LOG.info("[COMPUTE] path name:" + str(pd.name))
#  G_LOG.debug("[COMPUTE] path def:" + str(pd))
#  G_LOG.info("[COMPUTE] compute start for " + pd.name)
#
#  p, link_set  = build_paths(pd, wkG, rwkG)
#
#  G_LOG.info("[COMPUTE] compute end for" + pd.name)
#  G_LOG.debug("[COMPUTE] calc results")
#  G_LOG.debug("[COMPUTE] " + str(p))
#
#  #try: 
#  wk_results["time"]      = wk_Gc_t
#  wk_results["opttime"]   = int(time.time() * 1000)
#  wk_results["underlay"]  = pd.underlay
#  #wk_results[pname]["time"]     = comptime
#  wk_results["detail"]   = p
#  wk_results["link_set"] = link_set
#
#  # time check , NW change during 
#  #if wk_G_base_t != G_G_base_t:
#  last_g_time  = G_GM.get_last_g_time()
#  if wk_G_base_t != last_g_time:
#    G_LOG.info("[COMPUTE] NW change happend during compute")
#    return 
#
#  # delete path
#  if pd.name in G_PATH.keys():
#    delete_path(pd.name)
#  #if pd.name in G_PATH.keys():
#  #  for wk_l in G_PATH[pd.name]["link_set"]:
#  #    if wk_l in G_LINK_TO_PATH.keys():
#  #      G_LINK_TO_PATH[wk_l].discard(pd.name)
#
#  # add
#  for wk_l in wk_results["link_set"]:
#    if wk_l not in G_LINK_TO_PATH.keys():
#      G_LINK_TO_PATH[wk_l] = set()
#    G_LINK_TO_PATH[wk_l].add(pd.name)
#
#  if pd.name not in G_PATH.keys():
#    G_PATH[pd.name] = {}
#  G_PATH[pd.name] = wk_results
#
#  #print("G_PATH")
#  #print(G_PATH)
#  #print("link_to_X")
#  #print(G_LINK_TO_PATH)
#  #print("link_state")
#  #print(G_GM.linkstate)
#
#  return True  

def recompute_path_for_change():

  G_LOG.info("[COMPUTE] recompute path check start")

  #wk_graphs=G_GM.get_all_graphs()
  pathinfo = get_G_PATHINFO()
  #normalized = [
  #  pd
  #  for name, raw in pathinfo.items()
  #  if (pd := normalize_pathdef(name, raw, "ALL")) is not None
  #]
  normalized = []
  for name, raw in pathinfo.items():
    normalized.append(normalize_pathdef(name, raw))

  paths_sorted = sorted(
    normalized,
    key=lambda p: (-p.pri, -p.bw)
  ) 
  wk_G = {}
  #wk_skip_flg = False
  for pd in paths_sorted:
    wk_skip_flg = False
    G_LOG.info("[COMPUTE] " + str(pd.name))
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
        G_LOG.info("[COMPUTE] skip no link deleted a")
        wk_skip_flg = True
    if wk_skip_flg:
      G_LOG.info("[COMPUTE] compute skip")
      if pd.name in G_PATH.keys():
        G_PATH[pd.name]["opttime"] = int(time.time() * 1000)
      continue
    else:
      G_LOG.info("[COMPUTE] compute exec")
      qpri = ( 50, get_G_C_cnt() )
      G_C_Queue.put((qpri,{
        "type": "RECOMPUTEX",
        "comptype": pd.name
      }))

def check_path_optm():

  while True:
    pathinfo = get_G_PATHINFO()

    for k in G_PATH.keys():
      now      = int(time.time() * 1000)
      calctime = int(G_PATH[k]["opttime"])
      optmtime = pathinfo[k].get("optm")
      if optmtime != None:
        optmtime = max(optmtime * 1000, 10000)
        if now - calctime > optmtime:
          #qpri = ( 100, now )
          qpri = ( 100, get_G_C_cnt())
          print("qpri")
          print(qpri)
          G_C_Queue.put((qpri,{
            #"type": "RECOMPUTE4",
            "type": "RECOMPUTEX",
            "comptype": k,
          }))

    time.sleep(3.0)


def handle_event(ev):

  global G_PATH

  ev_time = int(time.time() * 1000)

  ev_t = ev["type"]
  G_LOG.info("[COMPUTE] event start "  + ev_t)

  if ev_t == "PATH_CONFIG":
    cid   = ev["diff"]["id"]
    difft = ev["diff"]["type"]
    if ( difft == DiffType.DEL ):
      #G_PATH.pop(cid)
      #qpri = (100, ev_time)
      qpri = ( 100, get_G_C_cnt())
      G_C_Queue.put((qpri,{
        "type": "PATHDEL",
        "pathid": cid,
      }))
      # delete path from nw(pcep)
    elif ( difft == DiffType.ADD ):
      #qpri = (100, ev_time)
      qpri = ( 100, get_G_C_cnt())
      G_C_Queue.put((qpri,{
        #"type": "RECOMPUTE4",
        "type": "RECOMPUTEX",
        "comptype": cid,
      }))
    elif ( difft == DiffType.MOD ):
      #qpri = (100, ev_time)
      qpri = ( 100, get_G_C_cnt())
      G_C_Queue.put((qpri,{
        #"type": "RECOMPUTE4",
        "type": "RECOMPUTEX",
        "comptype": cid,
      }))

  # currently notsupported
  elif ev_t == "NODE_CONFIG":
    pass
    #difft = ev["diff"]["type"]
    #if ( difft == DiffType.DEL ): # nothing to do
    #if ( difft == DiffType.ADD ): 
    #  if ev["diff"]["new"]["role"] == "PE":
    #if ( difft == DiffType.MOD ): # nothing to do
    #    if "PE" in ( ev["diff"]["new"]["role"], ev["diff"]["old"]["role"] ): 

  #elif ev_t == "RECOMPUTE4":
  #  #compute_paths4(comptype=ev["comptype"])
  #  compute_paths5(comptype=ev["comptype"])

  elif ev_t == "RECOMPUTEX":
    #compute_paths4(comptype=ev["comptype"])
    compute_pathsX(ev["comptype"])

  elif ev_t == "PATHDEL":
    delete_path(ev["pathid"])

def delete_path(pathid):
  # PCEP
  

  # delete from linktopath
  if pathid in G_PATH.keys():
    for wk_l in G_PATH[pathid]["link_set"]:
      if wk_l in G_LINK_TO_PATH.keys():
        G_LINK_TO_PATH[wk_l].discard(pathid)

  G_PATH.pop(pathid)
  pass

#-------------------------------------------
# compute
#-------------------------------------------
# remove link not up
def remove_link2(G, linkstate):
  #print("remove_link2")
  #print(linkstate)
  wkG = copy.deepcopy(G)
  for k in linkstate:
    #print(str(k) + str(linkstate[k]))
    if linkstate[k]["cstate"] != "up":
      if wkG.has_edge(k[0],k[1],k):
        #print("remove:" + str(linkstate[k]))
        wkG.remove_edge(k[0],k[1],k)
    #else:
    #  if wkG.has_edge(k[0],k[1],k):
    #    
    #    #print("in Graph")
  return wkG

# Get Info
def get_G_CONSTINFO():
  return G_CONSTINFO

def get_one_G_CONSTINFO(pid):
  if pid in G_CONSTINFO.keys():
    return G_CONSTINFO[pid]
  else:
    return None

def get_G_NODEINFO():
  return G_NODEINFO

def get_one_G_NODEINFO(pid):
  if pid in G_NODEINFO.keys():
    return G_NODEINFO[pid]
  else:
    return None

def get_G_PATHINFO():
  return G_PATHINFO

def get_one_G_PATHINFO(pid):
  if pid in G_PATHINFO.keys():
    return G_PATHINFO[pid]
  else:
    return None

def get_G_C_cnt():
  return next(G_C_Cnt)

#-------------------------------------------
# Main
# GLOBAL
# LOAD_INFO
G_CONSTINFO  = {}           #done
G_NODEINFO   = {}           #done
G_PATHINFO   = {}           #done

# LOG
setup_logging(os.path.dirname(os.path.abspath(__file__)))
G_LOG = logging.getLogger()

# GRAPH RELATED
G_LINK_TO_PATH = {} # Which path is on LINK(entry is set)

# Queue
#G_C_Queue                  = queue.Queue() # calc
G_C_Queue                  = queue.PriorityQueue() # calc
G_C_Cnt                    = count()
# Path
G_PATH                     = {}

# GraphManager
G_GM = GraphManager(G_LOG)

# PathManager

def main():

  global G_CONSTINFO, G_PATHINFO, G_NODEINFO, G_C_Queue

  G_LOG.info("Main start")

  ############## INIT load
  bgplsinfo, _               = load_defs("bgpls")
  G_NODEINFO,  nodemtime     = load_defs("node")
  G_CONSTINFO, constmtime    = load_defs("const")
  G_PATHINFO,  pathmtime     = load_defs("path")

  ############## BGP START
  bgp = BgpServer(bgplsinfo)
  bgp.register_main_callback(G_GM.on_bgpls_event)
  bgp.start()

  ############## threads
  # nw change
  th_b = threading.Thread( target=G_GM.check_nw_change,  daemon=True )
  th_b.start()
  
  # link statei check
  th_c = threading.Thread( target=G_GM.check_link_state, daemon=True )
  th_c.start()

  # g q check
  th_d = threading.Thread( target=G_GM.watch_graph_q,    daemon=True )
  th_d.start()

  ############## PCEP START

  ############## PATH START

  ############## file watcher start this change Global
  wn = threading.Thread(
    target=file_watcher,
    args=(G_C_Queue, G_NODEINFO,  nodemtime,  "node",  G_NODEINFO ), daemon=True,
  )
  wn.start()
  wc = threading.Thread(
    target=file_watcher,
    args=(G_GM.G_Queue, G_CONSTINFO, constmtime, "const", G_CONSTINFO ), daemon=True,
  )
  wc.start()
  wp = threading.Thread(
    target=file_watcher,
    args=(G_C_Queue, G_PATHINFO,  pathmtime,  "path",  G_PATHINFO ), daemon=True,
  )
  wp.start()

  ############## path optm
  po = threading.Thread(
    target=check_path_optm, daemon=True,
  )
  po.start()

  ############## Loop start 
  while True:
    pri, ev = G_C_Queue.get()
    #print(pri)
    #print(ev)
    handle_event(ev)

# start
if __name__ == "__main__":
  main()



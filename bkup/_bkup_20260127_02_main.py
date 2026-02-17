# import 

import networkx as nx
import heapq
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
#import pickle
import os
import time
#import tempfile
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Literal, Union
from datetime import datetime
import threading
from protocols.bgpserver import BgpServer
from utils.load   import load_defs,  file_watcher
from utils.diff   import DiffType
import queue

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
  #topo: int

def ecmp_paths(results, K):
    if not results:
        return []
    best = results[0]["cost"]
    ecmp = [r for r in results if r["cost"] == best]
    return ecmp[:K]

def k_paths(results, K):
    return results[:K]

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

    s1 = datetime.now()

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
          #if u not in parent2:
            # tree に接続できない dest
          #  raise RuntimeError(f"dest {u} is unreachable from current tree")
          #print(u)
          pi, key = parent2[u]
          path.append((pi, u, key))
          u = pi

        path.reverse()

        for e in path:
          tree_edges.add(e)
          tree_nodes.add(e[0])
          tree_nodes.add(e[1])

      #print("tree_edge")
      #print(tree_edges)
      #print(tree_nodes)
      for e in tree_edges:
        link_set.add(e[2])
      

      tree = tree_edges_to_json(tree_edges, src, Gc)
        #print(type(tree))
      #print(tree)

      results = tree


    s2 = datetime.now()

    print("comp for " + pd.name + " start" + str(s1))
    print("comp for " + pd.name + " end  " + str(s2))

    return results, link_set


def tree_edges_to_json(tree_edges, src, G):
    adj = build_adj(tree_edges,G)
    return {src: build_tree_json(adj, src)}

def build_tree_json(adj, u, parent=None):
    node  = {}
    node["children"]  = []

    #for v, key, cost in adj[u]:
    for v, key in adj[u]:
        if v == parent:
            continue

        #node[v] = {
        #    "link" : key,
        #    #"link_attr": cost,
        #    **build_tree_json(adj, v, u)
        #}
        #node["children"] = {
        #    "node" : v,
        #    "link" : key,
        #    #"link_attr": cost,
        #    **build_tree_json(adj, v, u)
        #}
        a = build_tree_json(adj, v, u)
        if a == None:
          node["children"].append({
              "node" : v,
              "link" : key,
              #"link_attr": cost,
              #**build_tree_json(adj, v, u)
          })
        else:
          node["children"].append({
              "node" : v,
              "link" : key,
              #"link_attr": cost,
              #**build_tree_json(adj, v, u)
              **a
          })
    
    if node["children"] == []:
      return None

    return node

def build_adj(tree_edges,G):
    adj = defaultdict(list)
    for u, v, key in tree_edges:
        #data = G[u][v][key]
        #cost = data.get("cost")
        #adj[u].append((v, key, copy.deepcopy(data)))
        #adj[v].append((u, key, copy.deepcopy(data)))
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

#def task(args):
def task(pd: PathDea, G, rG):
  #global presults
  print("task start for " + pd.name + " " + str(datetime.now()))

  try:
    comptime = str(int(time.time() * 1000))
    results, link_set  = build_paths(pd, G, rG)
    #presults[pd.name] = results
    #print("presults")
    #print(presults)
    print("task end for " + pd.name + " " + str(datetime.now()))
    return results, pd.name, comptime, link_set

  except Exception as e:
    raise RuntimeError(
      f"path={pd.name}, type={pd.type}, underlay={pd.underlay}"
    ) from e

  #print("task end for " + pd.name + " " + str(datetime.now()))

#### normalize
def normalize_pathdef(name: str, raw: dict, comptype="ALL") -> PathDea:
  
  global G_NODEINFO

  src_set = []
  dst_set = []
  nodes   = G_NODEINFO

  #print(G_NODEINFO)

  # remcomp_check
  if comptype == "PE": # PE CHANGE
    if "role:PE" not in ( raw["src"], raw["dst"] ):
      return
  elif comptype not in ( "ALL", "NWCHANGE"):
    if name != comptype:
      return

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

    #src_set = raw["src"]
    #if (len(src_set) == 1):
    #  src = src_set[0]
    #  if raw["dst"] == "role:PE":
    #    for n in nodes.keys():
    #      if nodes[n]["role"] == "PE":
    #        dst_set.append(n)
    #  elseini:
    #    dst_set = raw["dst"]

  return PathDea(
    name=name,
    type=raw["type"],
    underlay=raw["underlay"],
    src=src_set,
    dst=dst_set,
    mode=mode,
    K=K,
    delta=delta,
    #topo=current
  )

def remove_link(G):
  wkG = G.copy()
  #wkG = G
  for k in link_state:
    if link_state[k]["cstate"] != "up":
      if wkG.has_edge(k[0],k[1],k):
        wkG.remove_edge(k[0],k[1],k)
  return wkG


def compute_paths4(comptype="ALL"):
  print("compute4")
  global G_bkup_Gc, G_bkup_Gc_t, G_bkup_G_base, bk_changed_links, G_LINK_TO_PATH, presults, G_NODEINFO
  print(G_NODEINFO)

  # get G/linkstatte/modlinks
  wk_G_base        = copy.deepcopy(G_snap_G_base)
  wk_G_base_t      = copy.deepcopy(G_snap_G_base_t)
  wk_graphs        = copy.deepcopy(G_snap_Gc)
  wk_graphs_t      = copy.deepcopy(G_snap_Gc_t)
  wk_link_state    = copy.deepcopy(G_snap_LINKSTATE)

  wk_changed_links = copy.deepcopy(G_snap_CHANGED)
  if bk_changed_links != set():
    wk_changed_links.add(bk_changed_links)

  wk_downlinks    = set()
  wk_newlinks     = set()
  wk_modlinks     = set()
  wk_changed_const= set()
  wk_new_const    = set()

  # bk_G_base == None means INIT all comp
  if G_bkup_G_base == None:
    wk_changed_links = set()


  print("new_mod_const")
  print(wk_new_const)
  print(wk_changed_const)

  # loop changed_links to find change detail.
  if ( comptype == "NWCHANGE" ):
    # check changed const
    for ci in wk_graphs.keys():
      cit = wk_graphs_t[ci] 
      if ci in G_bkup_Gc.keys():
        if cit != G_bkup_Gc_t[ci]:
          wk_changed_const.add(ci)
      else:
        wk_new_const.add(ci)

    for cl in wk_changed_links:
      print(cl)
      if wk_G_base.has_edge(cl[0],cl[1],cl):
        if G_bkup_G_base.has_edge(cl[0],cl[1],cl):
          wk_modlinks.add(cl)
        else:
          wk_newlinks.add(cl)
      else:
        if G_bkup_G_base.has_edge(cl[0],cl[1],cl):
          wk_downlinks.add(cl)
          #pass #withdraw
          #else:
          #   pass # both none

  # pathdef normilze
  normalized = [
    pd
    for name, raw in G_PATHINFO.items()
    #if (pd := normalize_pathdef(nodeinfo, name, raw, comptype)) is not None
    if (pd := normalize_pathdef(name, raw, comptype)) is not None
  ]

  print(G_NODEINFO)
  print("normalized")
  print(normalized)

  wk_results = {}
  for pd in normalized:

    print("path compute check")
    print(pd.name)
    print(pd.underlay)


    if ( comptype == "NWCHANGE" ):
    # check if need recomp
      if pd.underlay not in wk_new_const and pd.underlay not in wk_changed_const:
        continue

      if pd.name in presults.keys():
        print("path in presults")
        if ( comptype == "NWCHANGE" ):
          wk_link_flg = False
          for wl in wk_downlinks:
            print(wl)
            if wl in G_LINK_TO_PATH.keys():
              if pd.name in G_LINK_TO_PATH[wl]:
                wk_link_flg = True
                break
          if wk_link_flg == False:
            continue

    print("submit start for " + pd.name + ":" +str(datetime.now()))
    #wkG   = remove_link(graphs[pd.underlay])
    wkG   = remove_link(G_Gc[pd.underlay])
    rwkG  = wkG.reverse(copy=False)

    p, pname, time, link_set = task(pd, wkG, rwkG)

    try: 
      #if pname not in presults.keys():
      #  presults[pname] = {}
    
      #presults[pname]["time"]   = time
      #presults[pname]["detail"] = p

      wk_results[pname] = {}
      wk_results[pname]["time"]     = time
      wk_results[pname]["detail"]   = p
      wk_results[pname]["link_set"] = link_set


    except Exception as e:
      print("=== PATH COMPUTE ERROR ===")
      print(e)
      import traceback
      traceback.print_exc()
      bk_changed_links = wk_changed_links
      return False

  # p change, link change
  # success
  print("path calc result")
  print(wk_results)
  for wk_r in wk_results.keys():

    if wk_r in presults.keys():
      # delete link_to_P
      for wk_l in presults[wk_r]["link_set"]:
        if wk_l in G_LINK_TO_PATH.keys():
          G_LINK_TO_PATH[wk_l].discard(wk_r)

    for wk_l in wk_results[wk_r]["link_set"]:
      if wk_l not in G_LINK_TO_PATH.keys():
        G_LINK_TO_PATH[wk_l] = set()
      G_LINK_TO_PATH[wk_l].add(wk_r) 
    
    if wk_r not in presults.keys():
      presults[wk_r] = {}

    presults[wk_r] = wk_results[wk_r]

  G_bkup_G_base   = copy.deepcopy(wk_G_base)
  G_bkup_Gc       = copy.deepcopy(wk_graphs)
  G_bkup_Gc_t     = copy.deepcopy(wk_graphs_t)
  bk_changed_links = set()


  print("presults")
  print(presults)
  print("link_to_X")
  print(G_LINK_TO_PATH)
  print("link_state")
  print(link_state)

  return True


# need some define
def make_G(base_G, const, constinfo):

  G_tmp = nx.MultiDiGraph()

  # node
  G_tmp.add_nodes_from(base_G.nodes(data=True))

  # link
  for u, v, k, d in base_G.edges(keys=True, data=True):

    new_d = check_link(d,constinfo)

    if new_d:
      G_tmp.add_edge(u, v, key=k, **new_d)

  return G_tmp


def check_path_optm():
  while True:
    for k in presults.keys():
      now      = int(time.time() * 1000)
      calctime = int(presults[k]["time"])
      optmtime = G_PATHINFO[k].get("optm")
      if optmtime != None:
        optmtime = max(optmtime * 1000, 10000)
        if now - calctime > optmtime:
          G_C_Queue.put({
            "type": "RECOMPUTE4",
            "comptype": k,
          })

    time.sleep(3.0)

def check_link_state():
  global link_state
  while True:
    now     = int(time.time() * 1000)
    for k in link_state.keys():
      if link_state[k]["cstate"] == "uping" :
        if ( now - link_state[k]["statetime"] ) > 2000:
          #print("change linkstate bef")
          #print(link_state)
          link_state[k]["cstate"] = "up"
          #print("change linkstate aft")
          #print(link_state)
      elif link_state[k]["cstate"] == "down":
        if ( now - link_state[k]["statetime"] ) > 3600000:
          #if not G_base.has_edge(k[0],k[1],k):
          #  link_state.pop(k)
          if not G_G_base.has_edge(k[0],k[1],k):
            link_state.pop(k)
            
    time.sleep(1.0)


def handle_event(ev):
  print("HANDLE EVENT")
  print(ev)
  global presults, G_PATHINFO, G_NODEINFO, link_state, changed_links

  if ev["type"] == "PATH_CONFIG":
    G_PATHINFO, _   = load_defs("path")
    cid = ev["diff"]["id"]
    if ( ev["diff"]["type"] == DiffType.DEL ): # just remove
      presults.pop(cid)
    if ( ev["diff"]["type"] == DiffType.ADD ):
      G_C_Queue.put({
        "type": "RECOMPUTE4",
        "comptype": cid,
      })
    if ( ev["diff"]["type"] == DiffType.MOD ):
      G_C_Queue.put({
        "type": "RECOMPUTE4",
        "comptype": cid,
      })
      #pass
  elif ev["type"] == "NODE_CONFIG":
    G_NODEINFO, _   = load_defs("node")
    if ( ev["diff"]["type"] == DiffType.DEL ): # nothing to do
      pass
    if ( ev["diff"]["type"] == DiffType.ADD ): 
      if ev["diff"]["new"]["role"] == "PE":
        pass
    if ( ev["diff"]["type"] == DiffType.MOD ): # nothing to do
      if ev["diff"]["new"]["role"] != ev["diff"]["old"]["role"]:
        if "PE" in ( ev["diff"]["new"]["role"], ev["diff"]["old"]["role"] ): 
          pass
  elif ev["type"] == "RECOMPUTE4":
    compute_paths4(comptype=ev["comptype"])



# Remove links not up from G
def remove_link(G):
  wkG = G.copy()
  for k in link_state:
    if link_state[k]["cstate"] != "up":
      if wkG.has_edge(k[0],k[1],k):
        wkG.remove_edge(k[0],k[1],k)
  return wkG




#-------------------------------------------
#-------------------------------------------
#-------------------------------------------
#-------------------------------------------
#-------------------------------------------
# make Graph
#-------------------------------------------
# check link if is match for const
# if ng return None
def check_link(linkdata, constinfo):

  new_d = dict(linkdata)
  cost = 65535

  if constinfo["metric"]  == "TE":
    cost = linkdata.get("te", 65535)
  else:
    cost = linkdata.get("igp", 65535)

  new_d["cost"] = cost

  return new_d

# make Graph
def make_G(base_G, const, constinfo):

  G_tmp = nx.MultiDiGraph()

  # node simple copy
  G_tmp.add_nodes_from(base_G.nodes(data=True))

  # link check
  for u, v, k, d in base_G.edges(keys=True, data=True):
    new_d = check_link(d,constinfo)
    if new_d:
      G_tmp.add_edge(u, v, key=k, **new_d)

  return G_tmp

#-------------------------------------------
# Trigger NW CHANGE
def check_changed_links():

  # loop 100msec / trigger over 200msec
  while True:
    if len(changed_links) >0 :
      if int(time.time() * 1000) - G_NWCHANGE_T > 200:
        G_G_Queue.put({
          "type": "RECOMPUTE4",
          "comptype": "NWCHANGE",
        })
    time.sleep(0.1)

# BGP_LS EVENT -> to graph queue
def on_bgpls_event(ev):
  global G_G_Queue 
  G_G_Queue.put(ev)

#def graph_ev():
#  global G_G_Queue 
#  while True:
#    ev = G_G_Queue.get()
#    handle_graph_event(ev)

#-------------------------------------------




#-------------------------------------------
# Main
# GLOBAL
# LOAD_INFO
G_CONSTINFO, G_CONSTMTIME  = {}, None
G_NODEINFO,  G_NODEMTIME   = {}, None
G_PATHINFO,  G_PATHMTIME   = {}, None

# GRAPH RELATED
G_LINK_TO_PATH             = {} # Which path is on LINK(entry is set)

G_Gc                       = {}
G_Gc_t                     = {}
G_G_base                   = nx.MultiDiGraph()
G_G_base_t                 = 0

# snapshot
G_snap_G_base              = None
G_snap_G_base_t            = 0
G_snap_Gc_t                = {}
G_snap_Gc                  = {}
G_snap_LINKSTATE           = {}
G_snap_CHANGED             = set()

# bkup
G_bkup_G_base              = None
G_bkup_Gc                  = {}
G_bkup_Gc_t                = {}

# Queue
G_G_Queue                  = queue.Queue() # Graph 
G_C_Queue                  = queue.Queue() # calc

# BGPLSSTATE
G_BGPLS_ACTIVE             = False # true is synced

# time
G_MAXTIME                  = 9999999999999
G_NWCHANGE_T               = G_MAXTIME


graphs      = {}
graphs_t    = {}
G_base      = nx.MultiDiGraph()
#s_G_base    = None
presults    = {}

link_state   = {} #up/down, bw

#ev_queue   = queue.Queue()

changed_links = set()
bk_changed_links = set()



def main():

  def graph_ev():
    while True:
      ev = G_G_Queue.get()
      handle_graph_event(ev)

  def handle_graph_event(ev):
    ev_time = int(time.time() * 1000)
    global   graphs, G_CONSTINFO, presults, changed_links, G_NWCHANGE_T, G_snap_CHANGED, G_snap_Gc, G_snap_Gc_t
    global   G_snap_G_base,G_snap_LINKSTATE, G_BGPLS_ACTIVE, G_C_Queue, G_snap_G_base_t
    global   G_Gc, G_Gc_t, G_G_base, G_G_base_t
    print(f"[main] bgpls ev {ev}")
    t = ev["type"]

    if t == "BGPLS_STATE":

      if ev["state"] == "SYNCING":
        G_BGPLS_ACTIVE = False

      elif ev["state"] == "ACTIVE":
        # initial
        if G_BGPLS_ACTIVE == False:
          # make G
          for const in G_CONSTINFO.keys():
            #graphs[const]  = make_G(G_base,const,G_CONSTINFO[const])
            #graphs_t[const]= ev_time
            G_Gc[const]  = make_G(G_G_base, const, G_CONSTINFO[const])
            G_Gc_t[const]= ev_time

          presults = {}
          G_G_Queue.put({
            "type": "RECOMPUTE4",
            "comptype": "NWCHANGE",
          })

          G_G_base_t = ev_time

        G_BGPLS_ACTIVE = True

    elif t.startswith("LS_"):
      handle_ls(ev,ev_time)

    #elif t.startswith("LS_"):
    elif t == "CONST_CONFIG":
      G_CONSTINFO, _   = load_defs("const")
      cid              = ev["diff"]["id"]
      if ( ev["diff"]["type"] == DiffType.DEL ): # Just Del
        if cid in graphs.keys():
          graphs.pop(cid)
        if cid in G_Gc.keys():
          G_Gc.pop(cid)

      elif ( ev["diff"]["type"] == DiffType.ADD ):
        if G_BGPLS_ACTIVE == True:
          #graphs[cid]   = make_G(G_base,cid,ev["diff"]["new"])
          #graphs_t[cid] = ev_time
          G_Gc[cid]   = make_G(G_G_base,cid,ev["diff"]["new"])
          G_Gc_t[cid] = ev_time

    elif t == "RECOMPUTE4":
      ctype = ev["comptype"]
      # copy to snapshot
      #G_snap_G_base    = copy.deepcopy(G_base)
      G_snap_G_base    = copy.deepcopy(G_G_base)
      G_snap_G_base_t  = G_G_base_t
      #G_snap_Gc        = copy.deepcopy(graphs)
      #G_snap_Gc_t      = copy.deepcopy(graphs_t)
      G_snap_Gc        = copy.deepcopy(G_Gc)
      G_snap_Gc_t      = copy.deepcopy(G_Gc_t)
      G_snap_LINKSTATE = copy.deepcopy(link_state)
      G_snap_CHANGED   = copy.deepcopy(changed_links)
      #print(changed_links)
      #changed_links  = set() 
      #
      changed_links  = set() 
      G_NWCHANGE_T   = G_MAXTIME
      G_C_Queue.put({
        "type": "RECOMPUTE4",
        "comptype": ctype,
      })

  def handle_ls(ev, ev_time):
    nlri = ev["nlri"]
    nlri_type = nlri["nlri_type"]
    if nlri_type == 1:        
      handle_node(ev,ev_time)
    elif nlri_type == 2:      
      handle_link(ev,ev_time)

  def handle_node(ev, ev_time):

    global   graphs, G_BGPLS_ACTIVE, G_Gc, G_G_base_t, G_G_base

    nlri    = ev["nlri"]
    d       = nlri["detail"]
    node_id = d["local_node"]["ipv4_router_id"]

    # withdraw
    if ev["type"] == "LS_WITHDRAW":

      #if G_base.has_node(node_id):
      #  G_base.remove_node(node_id)
      if G_G_base.has_node(node_id):
        G_G_base.remove_node(node_id)

      if G_BGPLS_ACTIVE == True:
        for const in G_CONSTINFO.keys():
          if const in graphs.keys():
            if graphs[const].has_node(node_id):
              graphs[const].remove_node(node_id)
          if const in G_Gc.keys():
            if G_Gc[const].has_node(node_id):
              G_Gc[const].remove_node(node_id)

      return

    # update/add
    #G_base.add_node(node_id,   nlri=d)
    G_G_base.add_node(node_id,   nlri=d)

    if G_BGPLS_ACTIVE == True:
      for const in G_CONSTINFO.keys():
        if const in graphs.keys():
          graphs[const].add_node(node_id, nlri=d)
        if const in G_Gc.keys():
          G_Gc[const].add_node(node_id, nlri=d)
 
  def handle_link(ev,ev_time):
    global   G_NWCHANGE_T, changed_links, G_BGPLS_ACTIVE, G_Gc, G_Gc_t
    
    nlri = ev["nlri"]
    key  = ev["key"]
    d = nlri["detail"]
    lsattr = ev["ls_attrs"]

    # None is ignore
    if None not in key:

      src = d["local_node"]["ipv4_router_id"]
      dst = d["remote_node"]["ipv4_router_id"]

      if ev["type"] == "LS_WITHDRAW":
            #links.pop(key, None)
          #if G_base.has_edge(src, dst, key):
          #    G_base.remove_edge(src, dst, key)
          if G_G_base.has_edge(src, dst, key):
              G_G_base.remove_edge(src, dst, key)
          if ev["key"] not in link_state:
            link_state[ev["key"]] = {}
          link_state[ev["key"]]["state"]  = "down"
          link_state[ev["key"]]["cstate"] = "down"
          link_state[ev["key"]]["statetime"] = ev_time

          #print(link_state)

          if G_BGPLS_ACTIVE == True:
              for const in G_CONSTINFO.keys():
                if graphs[const].has_edge(src,dst,key):
                  graphs[const].remove_edge(src,dst,key)
                  graphs_t[const] = ev_time
                if G_Gc[const].has_edge(src,dst,key):
                  G_Gc[const].remove_edge(src,dst,key)
                  G_Gc_t[const] = ev_time
              
              changed_links.add(key)
              G_NWCHANGE_T=min(G_NWCHANGE_T, ev_time)

          G_G_base_t = ev_time

          return

        # ADD / UPDATE
        #links[key] = d
        #G.add_edge(src, dst)
      if "igp_metric" in lsattr:
        #G_base.add_edge(src, dst, key=key, 
        #  igp=lsattr.get("igp_metric",65535), te=lsattr.get("te_metric",65535))
        G_G_base.add_edge(src, dst, key=key, 
          igp=lsattr.get("igp_metric",65535), te=lsattr.get("te_metric",65535))

        if ev["key"] in link_state:
          link_state[ev["key"]]["state"]  = "up"
          link_state[ev["key"]]["statetime"] = ev_time
          link_state[ev["key"]]["cstate"] = "uping"
        else:
          link_state[ev["key"]] = {}
          if ( G_BGPLS_ACTIVE == True ):
            link_state[ev["key"]]["state"]  = "up"
            link_state[ev["key"]]["statetime"] = ev_time
            link_state[ev["key"]]["cstate"] = "uping" # INIT
          else:
            link_state[ev["key"]]["state"]  = "up"
            link_state[ev["key"]]["statetime"] = ev_time
            link_state[ev["key"]]["cstate"] = "up" # INIT

        if G_BGPLS_ACTIVE == True:
            for const in G_CONSTINFO.keys():
              graphs[const].add_edge(src, dst, key=key, 
                igp=lsattr.get("igp_metric",65535), te=lsattr.get("te_metric",65535))
              graphs_t[const] = ev_time
              G_Gc[const].add_edge(src, dst, key=key, 
                igp=lsattr.get("igp_metric",65535), te=lsattr.get("te_metric",65535))
              G_Gc_t[const] = ev_time

            changed_links.add(key)
            G_NWCHANGE_T=min(G_NWCHANGE_T, ev_time)

        G_G_base_t = ev_time


  ############## main
  global G_CONSTINFO, G_PATHINFO, G_NODEINFO, G_C_Queue, G_G_Queue


  ############## INIT load
  bgplsinfo, _               = load_defs("bgpls")
  G_NODEINFO,  G_NODEMTIME   = load_defs("node")
  G_CONSTINFO, G_CONSTMTIME  = load_defs("const")
  G_PATHINFO,  G_PATHMTIME   = load_defs("path")

  ############## BGP START
  bgp = BgpServer(bgplsinfo)
  bgp.register_main_callback(on_bgpls_event)
  bgp.start()

  ############## watch graph ev
  gt = threading.Thread( target=graph_ev, daemon=True )
  gt.start()

  ############## check nw change thread
  thread_nwchg = threading.Thread( target=check_changed_links, daemon=True )
  thread_nwchg.start()

  ############## PCEP START

  ############## PATH START

  ############## file watcher start
  wn = threading.Thread(
    target=file_watcher, args=(G_C_Queue, G_NODEINFO, G_NODEMTIME, "node" ), daemon=True,
  )
  wn.start()
  wc = threading.Thread(
    target=file_watcher, args=(G_G_Queue, G_CONSTINFO, G_CONSTMTIME, "const" ), daemon=True,
  )
  wc.start()
  wp = threading.Thread(
    target=file_watcher, args=(G_C_Queue, G_PATHINFO, G_PATHMTIME, "path" ), daemon=True,
  )
  wp.start()

  ############## linkstatecheck
  ls = threading.Thread(
    target=check_link_state, daemon=True,
  )
  ls.start()

  ############## path optm
  po = threading.Thread(
    target=check_path_optm, daemon=True,
  )
  po.start()


  ############## Loop start 
  while True:
    ev = G_C_Queue.get()
    handle_event(ev)

# start
if __name__ == "__main__":
  main()



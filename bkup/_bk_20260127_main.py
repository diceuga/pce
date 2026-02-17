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

WORKER_GRAPHS = {}

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


#def init_worker(current):
#    
#    print("init worker start " + str(datetime.now()))
#    WORKER_GRAPHS.clear()
#    #base = f"G/{current}"
#    #for fname in os.listdir(base):
#    #  if fname.endswith(".pkl"):
#    #    cid = fname[:-4]
#    #    with open(os.path.join(base, fname), "rb") as f:
#    #      G, meta = pickle.load(f)
#    #    WORKER_GRAPHS[cid] = (G, G.aeverse(copy=False), meta)
#    for g in graphs.keys():
#      #if fname.endswith(".pkl"):
#      #  cid = fname[:-4]
#      #  with open(os.path.join(base, fname), "rb") as f:
#      #    G, meta = pickle.load(f)
#      WORKER_GRAPHS[g] = (graphs[g], graphs[g].reverse(copy=False), "")
#
#    print("init worker end " + str(datetime.now()))

def ecmp_paths(results, K):
    if not results:
        return []
    best = results[0]["cost"]
    ecmp = [r for r in results if r["cost"] == best]
    return ecmp[:K]

def k_paths(results, K):
    return results[:K]

#def expand_links(G, links):
#    expanded = []
#    for u, v, k in links:
#        data = G[u][v][k]
#        expanded.append({
#            #"u": u,
#            #"v": v,
#            "key": k,
#            "attr": data
#        })
#    return expanded

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
            #links.append({"key": k ,"attr": copy.deepcopy(data)})
            #links.append((u,v,k))
            links.append(k)

            dfs(
                v,
                cost + data["cost"],
                path,
                links,
                visited
            )

            links.pop()        # ★ 必須
            path.pop()
            visited.remove(v) # ★ 必須

    dfs(src, 0, [], [], {src})

    results.sort(key=lambda x: (x["cost"], len(x["path"])))

    if ( mode == "ecmp" ):
      R = ecmp_paths(results, K)
    else:
      R = k_paths(results,K)

    return R


#def build_paths(nodes, current_hash, p, pathdef, now, current):
#def build_paths(current_hash, pd: PathDea, now, current):
def build_paths(pd: PathDea, G, rG):

    s1 = datetime.now()
    sum_links = []

    #with open("G/" + str(current) + "/" +  pd.underlay + ".pkl", "rb") as f:
      #G, meta = pickle.load(f)
    Gc     = G
    Gc_rev = rG
    #Gc, Gc_rev, meta = WORKER_GRAPHS[pd.underlay]
    #print(G)

    #Gc     = G
    #Gc_rev = Gc.reverse(copy=False)

    results={}
    link_set=set()
    src_set = []
    dst_set = []

    if ( pd.type == "p2p" ): 
      for src in pd.src:
        results[src] = {}

        dist_src = nx.single_source_dijkstra_path_length( Gc, src, weight="cost" )
        #best = dist_src[src]

        for dst in pd.dst:

          if src == dst:
            continue

          best = dist_src.get(dst)
          if best is None:
            continue

          #paths = []
          dist_dst = nx.single_source_dijkstra_path_length( Gc_rev, dst, weight="cost" )
          paths = k_physical_paths_visited(
            Gc, src, dst, pd.K, pd.delta, dist_dst, best, pd.mode
          )

          results[src][dst]   = paths
          #link_to_p2mp[pd.name]
          #link_to_p2mp[(pd.name, src, dst)] = 
          for r in paths:
            for r2 in r["links"]:
              if r2 not in link_to_p2p:
                link_to_p2p[r2] = []

              #link_to_p2p[r2].append((pd.name, src, dst))
              link_to_p2p[r2].append(pd.name)
              link_set.add(r2)
    else:
    # p2mp
      #src_set = pathdef["src"]

      #if (len(src_set) == 1):
      #  src = src_set [0]
      #  if pathdef["dst"] == "role:PE":
      #    for n in nodes.keys():
      #      if nodes[n]["role"] == "PE":
      #        dst_set.append(n)
      #  else:
      #    dst_set = pathdef["dst"]

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

        # d から tree にぶつかるまで戻る
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
      #    for r in paths:
      #      for r2 in r["links"]:
        if e[2] not in link_to_p2mp:
          link_to_p2mp[e[2]] = []

        link_to_p2mp[e[2]].append((pd.name))
        link_set.add(e[2])
      

      tree = tree_edges_to_json(tree_edges, src, Gc)
        #print(type(tree))
      #print(tree)

      results = tree

    #with open("P/" + now + "/" + pd.name + ".pkl","wb") as f:
    #  #pickle.dump((results, current_hash), f)    
    #  pickle.dump((results, pd), f)    

    s2 = datetime.now()
    #print("-------------------------")
    #print(pd.name)
    print("comp for " + pd.name + " start" + str(s1))
    #print(s2)
    print("comp for " + pd.name + " end  " + str(s2))
    #print("comp end" + str(s1))
    #print(results)
    #print(pd)
    #print("-------------------------")

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
                if (u, v, key) in tree_edges:
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
            # まだ他の dest があるので continue

        for v, keydict in G[u].items():
            for key, attr in keydict.items():
                w = attr["cost"]   # ← ここがポイント
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
def normalize_pathdef(nodes, name: str, raw: dict, comptype="ALL") -> PathDea:
  
  src_set = []
  dst_set = []

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


# copmpute all
def compute_paths3(nodes, pathdef, comptype="ALL"):

  normalized = [
    pd
    for name, raw in pathdef.items()
    if (pd := normalize_pathdef(nodes, name, raw, comptype)) is not None
  ]

  for pd in normalized:
    print("submit start for " + pd.name + ":" +str(datetime.now()))
    wkG   = remove_link(graphs[pd.underlay])
    rwkG  = wkG.reverse(copy=False)
    p, pname, time, link_set = task(pd, wkG, rwkG)

    try:
      if pname not in presults.keys():
        presults[pname] = {}

      presults[pname]["time"]   = time
      presults[pname]["detail"] = p

    except Exception as e:
      print("=== PATH COMPUTE ERROR ===")
      print(e)
      import traceback
      traceback.print_exc()
      return False

  return True


#def compute_paths4(nodes, pathdef, comptype="ALL"):
def compute_paths4(comptype="ALL"):
  print("compute4")
  print(comptype)
  print(s_changed_links)
  global bk_graphs, bk_graphs_t, bk_G_base, bk_link_state, bk_changed_links, link_to_p, presults

  # get G/linkstatte/modlinks
  wk_G_base        = copy.deepcopy(s_G_base)
  wk_graphs        = copy.deepcopy(s_graphs)
  wk_graphs_t      = copy.deepcopy(s_graphs_t)
  wk_link_state    = copy.deepcopy(s_link_state)
  wk_changed_links = copy.deepcopy(s_changed_links)

  wk_downlinks    = set()
  wk_newlinks     = set()
  wk_modlinks     = set()
  wk_changed_const= set()
  wk_new_const    = set()

  # bk_G_base == None means INIT all comp
  if bk_G_base == None:
    wk_link_state = set()

  # check changed const
  for ci in wk_graphs.keys():
    cit = wk_graphs_t[ci] 
    if ci in bk_graphs.keys():
      if cit != bk_graphs_t[ci]:
        wk_changed_const.add(ci)
    else:
      wk_new_const.add(ci)

  print("new_mod_const")
  print(wk_new_const)
  print(wk_changed_const)

  # loop changed_links to find change detail.
  # bk_graphs = {} is INIT
  if ( comptype == "NWCHANGE" ):
    # check changed const
    for ci in wk_graphs.keys():
      cit = wk_graphs_t[ci] 
      if ci in bk_graphs.keys():
        if cit != bk_graphs_t[ci]:
          wk_changed_const.add(ci)
      else:
        wk_new_const.add(ci)

    for cl in wk_changed_links:
      print(cl)
      if wk_G_base.has_edge(cl[0],cl[1],cl):
        if bk_G_base.has_edge(cl[0],cl[1],cl):
          wk_modelinks.add(cl)
        else:
          wk_newlinks.add(cl)
      else:
        if bk_G_base.has_edge(cl[0],cl[1],cl):
          wk_downlinks.add(cl)
          #pass #withdraw
          #else:
          #   pass # both none

  # pathdef normilze
  normalized = [
    pd
    for name, raw in pathinfo.items()
    if (pd := normalize_pathdef(nodeinfo, name, raw, comptype)) is not None
  ]

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
            if wl in link_to_p2p.keys():
              if pd.name in link_to_p2p[wl]:
                wk_link_flg = True
                break
          if wk_link_flg == False:
            continue


    print("submit start for " + pd.name + ":" +str(datetime.now()))
    wkG   = remove_link(graphs[pd.underlay])
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
      bk_link_state    = None
      #bk_changed_links.add(s_changed_links)
      return False

  # p change, link change
  # success
  print("path calc result")
  print(wk_results)
  for wk_r in wk_results.keys():

    if wk_r in presults.keys():
      # delete link_to_P
      for wk_l in presults[wk_r]["link_set"]:
        if wk_l in link_to_p.keys():
          link_to_p[wk_l].discard(wk_r)

    for wk_l in wk_results[wk_r]["link_set"]:
      if wk_l not in link_to_p.keys():
        link_to_p[wk_l] = set()
      link_to_p[wk_l].add(wk_r) 
    
    if wk_r not in presults.keys():
      presults[wk_r] = {}

    presults[wk_r] = wk_results[wk_r]

  bk_G_base       = copy.deepcopy(wk_G_base)
  bk_graphs       = copy.deepcopy(wk_graphs)
  bk_graphs_t     = copy.deepcopy(wk_graphs_t)
  #wklink_state   = copy.deepcopy(s_link_state)
  bk_link_state   = None
  bk_changed_links = set()


  print("presults")
  print(presults)
  print("link_to_X")
  print(link_to_p)
  #print(link_to_p2mp)
  print("link_state")
  print(link_state)

  return True

def check_changed_links():
  # over 0.2 sec
  
  while True:
    if len(changed_links) >0 :
      if int(time.time() * 1000) - changed_time > 200:
        print("trigger NW CHANGE")
        print(changed_links)
        print(changed_time)
        #ev_queue.put({
        #  "type": "RECOMPUTE4",
        #  "comptype": "NWCHANGE",
        #})
        ev_g_queue.put({
          "type": "RECOMPUTE4",
          "comptype": "NWCHANGE",
        })

    time.sleep(0.1)


def check_link(linkdata, constinfo):

  new_d = dict(linkdata)

  cost = 65535

  if constinfo["metric"]  == "TE":
    cost = linkdata.get("te", 65535)
  else:
    cost = linkdata.get("igp", 65535)

  new_d["cost"] = cost

  return new_d

# need some define
def make_G(base_G, const, constinfo):
  #print(base_G)
  G_tmp = nx.MultiDiGraph()

  # node
  G_tmp.add_nodes_from(base_G.nodes(data=True))

  # link
  for u, v, k, d in base_G.edges(keys=True, data=True):

    new_d = check_link(d,constinfo)

    if new_d:
      G_tmp.add_edge(u, v, key=k, **new_d)

  return G_tmp

def path_optm():
  G_tmp      = None
  G_time_tmp = None

  while True:
    if G_tmp == None:
      G_tmp      = graphs.copy()
      G_time_tmp = graphs_t.copy()
    else:
      pass

    time.sleep(0.2)

def check_path_optm():
  while True:

    for k in presults.keys():
      now      = int(time.time() * 1000)
      calctime = int(presults[k]["time"])
      optmtime = pathinfo[k].get("optm")
      if optmtime != None:
        optmtime = max(optmtime * 1000, 10000)
        if now - calctime > optmtime:
          ev_queue.put({
            "type": "RECOMPUTE4",
            "comptype": k,
          })
          #print("recalc!")
          #print(presults[k]["time"])
          #compute_paths3(nodeinfo, pathinfo,comptype=k )
          #print(presults[k]["time"])

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
          if not G_base.has_edge(k[0],k[1],k):
            link_state.pop(k)
            
      
    time.sleep(1.0)

#def graph_ev():
#  while True:
#    ev = ev_g_queue.get()
#    on_bgpls_event2(ev)

def handle_event(ev):
  print("HANDLE EVENT")
  print(ev)
  global presults, pathinfo, constinfo, nodeinfo, link_state, changed_links
  #print(constinfo)
  #if ev["type"] == "GRAPH_INIT":
  #  
  #  print("init comp start       : " + str(datetime.now()))
  #  now     = str(int(time.time() * 1000))
  #  current = str(int(time.time() * 1000))
  #  #os.mkdir("P/" + now)
  #  presults = {}
  #  #compute_paths4(nodeinfo, pathinfo)
  #  compute_paths3(nodeinfo, pathinfo)
  #  print("init comp end         : " + str(datetime.now()))
  #  #presults = results
  #  print("presults")
  #  print(presults)
  #  print("link_to_X")
  #  print(link_to_p2p)
  #  print(link_to_p2mp)
  #  print("link_state")
  #  print(link_state)


  if ev["type"] == "PATH_CONFIG":
    pathinfo, _   = load_defs("path")
    cid = ev["diff"]["id"]
    if ( ev["diff"]["type"] == DiffType.DEL ): # just remove
      presults.pop(cid)
    if ( ev["diff"]["type"] == DiffType.ADD ):
      print(pathinfo)
      ev_queue.put({
        "type": "RECOMPUTE4",
        "comptype": cid,
      })
      #compute_paths3(nodeinfo, pathinfo, comptype=cid)
      #pass
    if ( ev["diff"]["type"] == DiffType.MOD ):
      #compute_paths3(nodeinfo, pathinfo, comptype=cid)
      ev_queue.put({
        "type": "RECOMPUTE4",
        "comptype": cid,
      })
      #pass
  elif ev["type"] == "NODE_CONFIG":
    nodeinfo, _   = load_defs("node")
    if ( ev["diff"]["type"] == DiffType.DEL ): # nothing to do
      pass
    if ( ev["diff"]["type"] == DiffType.ADD ): 
      if ev["diff"]["new"]["role"] == "PE":
        # recompute with role PE
        compute_paths3(nodeinfo, pathinfo, comptype="PE")
        #presults.update(results)
        #print("presults")
        #print(presults)
        #pass
    if ( ev["diff"]["type"] == DiffType.MOD ): # nothing to do
      if ev["diff"]["new"]["role"] != ev["diff"]["old"]["role"]:
        if "PE" in ( ev["diff"]["new"]["role"], ev["diff"]["old"]["role"] ): 
          compute_paths3(nodeinfo, pathinfo, comptype="PE")
          # recompute with role PE
          #pass
  elif ev["type"] == "RECOMPUTE":
    compute_paths3(nodeinfo, pathinfo, comptype=ev["comptype"])
  elif ev["type"] == "RECOMPUTE4":
    print(changed_links)
    compute_paths4(comptype=ev["comptype"])

def resolve_node(rid):
  for n in nodeinfo.keys():
    if nodeinfo[n].get("rid","") == rid:
      return n
  return ""

def resolve_node_conf(rid):
  for n in nodeinfo.keys():
    if nodeinfo[n]["rid"] == rid:
      return ( n,  nodeinfo[n] )
  return ( "", {} )

#-------------------------------------------
# Main
# GLOBAL
constinfo   = {}
constmtime  = None
nodeinfo    = {}
nodemtime   = None
pathinfo    = {}
pathmtime   = None
graphs      = {}
graphs_t    = {}
bk_graphs   = {}
bk_graphs_t = {}
s_graphs    = {}
s_graphs_t  = {}
G_base      = nx.MultiDiGraph()
s_G_base    = None
bk_G_base   = None
G_base_v    = 0
presults    = {}

node_to_pid = {}
link_state   = {} #up/down, bw
bk_link_state   = {} #up/down, bw
s_link_state   = {} #up/down, bw
link_to_p2p  = {}
link_to_p2mp = {}
link_to_p    = {}

ev_queue   = queue.Queue()
ev_g_queue = queue.Queue()
ev_c_queue = queue.Queue()

changed_links = set()
s_changed_links = set()
bk_changed_links = set()
changed_time  = 9999999999999

def main():

  #global constinfo, pathinfo, nodeinfo

  #bgpls_active = False      # EOR 受信済み
  #G_base = nx.MultiDiGraph()

  # bgpls event
  def on_bgpls_event(ev):
    global ev_g_queue
    print("on_bgpls_event")
    ev_g_queue.put(ev)

  def graph_ev():
    while True:
      ev = ev_g_queue.get()
      on_bgpls_event2(ev)

  def on_bgpls_event2(ev):
    ev_time = int(time.time() * 1000)
    nonlocal bgpls_active
    global   G_base_v, graphs, constinfo, presults, changed_links, changed_time, s_changed_links, s_graphs, s_graphs_t, s_link_state
    global   s_G_base
    #nonlocal bgpls_active, ev_queue
    print(f"[main] bgpls ev {ev}")
    t = ev["type"]

    if t == "BGPLS_STATE":
      if ev["state"] == "SYNCING":
        #if bgpls_active == True:
          # print(G_base.nodes(data=True))
        bgpls_active = False
      elif ev["state"] == "ACTIVE":
        # initial
        if bgpls_active == False:
          #print(G_base.edges(keys=True, data=True))
          #print(G_base.nodes(data=True))
          for const in constinfo.keys():
            graphs[const]  = make_G(G_base,const,constinfo[const])
            graphs_t[const]= ev_time

          #ev_queue.put({
          #  "type": "GRAPH_INIT",
          #  "graph": G_base,
          #})
          print("init comp start       : " + str(datetime.now()))
          #now     = str(int(time.time() * 1000))
          #current = str(int(time.time() * 1000))
          #os.mkdir("P/" + now)
          presults = {}
          #comptype = "INIT"
          #copy_Graph_data(comptype)
          #s_G_base       = copy.deepcopy(G_base)
          #s_graphs       = copy.deepcopy(graphs)
          #s_graphs_t     = copy.deepcopy(graphs_t)
          #s_link_state   = copy.deepcopy(link_state)
          #s_changed_links= copy.deepcopy(changed_links)
          #compute_paths4()
          ev_g_queue.put({
            "type": "RECOMPUTE4",
            "comptype": "NWCHANGE",
          })
          print("init comp end         : " + str(datetime.now()))
          #presults = results
          print("presults")
          print(presults)
          print("link_to_X")
          print(link_to_p2p)
          print(link_to_p2mp)
          print("link_state")
          print(link_state)

          G_base_v = int(time.time() * 1000)
        bgpls_active = True

    elif t.startswith("LS_"):
      handle_ls(ev,ev_time)

    #elif t.startswith("LS_"):
    elif t == "CONST_CONFIG":
      constinfo, _   = load_defs("const")
      if ( ev["diff"]["type"] == DiffType.DEL ): # Just Del
        cid = ev["diff"]["id"]
        #print(graphs.keys())
        if cid in graphs.keys():
          graphs.pop(cid)

      elif ( ev["diff"]["type"] == DiffType.ADD ):
        if bgpls_active == True:
          cid = ev["diff"]["id"]
          #print("CONST, ADD")
          #print(G_base)
          #print(ev["diff"]["new"])
          #if cid in graphs.keys():
          #  graphs.pop(cid)
          graphs[cid]   = make_G(G_base,cid,ev["diff"]["new"])
          graphs_t[cid] = ev_time
          #print(graphs)

    elif t == "RECOMPUTE4":
      ctype = ev["comptype"]
      #print(changed_links)
      if ctype == "INIT":
        s_G_base       = copy.deepcopy(G_base)
        s_graphs       = copy.deepcopy(graphs)
        s_graphs_t     = copy.deepcopy(graphs_t)
        s_link_state   = copy.deepcopy(link_state)
        s_changed_links= copy.deepcopy(changed_links)
        changed_links  = set() 
      elif ctype == "NWCHANGE":
        s_G_base       = copy.deepcopy(G_base)
        s_graphs       = copy.deepcopy(graphs)
        s_graphs_t     = copy.deepcopy(graphs_t)
        s_link_state   = copy.deepcopy(link_state)
        s_changed_links= copy.deepcopy(changed_links)
        changed_links  = set() 
      #print(changed_links)
      #print(s_changed_links)
      #changed_links  = set() 
      changed_time   = 9999999999999
      ev_queue.put({
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

  def handle_node(ev,ev_time):
    #nonlocal bgpls_active, ev_queue
    nonlocal bgpls_active
    global   G_base_v, graphs

    nlri = ev["nlri"]
    d = nlri["detail"]

    node_id   = d["local_node"]["ipv4_router_id"]

    if ev["type"] == "LS_WITHDRAW":
      print(bgpls_active)
      if G_base.has_node(node_id):
        G_base.remove_node(node_id)

      if bgpls_active == True:
        #ev_queue.put({
        #  "type" : "GRAPH_UPDATE",
        #  "type2": "NODE_WITHDRAW",
        #  "key"  : node_id
        #})
        for const in constinfo.keys():
          #if ev["type2"] == "NODE_WITHDRAW":
          if graphs[const].has_node(node_id):
            graphs[const].remove_node(node_id)

      return

    # update/add
    G_base.add_node(node_id,   nlri=d)

    if bgpls_active == True:
      #ev_queue.put({
      #  "type" : "GRAPH_UPDATE",
      #  "type2": "NODE_UPDATE",
      #  "key"  : node_id,
      #  "nlri" : d
      #})
      for const in constinfo.keys():
      #elif ev["type2"] == "NODE_UPDATE":
        #if graphs[const].has_node(node_id):
        #  graphs[const].remove_node(node_id)
        graphs[const].add_node(node_id, nlri=d)
 
  def handle_link(ev,ev_time):
    #nonlocal bgpls_active, ev_queue
    nonlocal bgpls_active
    global   changed_time, changed_links
    
    nlri = ev["nlri"]
    key  = ev["key"]
    d = nlri["detail"]
    lsattr = ev["ls_attrs"]

    #print(key)
    #print(ev["type"])

    if None not in key:
      #if "igp_metric" in lsattr:

      src = d["local_node"]["ipv4_router_id"]
      dst = d["remote_node"]["ipv4_router_id"]
        #src_name = resolve_node(src) 
        #dst_name = resolve_node(dst) 
        #if src_name != "": src = src_name
        #if dst_name != "": dst = dst_name

        #key = (src, dst)
      if ev["type"] == "LS_WITHDRAW":
            #links.pop(key, None)
          if G_base.has_edge(src, dst, key):
              G_base.remove_edge(src, dst, key)
            #print("G_base")
            #print(G_base)
          if ev["key"] not in link_state:
            link_state[ev["key"]] = {}
          link_state[ev["key"]]["state"]  = "down"
          link_state[ev["key"]]["cstate"] = "down"
          link_state[ev["key"]]["statetime"] = ev_time

          #print(link_state)

          if bgpls_active == True:
              #if node_name == "" :
              #ev_queue.put({
              #  "type" : "GRAPH_UPDATE",
              #  "type2": "LINK_WITHDRAW",
              #  "src"  : src,
              #  "dst"  : dst,
              #  "key"  : key
              #})
              for const in constinfo.keys():
                if graphs[const].has_edge(src,dst,key):
                  graphs[const].remove_edge(src,dst,key)
                  graphs_t[const] = ev_time
              
              changed_links.add(key)
              changed_time=min(changed_time, ev_time)

          G_base_v = int(time.time() * 1000)

          return

        # ADD / UPDATE
        #links[key] = d
        #G.add_edge(src, dst)
      if "igp_metric" in lsattr:
        G_base.add_edge(src, dst, key=key, 
          igp=lsattr.get("igp_metric",65535), te=lsattr.get("te_metric",65535))
        #print("G_base")
        #print(G_base)
        if ev["key"] in link_state:
          link_state[ev["key"]]["state"]  = "up"
          link_state[ev["key"]]["statetime"] = ev_time
          link_state[ev["key"]]["cstate"] = "uping"
        else:
          link_state[ev["key"]] = {}
          if ( bgpls_active == True ):
            link_state[ev["key"]]["state"]  = "up"
            link_state[ev["key"]]["statetime"] = ev_time
            link_state[ev["key"]]["cstate"] = "uping" # INIT
          else:
            link_state[ev["key"]]["state"]  = "up"
            link_state[ev["key"]]["statetime"] = ev_time
            link_state[ev["key"]]["cstate"] = "up" # INIT

        if bgpls_active == True:
          #if node_name == "" :
            #ev_queue.put({
            #  "type" : "GRAPH_UPDATE",
            #  "type2": "LINK_UPDATE",
            #  "src"  : src,
            #  "dst"  : dst,
            #  "key"  : key,
            #  "lsattr" : lsattr
            #})
            for const in constinfo.keys():
              #if graphs[const].has_edge(src, dst, key):
              #  graphs[const].remove_edge(src, dst, key)

            # define some condition
            #graphs[const].add_edge(ev["src"],ev["dst"],ev["key"])
              graphs[const].add_edge(src, dst, key=key, 
                igp=lsattr.get("igp_metric",65535), te=lsattr.get("te_metric",65535))
              graphs_t[const] = ev_time

            changed_links.add(key)
            changed_time=min(changed_time, ev_time)

        G_base_v = int(time.time() * 1000)


  ############## main
  global constinfo, pathinfo, nodeinfo, G_base, ev_queue, G_base_v

  bgpls_active = False      # EOR 受信済み

  ############## INIT load
  bgplsinfo, _           = load_defs("bgpls")
  nodeinfo,  nodemtime   = load_defs("node")
  constinfo, constmtime  = load_defs("const")
  pathinfo,  pathmtime   = load_defs("path")

  ############## BGP START
  bgp = BgpServer(bgplsinfo)
  bgp.register_main_callback(on_bgpls_event)
  bgp.start()

  ############## watch graph ev
  gt = threading.Thread( target=graph_ev, daemon=True )
  gt.start()
  ############## nw change graph ev
  nwc = threading.Thread( target=check_changed_links, daemon=True )
  nwc.start()

  ############## PCEP START

  ############## PATH START

  ############## file watcher start
  wn = threading.Thread(
    target=file_watcher, args=(ev_queue, nodeinfo, nodemtime, "node" ), daemon=True,
  )
  wn.start()
  wc = threading.Thread(
    target=file_watcher, args=(ev_g_queue, constinfo, constmtime, "const" ), daemon=True,
  )
  wc.start()
  wp = threading.Thread(
    target=file_watcher, args=(ev_queue, pathinfo, pathmtime, "path" ), daemon=True,
  )
  wp.start()

  ############## linkstatecheck
  ls = threading.Thread(
    target=check_link_state, daemon=True,
  )
  ls.start()

  ############## path optm
  po = threading.Thread(
    #target=check_path_optm, args=(ev_queue), daemon=True,
    target=check_path_optm, daemon=True,
  )
  po.start()


  ############## Loop start 
  while True:
    ev = ev_queue.get()
    handle_event(ev)

# start
if __name__ == "__main__":
  main()



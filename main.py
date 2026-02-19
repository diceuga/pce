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
from manager.bwmanager    import BWManager
from manager.graphmanager import GraphManager
from manager.pathmanager  import PathManager


#------------------------------------------
# PATH RELATED
#------------------------------------------
#def ecmp_paths(results, k):
#    if not results:
#        return []
#    best = results[0]["cost"]
#    ecmp = [r for r in results if r["cost"] == best]
#    return ecmp[:k]
#
#def k_paths(results, k):
#    return results[:k]

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
      R = G_PM.ecmp_paths(results, K)
    else:
      R = G_PM.k_paths(results,K)

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

  #graph / linkinfo:
  ( wkG, wkG_t ) = G_GM.get_one_graph_infos(pd.underlay)
  bef_g_time  = wkG_t

  #bwdiff = 0
  if cpathinfo != None:
    removed_wkG  = G_PM.remove_link(
                     wkG, pd.bw, cpathinfo["bw"], cpathinfo["link_set"]
                   )
  else:
    #new
    removed_wkG  = G_PM.remove_link(
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


#def recompute_path_for_change():
#
#  G_LOG.info("[COMPUTE] recompute path check start")
#
#  normalized = G_CM.get_all_pathdef()
#
#  paths_sorted = sorted(
#    normalized,
#    key=lambda p: (-p.pri, -p.bw)
#  ) 
#
#  wk_G = {}
#  #wk_skip_flg = False
#
#  # Link down check 
#  for pd in paths_sorted:
#    wk_skip_flg = False
#
#    G_LOG.info("[COMPUTE] compute check " + str(pd.name))
#
#    if pd.name in G_PATH.keys():
#      link_of_p     = G_PATH[pd.name]["link_set"]
#      under_of_p    = G_PATH[pd.name]["underlay"]
#      if under_of_p not in wk_G.keys():
#        wk_G[under_of_p] = G_GM.get_one_graph(under_of_p)
#      wk_del_flg = False
#      for lkey in link_of_p:
#        #if not wk_graphs[under_of_p].has_edge(lkey[0],lkey[1],lkey):
#        if not wk_G[under_of_p].has_edge(lkey[0],lkey[1],lkey):
#          wk_del_flg = True
#          break
#      if wk_del_flg == False:
#        #G_LOG.info("[COMPUTE] skip no link deleted")
#        wk_skip_flg = True
#
#    if wk_skip_flg:
#      G_LOG.info("[COMPUTE] compute skip")
#      if pd.name in G_PATH.keys():
#        G_PATH[pd.name]["opttime"] = int(time.time() * 1000)
#      continue
#
#    else:
#      G_LOG.info("[COMPUTE] compute exec")
#      qpri = ( 50, self.PM.get_C_cnt() )
#      #G_C_Queue.put((qpri,{
#      self.PM.C_Queue.put((qpri,{
#        "type": "RECOMPUTEX",
#        "comptype": pd.name
#      }))

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

  G_LOG.debug("[COMPUTE] HANDLE EVENT")

  ev_time = int(time.time() * 1000)

  ev_t = ev["type"]
  G_LOG.debug("[COMPUTE] event start "  + ev_t)

  if ev_t == "PATH_CONFIG":
    cid   = ev["diff"]["id"]
    difft = ev["diff"]["type"]
    if ( difft == DiffType.DEL ):
      pass
    elif ( difft == DiffType.ADD ):
      pass
    elif ( difft == DiffType.MOD ):
      qpri = ( 100, G_PM.C_cnt())
      G_PM.C_Queue.put((qpri,{
        #"type": "RECOMPUTE4",
        "type": "RECOMPUTEX",
        "comptype": cid,
      }))

  elif ev_t == "RECOMPUTEX":
    compute_pathsX(ev["comptype"])

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


#-------------------------------------------
# Main
# LOG
setup_logging(os.path.dirname(os.path.abspath(__file__)))
G_LOG = logging.getLogger()

# Path
G_PATH                     = {}
G_PATH_d                   = {}

# Manager
G_BM = BWManager(G_LOG)     # BWManager
G_GM = GraphManager(G_LOG)  # GraphManager
G_CM = ConfigManager(G_LOG) # ConfigManager
G_PM = PathManager(G_LOG)   # PathManager

# Attach
G_CM.attach_PM(G_PM)
G_CM.attach_GM(G_GM)

G_GM.attach_BM(G_BM)
G_GM.attach_CM(G_CM)
G_GM.attach_PM(G_PM)

G_PM.attach_X(compute_pathsX)
G_PM.attach_BM(G_BM)
G_PM.attach_CM(G_CM)
G_PM.attach_GM(G_GM)

def main():

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
    pri, ev = G_PM.C_Queue.get()
    G_PM.handle_event(ev)

# start
if __name__ == "__main__":
  main()



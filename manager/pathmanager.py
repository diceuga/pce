import queue
import time
from collections import defaultdict
from itertools import count
from utils.diff    import DiffType
import heapq
import networkx as nx
import threading


class PathManager:
  def __init__(self, log):
    self.log = log
    self.C_Queue      = queue.PriorityQueue()
    self.C_Cnt        = count()
    self.P_Queue      = queue

    #self.computeX     = None
    self.PATH         = {}
    self.PATH_d       = {}

    # start Q mon
    self.wq = threading.Thread( target=self.watch_compute_q, daemon=True )
    self.wq.start()

    # start path optm 
    self.wp = threading.Thread( target=self.check_path_optm,  daemon=True )
    self.wp.start()

  def on_pcep_event(self, ev):
    qpri = ( 100, self.get_C_cnt())
    self.C_Queue.put(qpri, ev)

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

  def watch_compute_q(self):
    while True:
      pri, ev = self.C_Queue.get()
      self.handle_event(ev)

  def check_path_optm(self):

    while True:

      time.sleep(3.0)

      bgpls_status = self.GM.get_bgpls_active()

      if bgpls_status == False:
        continue

      # test move p to _d for debug need PCEP
      self.debug_move_p()

      # all path def
      allpathdef = self.CM.get_all_pathdef()
      sorted_pathdef1 = sorted( allpathdef, key=lambda p: (-p.pri, -p.bw) ) # for new
      sorted_pathdef2 = sorted( allpathdef, key=lambda p: (p.pri, -p.bw)  ) # for bw

      # del not sync pcep
      #-----------------------------------------
      # need to implement
      for pd in sorted_pathdef1:
        pass

      # del not in path def
      #-----------------------------------------
      for k in list(self.PATH_d.keys()):
        #if self.PATH_d[k]["initiate"] == True:
        wk = self.CM.get_one_pathdef(k)
        if wk == None:
          #  if k not in self.PATH.keys():
          #    self.delete_p_from_d(k)
          qpri = ( 100, self.get_C_cnt())
          self.C_Queue.put((qpri,{
            "type": "DELETE_PATH",
            "comptype": k,
          }))

      # BW check
      #-----------------------------------------
      allbwinfo   = self.BM.getallbw()
      cpath       = set()
      for lk in list(allbwinfo.keys()):
        v = allbwinfo.get(lk)
        if v!= None:
          reducebw = v["sumpbw"] - v["maxbw"]
          #reducebw = 5 # debug
          if ( reducebw > 0  ):
            wkbw = 0
            for pd in sorted_pathdef2:
              if pd.name in self.PATH_d.keys():
                if self.PATH_d[pd.name]["bw"] > 0:
                  if lk in self.PATH_d[pd.name]["link_set"]:
                    cpath.add(pd.name)
                    wkbw += self.PATH_d[pd.name]["bw"]
                    if wkbw > reducebw:
                      break
      for p in cpath:
        qpri = ( 100, self.get_C_cnt())
        self.C_Queue.put((qpri,{
            "type": "RECOMPUTEX",
            "comptype": p,
        }))

      #-----------------------------------------
      # optm
      #-----------------------------------------
      for k in self.PATH_d.keys():
        pathinfo = self.CM.get_one_pathdef(k)
        now      = int(time.time() * 1000)
        calctime = int(self.PATH_d[k]["opttime"])
        optmtime = pathinfo.optm

        if optmtime != 0:
          optmtime = max(optmtime * 1000, 10000)
          if now - calctime > optmtime:
            qpri = ( 100, self.get_C_cnt())
            self.C_Queue.put((qpri,{
              "type": "RECOMPUTEX",
              "comptype": k,
            }))

      #-----------------------------------------
      # New
      #-----------------------------------------
      for pd in sorted_pathdef1:

        # if PCEP, currently skip

        if pd.name not in self.PATH_d.keys():
          qpri = ( 100, self.get_C_cnt())
          self.C_Queue.put((qpri,{
            "type": "RECOMPUTEX",
            "comptype": pd.name,
          }))

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

    normalized = self.CM.get_all_pathdef()
    paths_sorted = sorted(normalized, key=lambda p: (-p.pri, -p.bw))

    wk_G = {}

    # Link down check
    for pd in paths_sorted:
      wk_skip_flg = False

      self.log.info("[COMPUTE] compute check " + str(pd.name))

      if pd.name in self.PATH.keys():
        link_of_p     = self.PATH[pd.name]["link_set"]
        under_of_p    = self.PATH[pd.name]["underlay"]
        if under_of_p not in wk_G.keys():
          wk_G[under_of_p] = self.GM.get_one_graph(under_of_p)
        wk_del_flg = False
        for lkey in link_of_p:
          if not wk_G[under_of_p].has_edge(lkey[0],lkey[1],lkey):
            wk_del_flg = True
            break
        if wk_del_flg == False:
          wk_skip_flg = True

      if wk_skip_flg:
        self.log.info("[COMPUTE] compute skip")
        if pd.name in self.PATH.keys():
          self.PATH[pd.name]["opttime"] = int(time.time() * 1000)
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
        qpri = ( 100, self.C_cnt())
        self.C_Queue.put((qpri,{
          #"type": "RECOMPUTE4",
          "type": "RECOMPUTEX",
          "comptype": cid,
        }))
  
    elif ev_t == "RECOMPUTEX":
      self.compute_pathsX(ev["comptype"])

    elif ev_t == "DELETE_PATH":
      delete_p_from_d(ev["comptype"])

  def delete_p_from_d(self, p):
    if p in list(self.PATH_d.keys()):
      if self.PATH_d[p]["initiate"] == True:
        if p not in self.PATH.keys():
          for lk in self.PATH_d[p]["link_set"]:
            self.BM.delpbw(lk,p)
          self.PATH_d.pop(p)

  def delete_p_from_wk(self, p):
    if p in list(self.PATH.keys()):
      for lk in self.PATH[p]["link_set"]:
        self.BM.delwkpbw(lk,p)
      self.PATH.pop(p)

  def debug_move_p(self):
    for p in list(self.PATH.keys()):
      self.PATH_d[p] = self.PATH[p]

      for lk in list(self.PATH_d[p]["link_set"]):
        self.BM.addpbw(lk,self.PATH_d[p]["bw"],p)

      for lk in list(self.PATH[p]["link_set"]):
        self.BM.delwkpbw(lk,p)

      self.PATH.pop(p)

  #-----------------------------------------------
  # compute
  #-----------------------------------------------
  def compute_pathsX(self, pid):
    self.log.info("[COMPUTE] computeX rtn start for " + pid)

    if self.PATH.get(pid) != None:
      self.log.info("[COMPUTE] skip there is other candidate")
      return

    #cpathinfo  = self.PATH.get(pid)
    cpathinfo  = self.PATH_d.get(pid)
    pd         = self.CM.get_one_pathdef(pid)

    self.log.debug("[COMPUTE] path name: "  + str(pid))
    self.log.debug("[COMPUTE] path def:  "  + str(pd))

    wkG         = None
    wkG_t       = None
    wk_skip_flg = False

    #graph / linkinfo:
    ( wkG, wkG_t ) = self.GM.get_one_graph_infos(pd.underlay)
    bef_g_time  = wkG_t

    #bwdiff = 0
    if cpathinfo != None:
      removed_wkG  = self.remove_link(
                       wkG, pd.bw, cpathinfo["bw"], cpathinfo["link_set"]
                     )
    else:
      #new
      removed_wkG  = self.remove_link(
                       wkG, pd.bw, 0, set()
                     )

    removed_rwkG = removed_wkG.reverse(copy=False)
    wk_results = {}

    self.log.info("[COMPUTE] compute start for " + pd.name)

    p, link_set  = self.build_paths(pd, removed_wkG, removed_rwkG)

    self.log.info("[COMPUTE] compute end for" + pd.name)

    wk_results["time"]      = wkG_t
    wk_results["initiate"]  = True
    wk_results["opttime"]   = int(time.time() * 1000)
    wk_results["underlay"]  = pd.underlay
    wk_results["bw"]        = pd.bw
    wk_results["detail"]    = p
    wk_results["link_set"]  = link_set

    aft_g_time  = self.GM.get_last_g_time()

    if bef_g_time != aft_g_time:
      self.log.info("[COMPUTE] NW change happend during compute")
      return
    else:
      if pd.name in self.PATH_d.keys():
        if ( wk_results["detail"] == self.PATH_d[pd.name]["detail"] ):
          self.log.info("[COMPUTE] calc results same as bef")
          self.PATH_d[pd.name]["opttime"] = int(time.time() * 1000)
          return

      self.log.info("[COMPUTE] calc results")
      self.log.info("[COMPUTE] new path" + str(wk_results))

    # add
    for wk_l in wk_results["link_set"]:
      if cpathinfo != None:
        if wk_l in cpathinfo["link_set"]:
          bwdiff = max(0, wk_results["bw"] - cpathinfo["bw"] )
          self.BM.addwkpbw(wk_l, bwdiff,pd.name)
        else:
          self.BM.addwkpbw(wk_l, wk_results["bw"],pd.name)
      else:
        self.BM.addwkpbw(wk_l, wk_results["bw"],pd.name)

    if pd.name not in self.PATH.keys():
      self.PATH[pd.name] = {}

    self.PATH[pd.name] = wk_results

    return True

  def build_paths(self, pd, Gc, Gc_rev):
    results  = {}
    link_set = set()
    src_set  = []
    dst_set  = []

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
          paths = self.k_physical_paths_visited(
            Gc, src, dst, pd.K, pd.delta, dist_dst, best, pd.mode
          )

          results[src][dst]   = paths
          for r in paths:
            for r2 in r["links"]:
              link_set.add(r2)
    else:
    # p2mp

      src = pd.src[0]
      (dest_dist, parent0) = self.dijkstra_to_dests(Gc, src, pd.dst)
      o_dests = sorted(dest_dist, key=lambda d: dest_dist[d], reverse=True)

      if len(o_dests) == 0:
        return {}

      tree_edges = set()
      tree_nodes = set()

      # base tree ( to farest node )
      first = o_dests[0]
      path = self.build_one_path(parent0, src, first)

      for u, v, key in path:
        tree_edges.add((u, v, key))
        tree_nodes.add(u)
        tree_nodes.add(v)


      # join to tree for 2nd- node
      for d in o_dests[1:]:
        if d in tree_nodes:
          continue

        dist2, parent2 = self.dijkstra_from_tree(Gc, tree_nodes, tree_edges)

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

      tree = self.tree_edges_to_json(tree_edges, src)

      results = tree

    return results, link_set


  def ecmp_paths(self, results, k):
    if not results:
        return []
    best = results[0]["cost"]
    ecmp = [r for r in results if r["cost"] == best]
    return ecmp[:k]

  def k_paths(self, results, k):
    return results[:k]

  def k_physical_paths_visited(self, DG, src, dst, K, delta, dist_dst, best, mode):
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
      R = self.ecmp_paths(results, K)
    else:
      R = self.k_paths(results,K)

    return R

  def dijkstra_to_dests(self, G, src, dests):
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

  def build_one_path(self, parent, src, dest):
    path = []
    u = dest
    while u != src:
        p, key = parent[u]
        path.append((p, u, key))
        u = p
    return list(reversed(path))

  def dijkstra_from_tree(self, G, tree_nodes, tree_edges):

    dist = {}
    parent = {}
    pq = []

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

                #if (u, v, key) in tree_edges: #
                #    #w *= 0.5
                #    w *= 1.0

                new = cost_u + w
                if v not in dist or new < dist[v]:
                    dist[v] = new
                    parent[v] = (u, key)
                    heapq.heappush(pq, (new, v))

    return dist, parent

  def tree_edges_to_json(self, tree_edges, src):
    adj = self.build_adj(tree_edges)
    return {src: self.build_tree_json(adj, src)}

  def build_adj(self, tree_edges):
    adj = defaultdict(list)
    for u, v, key in tree_edges:
        adj[u].append((v, key))
        adj[v].append((u, key))
    return adj

  def build_tree_json(self, adj, u, parent=None):
    node  = {}
    node["children"]  = []

    #for v, key, cost in adj[u]:
    for v, key in adj[u]:
        if v == parent:
            continue

        a = self.build_tree_json(adj, v, u)

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

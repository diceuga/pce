import networkx as nx
import heapq
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import os
import hashlib
import time
import tempfile
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Literal, Union
from datetime import datetime

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
  topo: int


def init_worker(current):
    WORKER_GRAPHS.clear()
    base = f"G/{current}"
    for fname in os.listdir(base):
      if fname.endswith(".pkl"):
        cid = fname[:-4]
        with open(os.path.join(base, fname), "rb") as f:
          G, meta = pickle.load(f)
        WORKER_GRAPHS[cid] = (G, G.reverse(copy=False), meta)

def ecmp_paths(results, K):
    if not results:
        return []
    best = results[0]["cost"]
    ecmp = [r for r in results if r["cost"] == best]
    return ecmp[:K]

def k_paths(results, K):
    return results[:K]

def expand_links(G, links):
    expanded = []
    for u, v, k in links:
        data = G[u][v][k]
        expanded.append({
            #"u": u,
            #"v": v,
            "key": k,
            "attr": data
        })
    return expanded

def k_physical_paths_visited(DG, src, dst, K, delta, dist_dst, best, mode):
    results = []

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
            #for r in results:
            #  r["links"] = expand_links(Gc, r["links"])
            return

        for _, v, k, data in DG.out_edges(u, keys=True, data=True):
            if v in visited:
                continue

            visited.add(v)
            path.append(u)
            #links.append({"key": k ,"attr": copy.deepcopy(data)})
            links.append((u,v,k))

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

    for r in R:
      r["links"] = expand_links(DG, r["links"])

    return R


#def build_paths(nodes, current_hash, p, pathdef, now, current):
#def build_paths(current_hash, pd: PathDea, now, current):
def build_paths(pd: PathDea, now):

    s1 = datetime.now()

    #with open("G/" + str(current) + "/" +  pd.underlay + ".pkl", "rb") as f:
      #G, meta = pickle.load(f)
    Gc, Gc_rev, meta = WORKER_GRAPHS[pd.underlay]
    #print(G)

    #Gc     = G
    #Gc_rev = Gc.reverse(copy=False)

    results={} 
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

          results[src][dst] = paths
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


      #print(tree_edges)
      #print(tree_nodes)

      tree = tree_edges_to_json(tree_edges, src, Gc)
        #print(type(tree))
        #pprint.pprint(tree)

      results = tree


    with open("P/" + now + "/" + pd.name + ".pkl","wb") as f:
      #pickle.dump((results, current_hash), f)    
      pickle.dump((results, pd), f)    

    s2 = datetime.now()
    print("-------------------------")
    print(pd.name)
    print(s1)
    print(s2)
    print(results)
    print(pd)
    print("-------------------------")

        
    return results


def tree_edges_to_json(tree_edges, src,G):
    adj = build_adj(tree_edges,G)
    return {src: build_tree_json(adj, src)}

def build_tree_json(adj, u, parent=None):
    node = {}

    for v, key, cost in adj[u]:
        if v == parent:
            continue

        node[v] = {
            "link_key" : key,
            "link_attr": cost,
            **build_tree_json(adj, v, u)
        }

    return node

def build_adj(tree_edges,G):
    adj = defaultdict(list)
    for u, v, key in tree_edges:
        data = G[u][v][key]
        cost = data.get("cost")
        adj[u].append((v, key, copy.deepcopy(data)))
        adj[v].append((u, key, copy.deepcopy(data)))
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
def task(pd: PathDea, now):
  print("task start for " + pd.name + " " + str(datetime.now()))

  try:
    build_paths(pd, now)
  except Exception as e:
    raise RuntimeError(
      f"path={pd.name}, type={pd.type}, underlay={pd.underlay}"
    ) from e

  print("task end for " + pd.name + " " + str(datetime.now()))

#### normalize
def normalize_pathdef(nodes, current, name: str, raw: dict) -> PathDea:
  src_set = []
  dst_set = []
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
          src_set.append(n)
    else:
      src_set = raw["src"]

    if raw["dst"] == "role:PE":
      for n in nodes.keys():
        if nodes[n]["role"] == "PE":
          dst_set.append(n)
    else:
      dst_set = raw["dst"]

    # p2mp
  else:
    src_set = raw["src"]
    if (len(src_set) == 1):
      src = src_set[0]
      if raw["dst"] == "role:PE":
        for n in nodes.keys():
          if nodes[n]["role"] == "PE":
            dst_set.append(n)
      else:
        dst_set = raw["dst"]

  return PathDea(
    name=name,
    type=raw["type"],
    underlay=raw["underlay"],
    src=src_set,
    dst=dst_set,
    mode=mode,
    K=K,
    delta=delta,
    topo=current
  )

#### compute path
def compute_paths(nodes, pathdef, now, current):

  normalized = [
    normalize_pathdef(nodes, current, name, raw) for name, raw in pathdef.items()
  ]

  #with ProcessPoolExecutor(max_workers=4) as executor:
  with ProcessPoolExecutor(
    initializer=init_worker,
    initargs=(current,)
  ) as executor:

    futures = []
    for pd in normalized:
      print("submit start for " + pd.name + ":" +str(datetime.now()))
      futures.append(executor.submit(task, pd, now))

    for f in as_completed(futures):
      try:
        f.result()
      except Exception as e:
        print("=== PATH COMPUTE ERROR ===")
        print(e)
        import traceback
        traceback.print_exc()
        return False

    return True

#### get file hash
def safe_file_hash(path, retry=5, wait=0.1):
  for i in range(retry):
    try:
      return file_hash(path)
    except FileNotFoundError:
      if i == retry - 1:
        raise
      time.sleep(wait)

def file_hash(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
      h.update(chunk)
  return h.hexdigest()

#### main
def main():

  old_hash = {
    "node_hash" : "",
    "path_hash" : ""
  }
  old_current = 0

  while True:
    #print("main_while_start" + str(datetime.now()))

    with open('G/current') as f:
      current = int(f.read())

    recomp = False

    if current > old_current:
      recomp = True
      #init_worker(current)
        
    current_hash = {
      "node_hash"  : safe_file_hash("etc/node.json"),
      "path_hash"  : safe_file_hash("etc/path.json")
    }

    if current_hash != old_hash:
      recomp = True

      with open('etc/node.json') as f:
        nodes = json.load(f)

      with open('etc/path.json') as f:
        pathdef = json.load(f)

    if recomp == True:
      print("recomp_start       : " + str(datetime.now()))
      now = str(int(time.time() * 1000))
      os.mkdir("P/" + now)
      results = compute_paths(nodes, pathdef, now, current)

      if results:
        dir = os.path.dirname("P/current")
        with tempfile.NamedTemporaryFile("w", dir=dir, delete=False) as f:
          f.write(now)
          tempname = f.name
        os.replace(tempname, "P/current")

    old_hash = current_hash
    old_current = current


    break
    #time.sleep(0.005)

# start
if __name__ == "__main__":
  main()



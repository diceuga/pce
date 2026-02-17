import networkx as nx
import heapq
import datetime
import json
from concurrent.futures import ProcessPoolExecutor

# graph load
def load_topology(path):
    with open(path) as f:
        topo = json.load(f)

    G = nx.MultiDiGraph()
    for n in topo["nodes"].keys():
        G.add_node(n)

    for l in topo["links"]:
        G.add_edge(l["u"], l["v"], key=l["key"], cost=l["cost"])

    N = topo["nodes"]

    return G, N

#def k_physical_paths_visited(G, src, dst, K, delta, dist_src, dist_dst, best):
#    results = []
#
#    def worst_cost():
#        if len(results) < K:
#            return float("inf")
#        return results[-1][0]
#
#    def dfs(u, cost, path, visited):
#        # TE 剪定（ここが一番効く）
#        if cost + dist_dst.get(u, float("inf")) > best + delta:
#            return
#
#        # K 剪定
#        if cost > worst_cost():
#            return
#
#        if u == dst:
#            results.append((cost, path.copy()))
#            results.sort(key=lambda x: x[0])
#            if len(results) > K:
#                results.pop()
#            return
#
#        for _, v, k, data in G.out_edges(u, keys=True, data=True):
#            # ノードループ禁止
#            if v in visited:
#                continue
#
#            visited.add(v)
#            path.append((u, v, k))
#            dfs(v, cost + data["cost"], path, visited)
#            path.pop()
#            visited.remove(v)
#
#    dfs(src, 0, [], {src})
#    return results

def k_physical_paths_visited(DG, src, dst, K, delta, dist_src, dist_dst, best):
    results = []

    def dfs(u, cost, path, links, visited):
        # prune
        if cost + dist_dst.get(u, float("inf")) > best + delta:
            return

        if u == dst:
            results.append({
                "cost": cost,
                "path": path + [u],
                "links": links.copy()
            })
            return

        for _, v, k, data in DG.out_edges(u, keys=True, data=True):
            if v in visited:
                continue

            visited.add(v)
            links.append(k)

            dfs(
                v,
                cost + data["cost"],
                path + [u],
                links,
                visited
            )

            links.pop()        # ★ 必須
            visited.remove(v) # ★ 必須

    dfs(src, 0, [], [], {src})

    results.sort(key=lambda x: x["cost"])
    return results[:K]


def dijkstra_to_dst(G, dst):
    """
    MultiDiGraph G に対して
    各ノードから dst までの最短距離を返す
    """
    dist = {n: float("inf") for n in G.nodes}
    dist[dst] = 0

    pq = [(0, dst)]  # (distance, node)

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue

        # in_edges を使う（逆向き探索）
        for v, _, _, data in G.in_edges(u, keys=True, data=True):
            w = data.get("cost", 0)
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist

# ----------------------------
# Relaxed ECMP DAG 構築
# ----------------------------
def build_relaxed_dag(G, src, dst, delta):
    dist = nx.single_source_dijkstra_path_length(G, src, weight="cost")
    DG = nx.MultiDiGraph()

    dist_dst = dijkstra_to_dst(G, dst)

    for u, v, k, data in G.edges(keys=True, data=True):
        if u not in dist or v not in dist:
            continue
        if u == dst:
            continue
        if (
          #( dist[u] < dist[v] or v == dst )
          ( delta + dist_dst[u] > dist_dst[v] or v == dst )
            and ( dist[u] + data["cost"] + dist_dst[v] <= dist[dst] + delta )
        ):
          DG.add_edge(u, v, key=k, **data)

    return DG, dist

def build_relaxed_dag_te(G, src, dst, delta):
    dist_src = nx.single_source_dijkstra_path_length(G, src, weight="cost")
    dist_dst = dijkstra_to_dst(G, dst)

    #shortest = dist_src.get(dst, float("inf"))
    #DG = nx.MultiDiGraph()

    #if dst not in dist_src:
    #    return DG, dist_src, dist_dst

    shortest = dist_src[dst]
    if shortest is None:
        return nx.MultiDiGraph(), dist_src, dist_dst

    DG = nx.MultiDiGraph()

    for u, v, k, data in G.edges(keys=True, data=True):
        if u not in dist_src or v not in dist_dst:
            continue

        cost = data["cost"]
        #if dist_dst[u] <= dist_dst[v]:
        #  continue

        # TE 制約
        if dist_src[u] + cost + dist_dst[v] > shortest + delta:
            continue

        # potential による DAG 制約（超重要）
        #if dist_src[u] < dist_src[v] or v == dst:
        DG.add_edge(u, v, key=k, **data)

    return DG, dist_src, dist_dst


#def build_relaxed_dag_to_dst(G, dst, delta):
#    dist = dijkstra_to_dst(G, dst)  # 逆向き Dijkstra
#    DG = nx.MultiDiGraph()
#
#    for u, v, k, data in G.edges(keys=True, data=True):
#        if u not in dist or v not in dist:
#            continue
#
#        if (
#            dist[u] > dist[v] and
#            dist[u] >= dist[v] + data["cost"] - delta
#        ):
#            DG.add_edge(u, v, key=k, **data)
#
#    return DG, dist


# ----------------------------
# DAG 上で K 本の physical path
# ----------------------------
def k_physical_paths2(DG, src, dst, K):
    heap = []

    #def worst_cost():
    #    if len(heap) < K:
    #        return float("inf")
    #    return heap[-1][0]   # heapは後でソートする前提
      
    def dfs(node, path, cost):
        #if cost > worst_cost():
        #    return

        if node == dst:
            heapq.heappush(heap, (cost, path.copy()))
            #heap.append((cost, path.copy()))
            #heap.sort(key=lambda x: x[0])
            #if len(heap) > K:
            #    heap.pop()   # 一番悪いものを捨てる
            return

        for _, v, k, data in DG.out_edges(node, keys=True, data=True):
            # DAG なのでノードループチェックは不要
            path.append((node, v, k))
            dfs(v, path, cost + data["cost"])
            path.pop()

    dfs(src, [], 0)

    # cost 昇順で K 本
    heap.sort(key=lambda x: x[0])
    return heap[:K]

def k_physical_paths(DG, src, dst, K):
    heap = []

    def worst_cost():
        if len(heap) < K:
            return float("inf")
        return heap[-1][0]   # heapは後でソートする前提
      
    def dfs(node, path, cost):
        if cost > worst_cost():
            return

        if node == dst:
            #heapq.heappush(heap, (cost, path.copy()))
            heap.append((cost, path.copy()))
            heap.sort(key=lambda x: x[0])
            if len(heap) > K:
                heap.pop()   # 一番悪いものを捨てる
            return

        for _, v, k, data in DG.out_edges(node, keys=True, data=True):
            # DAG なのでノードループチェックは不要
            path.append((node, v, k))
            dfs(v, path, cost + data["cost"])
            path.pop()

    dfs(src, [], 0)

    # cost 昇順で K 本
    heap.sort(key=lambda x: x[0])
    return heap[:K]

#
# ----------------------------
# パス表示用（node列にも変換）
# ----------------------------
def path_to_nodes(path):
    nodes = [path[0][0]]
    for _, v, _ in path:
        nodes.append(v)
    return nodes

def build_dag_and_kpaths(G, src_set, dst_set, K, constraint, delta):
    """
    G: MultiDiGraph
    src/dst: 始点/終点
    K: 求めるパス数
    constraint: 'IGP', 'FA128', 'TE' など
    """
    Gc     = G
    Gc_rev = Gc.reverse(copy=False)

    # 制約条件に応じてコストやリンク制限を調整
    #if constraint["metric"] == 'IGP':
    #    Gc = G  # デフォルト IGP コストそのまま
    #elif constraint == 'FA128':
    #    Gc = apply_flexalgo_constraint(G, algo=128)
    #elif constraint == 'TE':
    #    Gc = apply_te_constraint(G)

    results={} 
    for src in src_set:
      results[src] = {}
      dist_src = nx.single_source_dijkstra_path_length( Gc, src, weight="cost" )
      best = dist_src[src]

      for dst in dst_set:
        if src == dst:
          continue

        paths = []
        dist_dst = nx.single_source_dijkstra_path_length( Gc_rev, dst, weight="cost" )
        paths = k_physical_paths_visited(
          Gc, src, dst, K, delta, dist_src, dist_dst, best
        )

        results[src][dst] = paths
        

    # DAG構築＋K-path計算
    #DG, dist = build_relaxed_dag(Gc, src, delta=300)
    #paths = k_physical_paths(DG, src, dst, K)
    return results

def task(args):
    G, src_set, dst_set, K, delta, c, const = args
    return c, build_dag_and_kpaths(G, src_set, dst_set, K, const, delta)



def recompute_kpaths(G, src_set, dst_set, K, constraints, changed_links, delta):
    # 変更をグラフに反映
    if changed_links == []:
      pass
    else:
      pass
      #update_graph_links(G, changed_links)

    results = {}

    #def task(args):
    #    c, const = args
    #    return build_dag_and_kpaths(G, src_set, dst_set, K, const, delta)

    # 並列実行
    with ProcessPoolExecutor() as executor:
          
        args_list = [(G, src_set, dst_set, K, delta, c, constraints[c]) for c in constraints.keys()]
        print(args_list)
        for c, paths in executor.map(task, args_list):
            results[c] = paths

    return results

# ----------------------------
# main
# ----------------------------
def main():

    debugp = ""

    # init caluclation / need to care before info ...
    G2, N = load_topology("topology/topology.json")
    G2_rev = G2.reverse(copy=False)

    changed_links = []
    constraints = {
      "0": { "metric": "IGP" }
    }

    K = 10
    delta = 300

    src_set = []
    dst_set = []

    # add to src/dst
    for n in N.keys():
      if N[n]["role"] == "PE":
        src_set.append(n)
        dst_set.append(n)

    start = datetime.datetime.now()
    results = recompute_kpaths(G2, src_set, dst_set, K, constraints, changed_links, delta)
    end = datetime.datetime.now()

    print(results)
    print(start)
    print(end)
    exit()


    #best = dist_src["7"]

    #paths = k_physical_paths_visited(
    #  G2,
    #  src="0",
    #  dst="5",
    #  K=10,
    #  delta=300,
    #  dist_src=dist_src,
    #  dist_dst=dist_dst,
    #  best=best
    #)
    #print(paths)
    #exit()

    # parameter K : max K:path / delta : diff with min path cost
    K = 10
    delta = 300

    start = datetime.datetime.now()

    for src in src_set:
      dist_src = nx.single_source_dijkstra_path_length( G2, src, weight="cost" )
      best = dist_src[src]

      for dst in dst_set:
        if src == dst:
          continue
        dist_dst = nx.single_source_dijkstra_path_length( G2_rev, dst, weight="cost" )
        paths = k_physical_paths_visited(
          G2, src, dst, K, delta, dist_src, dist_dst, best
        )

        #DG, dist = build_relaxed_dag(G2, src, dst, delta)

        #if src == dst:
        #  continue
        #if src not in DG or dst not in DG:
        #  continue
        #paths = k_physical_paths(DG, src, dst, K)

        if not paths:
          continue
         

        debugp += f"{src} -> {dst}\n"
        #for c, p in paths:
        #  debugp += f"cost={c}, path={path_to_nodes(p)}, links={p}\n"
        for p in paths:
          debugp += f"cost={p['cost']}, path={p['path']}, links={p['links']}\n"

    end = datetime.datetime.now()

    print(debugp)
    print(start)
    print(end)


# main
if __name__ == "__main__":
    main()



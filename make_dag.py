import networkx as nx
import datetime
import json
import pickle
import os
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile

# graph load
def make_topology(cid, c, topo, flg, chash, now):

  if os.path.exists(cid + ".pkl"): 
    with open(cid + ".pkl","rb") as f:
      X, meta = pickle.load(f)
      current_const_hash = meta["const_hash"]
  else:
    current_const_hash = ""

  const_hash = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
  meta = { 
    "const_hash": const_hash,
    "topo_hash" : chash["topo_hash"]
  }

  # topochange or constchange
  #if ( flg == True ) or ( current_const_hash != const_hash ):
  if ( 1 == 1 ) or ( current_const_hash != const_hash ):
    G = nx.MultiDiGraph()

    # constraints check and add
    for l in topo["links"]:
      if c["metric"] == "IGP":
        G.add_edge(l["u"], l["v"], key=l["key"], cost=l["cost"])
      if c["metric"] == "TE":
        G.add_edge(l["u"], l["v"], key=l["key"], cost=l["te"])
  
    with open("G/" + now + "/" + cid + ".pkl","wb") as f:
      pickle.dump((G, meta), f)

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

def combined_hash(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(file_hash(p).encode())
    return h.hexdigest()

# main
def main():

    old_hash = {
      "topo_hash": "",
      "const_hash": "",
    }

    while True:

      current_hash = {
        "topo_hash"  : safe_file_hash("topology/topology.json"),
        "const_hash" : safe_file_hash("etc/const.json")
      }
      topoflg = False

      if current_hash != old_hash:

        now = str(int(time.time() * 1000))
        os.mkdir("G/" + now)

        if current_hash["topo_hash"] != old_hash["topo_hash"]:
          topoflg = True

        #start = datetime.datetime.now()
        # load const
        with open('etc/const.json') as f:
          const = json.load(f)
        # load topology
        with open('topology/topology.json') as f:
          topo  = json.load(f)

        with ProcessPoolExecutor() as executor:
          futures = []
          for cid, c in const.items():
            futures.append(executor.submit(make_topology, cid, c, topo, topoflg, current_hash, now))
          for f in as_completed(futures):
            f.result()

        dir = os.path.dirname("G/current")
        with tempfile.NamedTemporaryFile("w", dir=dir, delete=False) as f:
          f.write(now)
          tempname = f.name
        os.replace(tempname, "G/current")  # atomic

        old_hash = current_hash

      time.sleep(0.005)

    #end = datetime.datetime.now()


# start
if __name__ == "__main__":
    main()



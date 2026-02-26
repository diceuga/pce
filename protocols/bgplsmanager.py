# bgpls/manager.py

import hashlib
import json
#from .bgplsdecode import decode_bgp_update
#from .bgplspeer2 import BgplsPeer
from protocols.bgplsdecode import decode_bgp_update
from protocols.bgplspeer import BgplsPeer
import logging
log = logging.getLogger("bgpls")

class BgplsManager:
    def __init__(self):
      self.peers = {}
      self.peer_state = {}
      self.peer_eor = set()

      self.peer_lsdb = {}   # peer → { ls_id: obj }
      self.best_lsdb = {}   # ls_id → {hash, peer, data}
      self.routes    = {}

      self.main_cb = None
      self.global_state = "INIT"

    def stoppeer(self, peer):
      if peer in  self.peers:
         self.peers[peer].running = False

    def register_main_callback(self, cb):
      self.main_cb = cb

    def register_peer(self, peer_addr, sock, router_id, peer):
      peer_obj = BgplsPeer(peer_addr, sock, self.on_peer_event, router_id)
      self.peers[peer_addr] = {
        "cfg": peer,
        "obj": peer_obj
      }
      self.peer_state[peer_addr] = "INIT"
      peer_obj.start()

    # ---------- peer → manager ----------
    def on_peer_event(self, ev):
        t = ev["type"]
        peer = ev["peer"]

        if t == "BGP_OPEN":
          self.peer_state[peer] = "SYNCING"
          #print("[BGPM]" + str(peer) + ": SYNCING")
          self._update_global_state()

        elif t == "BGP_UPDATE":
          #self._handle_update(peer, ev["raw"])
          #print(f"[DEBUG] UPDATE raw len={len(ev["raw"])}")
          #print(f"[DEBUG] UPDATE raw head={ev["raw"][:20].hex()}")
          self._handle_update(peer, ev["raw"])

        elif t == "BGP_EOR":
          self.peer_state[peer] = "SYNCED"
          #print("[BGPM]" + str(peer) + ": SYNCED")
          self._handle_eor(peer)

        elif t == "PEER_DOWN":
          self._handle_peer_down(peer)

    # ---------- UPDATE ----------
    def _handle_update(self, peer, raw):
        nlris = decode_bgp_update(raw)
        #print("#################decode done")
        #print(nlris)
        #print("#################decode done")
        #ls_list = self.decode_bgpls(raw)
        #for nlri in nlris:
        #    print(nlri)
        #    print(type(nlri))
        #    self._process_ls(peer, nlri)
        # withdraw
        for nlri in nlris.get("withdraw", []):
            self._process_nlri(peer, nlri, withdraw=True)

        # announce
        attrs = nlris.get("path_attributes", [])
        ls_attrs = nlris.get("ls_attributes", [])
        for nlri in nlris.get("announce", []):
        #    attrs = update.get("path_attributes", [])
            self._process_nlri(peer, nlri, attrs=attrs, ls_attrs=ls_attrs)

    def _process_nlri(self, peer, nlri, attrs=None, withdraw=False, ls_attrs=None):

      t = nlri["nlri_type"]
      #d = nlri.get("detail", {})

      if t in ( 1, 2 ):
        #print("############ process_nlri")
        #print(nlri)
        key = self.build_nlri_key(nlri)
        #print("############")
        #print("key:" + str(key))
        #print(nlri)
        #print("############")

        if withdraw:
          route = self.routes.get(key)
          if not route:
            return

          route["paths"].pop(peer, None)
          if not route["paths"]:
              self.routes.pop(key)
              self.best_lsdb.pop(key, None)
              if None not in key:
                self._emit_main({"type": "LS_WITHDRAW", "key": key, "nlri": nlri, "ls_attrs":{}})
          else:
              self._recalc_best(key)
          return

        # announce
        route = self.routes.setdefault(key, {
          "paths": {},
          "best": None
        })

        h = self._hash((nlri, attrs, ls_attrs))
        route["paths"][peer] = {
          "nlri": nlri,
          "attrs": attrs,
          "ls_attrs": ls_attrs,
          "hash": h
        }
  
        self._recalc_best(key)


    def build_nlri_key(self, nlri):
      t = nlri["nlri_type"]
      d = nlri.get("detail", {})

      # Node NLRI
      if t == 1:
          n = d["local_node"]
          return (
              #"node",
              #nlri["protocol_id"],
              #n.get("asn"),
              #n.get("area_id"),
              n.get("ipv4_router_id"),
          )

      # Link NLRI
      elif t == 2:
          #ld = d["detail"]
          # only ipv4
          return (
              #"link",
              #nlri["protocol_id"],
              d.get("local_node").get("ipv4_router_id"),
              d.get("remote_node").get("ipv4_router_id"),
              d.get("ipv4_interface_address"),
              d.get("ipv4_neighbor_address"),
          )

      # Prefix NLRI not needed
      elif t == 3:
          pass
          #pd = d["prefix_descriptor"]
          #return (
          #    "prefix",
          #    nlri["protocol_id"],
              #"100.127.1.0",
              #"24"
          #    d.get("ip_reachability_info").get("prefix"),
          #    d.get("ip_reachability_info").get("prefix_len"),
          #)

      else:
        raise ValueError(f"unknown nlri_type {t}")

    def _select_best(self, paths):
      return min(paths.keys())

    def _recalc_best(self, key):
      #print("key")
      #print(key)
      route = self.routes[key]
      old_best = route["best"]

      new_best = self._select_best(route["paths"])
      route["best"] = new_best

      best_entry = route["paths"][new_best]

      old = self.best_lsdb.get(key)
      if old and old["hash"] == best_entry["hash"]:
          return

      self.best_lsdb[key] = {
          "peer": new_best,
          "hash": best_entry["hash"],
          "nlri": best_entry["nlri"],
          "attrs": best_entry["attrs"],
          "ls_attrs": best_entry["ls_attrs"]
      } 

      #print("recalc best")
      #print(key)

      if None not in key:
        self._emit_main({
          "type": "LS_UPDATE" if old else "LS_ADD",
          "key": key,
          "nlri": best_entry["nlri"],
          #"attrs": best_entry["attrs"],
          "ls_attrs": best_entry["ls_attrs"]
        })
      #print("#########best###########")
      #print(best_entry)

    def _handle_eor(self, peer):
        if peer in self.peer_eor:
            return
        self.peer_eor.add(peer)
        self.peer_state[peer] = "EOR"
        self._update_global_state()

    def _handle_peer_down(self, peer):
        # implicit withdraw は後で

        print(self.routes)
       
        for r in list(self.routes.keys()):
          print ("peer down")
          print (r)
          print (peer)
          route = self.routes.get(r)
          route["paths"].pop(peer, None)
          if not route["paths"]:
            self.routes.pop(r)
            nlri = self.best_lsdb[r]["nlri"]
            self.best_lsdb.pop(r, None)
            if None not in r:
              self._emit_main({"type": "LS_WITHDRAW", "key": r, "nlri": nlri, "ls_attrs":{}})
          else:
            self._recalc_best(r)
            
        self.peers.pop(peer, None)
        self.peer_state.pop(peer, None)
        self.peer_eor.discard(peer)
        self._update_global_state()

    def _update_global_state(self):
        total = len(self.peer_state)
        eor = len(self.peer_eor)

        if eor == 0:
            state = "SYNCING"
        #elif eor < total:
        #    state = "PARTIAL_ACTIVE"
        else:
            state = "ACTIVE"

        if state != self.global_state:
            self.global_state = state
            self._emit_main({
                "type": "BGPLS_STATE",
                "state": state
            })

    def _emit_main(self, ev):
        if self.main_cb:
            self.main_cb(ev)

    def _hash(self, obj):
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True).encode()
        ).hexdigest()


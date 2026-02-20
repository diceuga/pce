# protocols/pcepmanager.py
from protocols.pceppcc import PcepPcc
#from protocols.pcepdecode import decode_pcrpt
import queue

class PcepManager:
    def __init__(self, tx_queue: queue.Queue):
        self.peers = {}
        self.lsps = {}
        self.main_cb = None
        self.tx_queue = tx_queue

    def register_main_callback(self, cb):
        self.main_cb = cb

    def register_peer(self, peer_addr, sock, peer_cfg):
        peer = PcepPcc(peer_addr, sock, self.on_peer_event)
        self.peers[peer_addr] = peer
        peer.start()

    # ---------- peer → manager ----------
    def on_peer_event(self, ev):
        t = ev["type"]
        pcc = ev["pcc"]

        if t == "PCEP_REPORT":
            self._handle_report(peer, ev["raw"])

        elif t == "PEER_DOWN":
            self._handle_peer_down(peer)
        elif t == "PCC_SYNC":
          # just forward to main
          print(ev)
          self._emit_main(ev)
        else:
          print("event!!")
          print(t)

    def _handle_report(self, peer, raw):
        lsp = decode_pcrpt(raw)
        key = (peer, lsp["plsp_id"])

        old = self.lsps.get(key)
        self.lsps[key] = lsp

        self._emit_main({
            "type": "LSP_UPDATE" if old else "LSP_ADD",
            "key": key,
            "lsp": lsp
        })

    def _handle_peer_down(self, peer):
        for k in list(self.lsps.keys()):
            if k[0] == peer:
                lsp = self.lsps.pop(k)
                self._emit_main({
                    "type": "LSP_WITHDRAW",
                    "key": k,
                    "lsp": lsp
                })

    # ---------- main → manager ----------
    def send_pce_command(self, cmd):
        """
        main から Queue 経由で飛んできた指示
        """
        peer = cmd["peer"]
        payload = cmd["payload"]
        if peer in self.peers:
            self.peers[peer].send(payload)

    def _emit_main(self, ev):
        if self.main_cb:
            self.main_cb(ev)


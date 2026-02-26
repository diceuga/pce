# protocols/pcepserver.py
import socket
import threading
from protocols.pcepmanager import PcepManager
from utils.diff            import DiffType

class PcepServer(threading.Thread):
    def __init__(self, config, tx_queue, log):
        super().__init__(daemon=True)
        self.log = log
        self.bind_addr = (config["listen"], config["port"])
        self.peer_config = config.get("pccs", {})
        #self.peer_config = {p["address"]: p for p in config.get("pccs", [])}
        self.manager = PcepManager(tx_queue, self.log)

    def update_pccs(self, ev):
      if ( ev["diff"]["type"] == DiffType.DEL ): # Just Del
        self.peer_config.pop(ev["diff"]["id"])
        self.manager.stoppcc(ev["diff"]["id"])
        print(self.peer_config)
      elif ( ev["diff"]["type"] == DiffType.ADD ):
        self.peer_config[ev["diff"]["id"]] = ev["diff"]["new"] 
        print(self.peer_config)
      

    def register_main_callback(self, cb):
        self.manager.register_main_callback(cb)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(self.bind_addr)
        sock.listen()

        while True:
            conn, addr = sock.accept()
            peer_ip = addr[0]

            if peer_ip not in self.peer_config:
                conn.close()
                continue
            self.manager.register_peer(peer_ip, conn, self.peer_config[peer_ip])


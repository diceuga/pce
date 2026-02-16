import socket
from protocols.bgplsmanager import BgplsManager
import threading

class BgpServer(threading.Thread):
  def __init__(self, config):
    super().__init__(daemon=True)
    bgp = config
    self.bind_addr = (bgp["listen"], bgp["port"])
    self.router_id = bgp["router_id"]
    self.peer_config = {
      p["address"]: p
      for p in bgp.get("peers", [])
    }

    self.manager = BgplsManager()

  def register_main_callback(self, cb):
    self.manager.register_main_callback(cb)

  #def start(self):
  def run(self):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(self.bind_addr)
    sock.listen()

    #print(f"[BGP] listen on {self.bind_addr}")

    while True:
      conn, addr = sock.accept()
      peer_ip = addr[0]

      if peer_ip not in self.peer_config:
        #print(f"[BGP] reject unknown peer {peer_ip}")
        conn.close()
        continue

      peer = self.peer_config[peer_ip]
      #print(f"[BGP] accepted {peer['name']} ({peer_ip})")

      self.manager.register_peer(peer_ip, conn, self.router_id, peer)


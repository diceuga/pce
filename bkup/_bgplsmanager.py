import socket
import threading
#from .peer import PeerSession
from .bgplspeer import PeerSession


class BgplsManager:
    def __init__(self, config, event_queue):
        #self.port = config["listen_port"]
        self.port = 179
        self.router_id = config["router_id"]
        self.peers_cfg = config.get("peers", [])
        self.event_queue = event_queue
        self.sessions = {}

    def start(self):
        t = threading.Thread(target=self._listen, daemon=True)
        t.start()

    def _listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.listen(10)

        print(f"[BGPLS] listening on :{self.port}")

        while True:
            conn, addr = sock.accept()
            peer_ip, peer_port = addr
            print(f"[BGPLS] connection from {peer_ip}")

            session = PeerSession(
                peer_ip=peer_ip,
                conn=conn,
                event_queue=self.event_queue,
                router_id=self.router_id
            )
            self.sessions[peer_ip] = session
            session.start()


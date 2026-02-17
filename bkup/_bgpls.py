# proto/bgpls.py
import threading
import time
from event.event import Event
from event.queue import event_queue


class BgplsClient:
    def __init__(self, peer_name: str):
        self.peer_name = peer_name
        self.running = False

    def start(self):
        self.running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        """
        本来はここで BGP セッション確立・受信ループ
        """
        print(f"[bgpls] peer {self.peer_name} started")

        # --- ダミー: 定期的に LS Update を生成 ---
        while self.running:
            time.sleep(5)

            # 本物なら NLRI / TLV を decode した結果
            ls_update = {
                "peer": self.peer_name,
                "type": "link",
                "id": ("R1", "R2"),
                "metric": 10,
                "bw": 100
            }

            ev = Event(
                type="LS_UPDATE",
                payload=ls_update
            )
            event_queue.put(ev)

    def stop(self):
        self.running = False


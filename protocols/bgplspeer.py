# bgpls/peer.py

import threading
import socket
import struct
import time
import traceback

BGP_HEADER_LEN = 19
AFI_BGPLS = 16388
SAFI_BGPLS = 71
ATTR_MP_UNREACH = 15
ATTR_MP_REACH = 14

class BgplsPeer(threading.Thread):

    def __init__(self, peer_addr, sock, event_cb, router_id):
      super().__init__(daemon=True)
      self.peer_addr = peer_addr
      self.peer_ip = peer_addr
      self.conn = sock
      self.event_cb = event_cb
      self.router_id = socket.inet_aton(router_id)
      self.running = True
      self.hold_time = 90

    def _hold_timer(self):
      while self.running:
        time.sleep(1)
        if time.time() - self.last_rx > self.hold_time:
          self.running = False
          break

    def _handle_message(self, msg_type, payload):
        self.last_rx = time.time()

        if msg_type == 1:
          #print(f"[BGP] OPEN from {self.peer_ip}")
          self._send_open(payload)
          hold = struct.unpack("!H", payload[3:5])[0]
          self.hold_time = min(self.hold_time, hold)
          self.event_cb({ "type": "BGP_OPEN", "peer": self.peer_ip })

        elif msg_type == 2:
          #print(f"[BGP] UPDATE from {self.peer_ip} ({len(payload)} bytes)")

          if self._is_bgp_ls_eor(payload):
            #print(f"[BGPLS] EOR received from {self.peer_ip}")
            self.event_cb({
              "type": "BGP_EOR",
              "peer": self.peer_ip
            })
          else:
            self.event_cb({
              "type": "BGP_UPDATE",
              "peer": self.peer_ip,
              "raw": payload
            })

        elif msg_type == 3:
          pass
          #print(f"[BGP] NOTIFICATION from {self.peer_ip}")

        # KEEPALIVE
        elif msg_type == 4:
          #print(f"[BGP] KEEPALIVE from {self.peer_ip}")
          self._send_keepalive()

        # REFRESH ignore
        elif msg_type == 5:
          pass

        else:
          pass
          #print(f"[BGP] UNKNOWN {msg_type} from {self.peer_ip}")

    def _recv_all(self, size):
        buf = b""
        while len(buf) < size:
            chunk = self.conn.recv(size - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def run(self):
      watchdog = threading.Thread(
        target=self._hold_timer,
        daemon=True
      )
      watchdog.start()

      try:
        while self.running:
          header = self._recv_all(BGP_HEADER_LEN)
          if not header:
            break

          marker, length, msg_type = struct.unpack("!16sHB", header)
          payload_len = length - BGP_HEADER_LEN
          payload = self._recv_all(payload_len)
          self._handle_message(msg_type, payload)

      except Exception as e:
        print(f"[BGPLS] peer {self.peer_ip} error: {e}")
        traceback.print_exc()

      finally:
        self.conn.close()
        #print(f"[BGPLS] peer {self.peer_ip} disconnected")
        self.event_cb({
          "type": "PEER_DOWN",
          "peer": self.peer_ip
        })

    def _build_bgp_message(self, msg_type, payload):
        marker = b"\xff" * 16
        length = 19 + len(payload)
        header = struct.pack("!16sHB", marker, length, msg_type)
        return header + payload

    # keepalive
    def _send_keepalive(self):
        msg = self._build_bgp_message(4, b"")
        self.conn.sendall(msg)
        #print(f"[BGP] KEEPALIVE sent to {self.peer_ip}")

    # change only routre-id
    def _send_open(self, recv_payload):
      payload = bytearray(recv_payload)
      payload[5:9] = self.router_id
      msg = self._build_bgp_message(1, payload)
      self.conn.sendall(msg)
      #print(f"[BGP] OPEN sent to {self.peer_ip}")

    def _is_bgp_ls_eor(self, payload: bytes) -> bool:
      try:
        pos = 0

        # Withdrawn Routes
        withdrawn_len = int.from_bytes(payload[pos:pos+2], "big")
        pos += 2 + withdrawn_len

        # Path Attributes
        attr_len = int.from_bytes(payload[pos:pos+2], "big")
        pos += 2
        end = pos + attr_len

        while pos < end:
            flags = payload[pos]
            code = payload[pos + 1]
            pos += 2

            if flags & 0x10:  # Extended Length
                length = int.from_bytes(payload[pos:pos+2], "big")
                pos += 2
            else:
                length = payload[pos]
                pos += 1

            value = payload[pos:pos + length]

            # ★ MP_UNREACH_NLRI
            if code == ATTR_MP_UNREACH:
                afi = int.from_bytes(value[0:2], "big")
                safi = value[2]

                withdrawn_nlri = value[3:]

                if (
                    afi == AFI_BGPLS and
                    safi == SAFI_BGPLS and
                    len(withdrawn_nlri) == 0
                ):
                    return True

            pos += length

      except Exception as e:
        print(f"[BGPLS] EOR parse error from {self.peer_ip}: {e}")

      return False



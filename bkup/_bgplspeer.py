import threading
import struct
import socket


BGP_HEADER_LEN = 19
AFI_BGPLS = 16388
SAFI_BGPLS = 71
ATTR_MP_UNREACH = 15
ATTR_MP_REACH = 14


class PeerSession(threading.Thread):
    def __init__(self, peer_ip, conn, event_queue, router_id):
        super().__init__(daemon=True)
        self.peer_ip = peer_ip
        self.conn = conn
        self.event_queue = event_queue
        self.router_id = socket.inet_aton(router_id)
        self.event_cb = event_queue # to manager

    def run(self):
        try:
            while True:
                header = self._recv_all(BGP_HEADER_LEN)
                if not header:
                    break

                marker, length, msg_type = struct.unpack("!16sHB", header)
                payload_len = length - BGP_HEADER_LEN
                payload = self._recv_all(payload_len)

                self._handle_message(msg_type, payload)

        except Exception as e:
            print(f"[BGPLS] peer {self.peer_ip} error: {e}")
        finally:
            self.conn.close()
            print(f"[BGPLS] peer {self.peer_ip} disconnected")

    def _recv_all(self, size):
        buf = b""
        while len(buf) < size:
            chunk = self.conn.recv(size - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _send_open(self, recv_payload):
        # recv_payload をコピーして Router-ID だけ変更
        payload = bytearray(recv_payload)
        payload[5:9] = self.router_id

        msg = self._build_bgp_message(1, payload)
        self.conn.sendall(msg)

        print(f"[BGP] OPEN sent to {self.peer_ip}")

    def _build_bgp_message(self, msg_type, payload):
        marker = b"\xff" * 16
        length = 19 + len(payload)
        header = struct.pack("!16sHB", marker, length, msg_type)
        return header + payload

    def _send_keepalive(self):
        msg = self._build_bgp_message(4, b"")
        self.conn.sendall(msg)

        print(f"[BGP] KEEPALIVE sent to {self.peer_ip}")

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


    def _handle_message(self, msg_type, payload):
        if msg_type == 1:
            print(f"[BGP] OPEN from {self.peer_ip}")
            self._send_open(payload)
            self._send_keepalive()

        elif msg_type == 2:
            print(f"[BGP] UPDATE from {self.peer_ip} ({len(payload)} bytes)")
            if self._is_bgp_ls_eor(payload):
              print(f"[BGPLS] EOR received from {self.peer_ip}")
              self.event_queue.put({
                "type": "BGPLS_EOR",
                "peer": self.peer_ip
              })
            else:
              self.event_queue.put({
                "type": "BGPLS_UPDATE",
                "peer": self.peer_ip,
                "raw": payload
              })

            #self.event_queue.put({
            #    "type": "BGPLS_UPDATE",
            #    "peer": self.peer_ip,
            #    "raw": payload
            #})
        elif msg_type == 3:
            print(f"[BGP] NOTIFICATION from {self.peer_ip}")
        # KEEPALIVE
        elif msg_type == 4:
          #print(f"[BGP] KEEPALIVE from {self.peer_ip}")
          self._send_keepalive()
        # REFRESH ignore
        elif msg_type == 5:
          pass
        else:
            print(f"[BGP] UNKNOWN {msg_type} from {self.peer_ip}")


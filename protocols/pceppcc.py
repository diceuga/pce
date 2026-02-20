# protocols/pceppeer.py
import threading
import struct
import time
import traceback

from protocols.pcepdecode import decode_pcep_open
from protocols.pcepdecode import decode_pcep_report

PCEP_HDR_LEN = 4

PCEP_OPEN       = 1
PCEP_KEEPALIVE  = 2
PCEP_PCREQ      = 3
PCEP_PCREP      = 4
PCEP_PCNOTIFY   = 5
PCEP_PCERROR    = 6
PCEP_PCREPORT   = 10
PCEP_PCINITIATE = 11


class PcepPcc(threading.Thread):
    def __init__(self, peer_addr, sock, event_cb):
        super().__init__(daemon=True)
        self.peer_addr = peer_addr
        self.conn = sock
        self.event_cb = event_cb
        self.running = True
        self.last_rx = time.time()

    def _recv_all(self, size):
        buf = b""
        while len(buf) < size:
            chunk = self.conn.recv(size - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _handle_message(self, msg_type, payload):
      self.last_rx = time.time() 
      if msg_type == PCEP_OPEN:
        openinfo,respopen = decode_pcep_open(payload)
        print("PCEP OPEN")
        print(openinfo)
        self.send_open(respopen)

      elif msg_type == PCEP_KEEPALIVE:
        print("KA")
        self.send_ka()
      elif msg_type == PCEP_PCREPORT:
        print("REPORT")
        #print(payload)
        print(time.time())
        print(decode_pcep_report(payload))
      elif msg_type == PCEP_PCINITIATE:
        pass
      elif msg_type == PCEP_PCERROR:
        print(payload)
      else:
        print(msg_type)

    def build_pcep_open(self,keepalive=30, deadtimer=120, sid=1):
      # OPEN Object Header
      # Class=1, Type=1
      obj_class = 1
      obj_type = 1
      print(type(keepalive))
      print(type(deadtimer))
      print(type(sid))


      ver_flags = (1 << 5)  # PCEP v2
      body = struct.pack("!BBBB", ver_flags, keepalive, deadtimer, sid)

      length = 4 + len(body)
      header = struct.pack("!BBH", obj_class, obj_type, length)

      return header + body

    def send_ka(self):
      ver_flags = (1 << 5)
      msg_type = 2
      length = 4 
      header = struct.pack("!BBH", ver_flags, msg_type, length)
      self.conn.sendall(header)

    def send_open(self, payload):
      #payload = self.build_pcep_open(
      #    keepalive=open_info["keepalive"],
      #    deadtimer=open_info["deadtimer"],
      #    sid=1,
      #)
      #msg = self.build_pcep_open(PCEP_OPEN, payload)

      obj_class = 1
      obj_type = (1 << 4)
      length = 4 + len(payload)
      openobj = struct.pack("!BBH", obj_class, obj_type, length) + payload

      ver_flags = (1 << 5)
      msg_type = 1
      length = 4 + len(openobj)
      header = struct.pack("!BBH", ver_flags, msg_type, length)


      self.conn.sendall(header + openobj)
    
    def run(self):
        try:
          while self.running:
            hdr = self._recv_all(PCEP_HDR_LEN)
            if not hdr:
              break

            ver_flags, msg_type, length = struct.unpack("!BBH", hdr)
            payload = self._recv_all(length - 4)
            self._handle_message(msg_type, payload)
            
        except Exception as e:
          print(f"[PCEP] peer {self.peer_addr} error: {e}")
          traceback.print_exc()
        finally:
          self.conn.close()
          self.event_cb({"type": "PCC_DOWN", "peer": self.peer_addr})

    def send(self, payload: bytes):
        self.conn.sendall(payload)


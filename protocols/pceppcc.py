# protocols/pceppeer.py
import threading
import struct
import time
import traceback
import queue
import itertools

from protocols.pcepdecode import decode_pcep_open
from protocols.pcepdecode import decode_pcep_report
from protocols.pcepencode import build_pcinitiate_from_path

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
    def __init__(self, peer_addr, sock, event_cb,log):
        super().__init__(daemon=True)
        self.peer_addr = peer_addr
        self.conn = sock
        self.event_cb = event_cb
        self.running = True
        self.last_rx = time.time()
        self.openinfo = {}
        self.sync    = False
        self.hold_time = 120
        self.send_queue = queue.Queue()
        self.log = log
        self.srpid = itertools.count(1)

    def _hold_timer(self):
      while self.running:
        time.sleep(1)
        if time.time() - self.last_rx > self.hold_time:
        #if time.time() - self.last_rx > 5:
          self.running = False
          break

    def _recv_all(self, size):
        buf = b""
        while len(buf) < size:
            chunk = self.conn.recv(size - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def checkeos(repinfo):
      ret = False
      if ( repinfo["lsp"]["plsp_id"] == 0):
        ret = True
      pass

    def _handle_message(self, msg_type, payload):
      self.last_rx = time.time() 

      # open
      if msg_type == 1:
        print("PCEP OPEN")
        self.openinfo,respopen = decode_pcep_open(payload)
        self.hold_time = self.openinfo.get("deadtimer",120)
        print(self.openinfo)
        self.send_open(respopen)

      # keep alive
      elif msg_type == 2:
        self.send_ka()

      elif msg_type == PCEP_PCREPORT:
        print("REPORT")
        #print(payload)
        #print(time.time())
        repinfos = decode_pcep_report(payload)
        print(repinfos)
        for repinfo in repinfos:
          if (repinfo["lsp"]["sync"] == False) and ( repinfo["lsp"]["plsp_id"] == 0): 
            if (self.sync == False):
              self.sync = True
            ev = {
              "type": "PCC_SYNC",
              "pcc" : self.peer_addr,
              "info": self.openinfo
            }
            self.event_cb(ev)
          else:
            ev = {
              "type": "PCEP_REPORT",
              "pcc" : self.peer_addr,
              "info": repinfo
            }
            self.event_cb(ev)
         #pass 

      elif msg_type == PCEP_PCINITIATE:
        pass
      elif msg_type == PCEP_PCERROR:
        print("PCEP_PCERROR")
        print(payload)
      else:
        print("OTHER MSGTYPE")
        print(msg_type)

    def send_ka(self):
      # KA
      ver_flags = (1 << 5)
      msg_type = 2
      length = 4 
      header = struct.pack("!BBH", ver_flags, msg_type, length)
      self.conn.sendall(header)

    def send_open(self, payload):
      # payload is open object
      ver_flags = (1 << 5)
      msg_type = 1
      length = 4 + len(payload)
      header = struct.pack("!BBH", ver_flags, msg_type, length)

      self.conn.sendall(header + payload)

    def _handle_command(self, cmd):
      if cmd["type"] == 1: #"JUST SEND":
        self.conn.sendall(cmd["data"])
      elif cmd["type"] == "PATH UPDATE": #"JUST SEND":
        print("pcc handleevent")
        #print(self.peer_addr)
        print(cmd)
        srpid = next(self.srpid)
        A = build_pcinitiate_from_path(cmd,srpid)
        if A != None:
          ver_flags = (2 << 5)
          length = 4 + len(A)
          pcepmsg = struct.pack("!BBH", ver_flags, 12, length) + A
          print(pcepmsg)
          self.send_queue.put({
            "type": 1,
            "data": pcepmsg
          })



    def start_sender(self):
      t = threading.Thread(target=self._send_loop, daemon=True)
      t.start()
    
    def _send_loop(self):
      while self.running:
        try:
            cmd = self.send_queue.get()
            self._handle_command(cmd)

        except Exception as e:
            self.running = False 
            print(f"[PCEP] peer {self.peer_addr} send error: {e}")
    
    def run(self):
        watchdog = threading.Thread(
          target=self._hold_timer,
          daemon=True
        )
        watchdog.start()

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
          self.event_cb({"type": "PCC_DOWN", "pcc": self.peer_addr})

    def send(self, ev):
      self.send_queue.put(ev)
        #self.conn.sendall(payload)
        # queue


# protocols/pceppeer.py
import threading
import struct
import time
import traceback
import queue
import itertools
import logging

from protocols.pcepdecode import decode_pcep_open
from protocols.pcepdecode import decode_pcep_report
from protocols.pcepdecode import decode_pcep_error
from protocols.pcepencode import build_pcinitiate_from_path
from protocols.pcepencode import build_pcdelete_from_path

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
        self.last_tx = time.time()
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
        if time.time() - self.last_tx > 10:
          self.send_ka()

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
        self.log.info("[PCEP] OPEN " + str(self.peer_addr))
        self.openinfo, respopen = decode_pcep_open(payload)
        self.hold_time = self.openinfo.get("deadtimer",120)
        self.log.info("[PCEP] OPEN INFO " + str(self.peer_addr) + " " + str(self.openinfo))
        #print(self.openinfo)
        self.send_open(respopen)

      # keep alive
      elif msg_type == 2:
        self.send_ka()

      elif msg_type == PCEP_PCREPORT:
        self.log.info("[PCEP] REPORT " + str(self.peer_addr))
        repinfos = decode_pcep_report(payload)
        self.log.info("[PCEP] REPORT INFO " + str(self.peer_addr) + " " + str(repinfos))
        #print(repinfos)
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
        #self.log.info("[PCEP] ERROR " + str(self.peer_addr))
        self.log.info("[PCEP] ERROR " + str(self.peer_addr) + " " + str(payload))
        errinfos = decode_pcep_error(payload)
        if errinfos != None:
          if "srp" in errinfos.keys():
            print("send PCEP_PCERROR")
            ev = {
              "type": "PCEP_ERROR",
              "pcc" : self.peer_addr,
              "info": errinfos
            }
            self.event_cb(ev)
            print("send PCEP_PCERROR")
        #print(payload)
      else:
        self.log.info("[PCEP] OTHER MSG " + str(self.peer_addr) + " " + str(msg_type))
        #print(msg_type)

    def send_ka(self):
      # endA
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

      elif cmd["type"] == "PATH DELETE":
        self.log.info("[PATH] PATH DELETE")
        self.log.info("[PATH] " + str(cmd))
        A = build_pcdelete_from_path(cmd)
        if A != None:
          ver_flags = (1 << 5)
          length = 4 + len(A)
          pcepmsg = struct.pack("!BBH", ver_flags, 12, length) + A
          self.log.info("[PATH] INITIATE(DEL)")
          #print(pcepmsg)
          self.send_queue.put({
            "type": 1,
            "data": pcepmsg
          })

      elif cmd["type"] == "PATH UPDATE": 
        self.log.info("[PATH] PATH UPDATE")
        self.log.info("[PATH] " + str(cmd))
        #print("pcc handleevent")
        #print(self.peer_addr)
        #print(cmd)
        #srpid = next(self.srpid)
        #A = build_pcinitiate_from_path(cmd,srpid)
        A = build_pcinitiate_from_path(cmd)

        plsp_id = cmd["detail"]["plsp_id"]

        if A != None:
          ver_flags = (1 << 5)
          length = 4 + len(A)
          if plsp_id == 0:
            pcepmsg = struct.pack("!BBH", ver_flags, 12, length) + A
            self.log.info("[PATH] INITIATE")
          else:
            pcepmsg = struct.pack("!BBH", ver_flags, 11, length) + A
            self.log.info("[PATH] UPDATE")
            #print(pcepmsg)
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
            traceback.print_exc()
    
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


# import 
#import networkx as nx
#import heapq
#import json
import os
import time
#import copy
#from collections import defaultdict
#from datetime import datetime
#import threading
from protocols.bgpserver  import BgpServer
from protocols.pcepserver import PcepServer
from utils.logging import setup_logging
from utils.config  import ConfigManager
import logging
from manager.bwmanager    import BWManager
from manager.graphmanager import GraphManager
from manager.pathmanager  import PathManager


def main():

  # LOG
  setup_logging(os.path.dirname(os.path.abspath(__file__)))
  G_LOG = logging.getLogger()
  G_LOG.info("Main start")

  # Manager 
  G_CM = ConfigManager(G_LOG) # ConfigManager
  G_BM = BWManager(G_LOG)     # BWManager
  G_GM = GraphManager(G_LOG)  # GraphManager
  G_PM = PathManager(G_LOG)   # PathManager

  # Attach 
  G_CM.attach_PM(G_PM)
  G_CM.attach_GM(G_GM)
  
  G_GM.attach_BM(G_BM)
  G_GM.attach_CM(G_CM)
  G_GM.attach_PM(G_PM)

  G_PM.attach_BM(G_BM)
  G_PM.attach_CM(G_CM)
  G_PM.attach_GM(G_GM)

  # BGP START
  bgp = BgpServer(G_CM.BGPLSINFO)
  bgp.register_main_callback(G_GM.on_bgpls_event)
  #G_CM.attach_BGP(bgp.bgpmanager)
  G_CM.attach_bgpserver(bgp)
  bgp.start()

  # PCEP START
  pcep = PcepServer(G_CM.PCEPINFO, G_PM.P_Queue, G_LOG)
  pcep.register_main_callback(G_PM.on_pcep_event)
  G_CM.attach_pcepserver(pcep)
  pcep.start()

  # PCEP Server Q attach to PM
  G_PM.attach_PCEP_Q(pcep.manager.rx_queue)

  ############## Loop start 
  while True:
    time.sleep(5)
    #pri, ev = G_PM.C_Queue.get()
    #G_PM.handle_event(ev)

# start
if __name__ == "__main__":
  main()



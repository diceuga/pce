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


#-------------------------------------------------------
# LOG
setup_logging(os.path.dirname(os.path.abspath(__file__)))
G_LOG = logging.getLogger()

# Manager
G_BM = BWManager(G_LOG)     # BWManager
G_GM = GraphManager(G_LOG)  # GraphManager
G_CM = ConfigManager(G_LOG) # ConfigManager
G_PM = PathManager(G_LOG)   # PathManager

# Attach
G_CM.attach_PM(G_PM)
G_CM.attach_GM(G_GM)

G_GM.attach_BM(G_BM)
G_GM.attach_CM(G_CM)
G_GM.attach_PM(G_PM)

#G_PM.attach_X(compute_pathsX)
G_PM.attach_BM(G_BM)
G_PM.attach_CM(G_CM)
G_PM.attach_GM(G_GM)

def main():

  G_LOG.info("Main start")

  # BGP START
  bgp = BgpServer(G_CM.BGPLSINFO)
  bgp.register_main_callback(G_GM.on_bgpls_event)
  bgp.start()

  # PCEP START
  pcep = PcepServer(G_CM.PCEPINFO, G_PM.P_Queue)
  pcep.register_main_callback(G_PM.on_pcep_event)
  pcep.start()

  ############## Loop start 
  while True:
    time.sleep(5)
    #pri, ev = G_PM.C_Queue.get()
    #G_PM.handle_event(ev)

# start
if __name__ == "__main__":
  main()



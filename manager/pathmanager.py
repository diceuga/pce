import queue
import time
from collections import defaultdict
from itertools import count
from utils.diff    import DiffType
import heapq
import networkx as nx
import threading


class PathManager:
  def __init__(self, log):
    self.log = log
    self.C_Queue      = queue.PriorityQueue()
    self.C_Cnt        = count()
    self.P_Queue      = queue.Queue()
    self.pcep_queue   = None

    #self.computeX     = None
    self.PATH         = {}
    self.PATH_d       = {}
    self.PATH_n       = {}

    # pcep status
    self.PCC          = {}
    self.PCC_srpid   = count(1)

    # start Q mon
    self.wq = threading.Thread( target=self.watch_compute_q, daemon=True )
    self.wq.start()

    # start path optm 
    self.wp = threading.Thread( target=self.check_path_optm,  daemon=True )
    self.wp.start()

  def handle_pcep_event(self, ev):
    t   = ev["type"]
    pcc = ev["pcc"]

    if t == "PCC_SYNC":
      print("PCC_SYNC")
      if pcc not in self.PCC.keys():
        self.PCC[pcc] = {}
      self.PCC[pcc]["state"] = "sync"
      #self.PCC[pcc]["info"]  = ev["info"]
      self.PCC[pcc]["info"]  = ev["info"]
    elif t == "PCC_DOWN":
      if pcc in self.PCC.keys():
        self.PCC.pop(pcc)
      #self.pathdelete_for_onepcc(pcc)
      qpri = ( 100, self.get_C_cnt())
      self.C_Queue.put((qpri,{
        "type": "DELETE_PATH_FOR_PCC",
        "pcc" : pcc,
      }))

    elif t == "PCEP_REPORT":
      repinfo = ev["info"]
      #print("PCEP_REPORT")
      #print(repinfo)
      self.handle_pcep_report(pcc, repinfo)

      

      #add or delete
    else:
      print(t)

  def delete_path_for_pcc(pcc):
    for p in self.PATH_n.keys():
      if pcc == self.PATH_n[p]["c"]["src"]:
        self.PATH_n.pop(p)

  def delete_p_from_d_for_onepcc(pcc):
    for p in list(self.PATH_d.keys()):
      if pcc == self.PATH_d[p]["pcc"]:
        qpri = ( 100, self.get_C_cnt())
        self.C_Queue.put((qpri,{
          "type": "DELETE_PATH_D",
          "comptype": p,
        }))
         

  def handle_pcep_report(self, pcc, repinfo):
    #ope      = repinfo["lsp"].get('ope',0)
    #create   = repinfo["lsp"].get('create',False)
    #delegate = repinfo["lsp"].get('delegate',False)
    #srpid    = repinfo["srp"].get('srpid',0)
    name     = repinfo["lsp"].get('name')

    qpri = ( 100, self.get_C_cnt())
    self.C_Queue.put((qpri,{
      "type": "UPDATE_PATH_D",
      "pcc" : pcc,
      "comptype": name,
      "detail"  : repinfo
    }))


    #print(self.PATH_d)
    

  def on_pcep_event(self, ev):
    self.handle_pcep_event(ev)
    #qpri = ( 100, self.get_C_cnt())
    #self.C_Queue.put(qpri, ev)

  def attach_PCEP_Q(self,X):
    self.pcep_queue = X
    print(self.pcep_queue)

  def attach_X(self,X):
    self.computeX = X

  def attach_BM(self,BM):
    self.BM = BM

  def attach_CM(self,CM):
    self.CM = CM

  def attach_GM(self,GM):
    self.GM = GM

  def get_C_cnt(self):
    return next(self.C_Cnt)

  def get_srpid(self):
    return next(self.PCC_srpid)

  def watch_compute_q(self):
    while True:
      pri, ev = self.C_Queue.get()
      self.handle_event(ev)

  def check_path_optm(self):

    while True:
      time.sleep(3.0)
      self.log.info("[PATH] PATH optm start")

      # check bgp state
      bgpls_status = self.GM.get_bgpls_active()
      if bgpls_status == False:
        continue

      now      = int(time.time() * 1000)
      # test move p to _d for debug need PCEP
      #self.debug_move_p()

      # all pah def
      allpathdef = self.CM.get_all_pathdef()
      sorted_pathdef1 = sorted( allpathdef, key=lambda p: (-p.pri, -p.bw) ) # for new
      sorted_pathdef2 = sorted( allpathdef, key=lambda p: (p.pri, -p.bw)  ) # for bw

      # del not sync pcep from wk
      #----------------------------------------
      #for p in list(self.PATH_n.keys()):
      #  src = self.PATH_n[p]["c"]["src"]
      #  wkdel = False
      #  for s in src:
      #    if s in self.PCC.keys():
      #      if self.PCC[s]["state"] != "sync":
      #        wkdel = True
      #    else:
      #      wkdel = True
      #  if wkdel == True:
      #    qpri = ( 100, self.get_C_cnt())
      #    self.C_Queue.put((qpri,{
      #      "type": "DELETE_PATH",
      #      "comptype": p,
      #    }))
      #    self.log.info("[PATH] delete PATH: " + p )
      #-----------------------------------------
      # del not in path def
      #-----------------------------------------
      for k in list(self.PATH_n.keys()):
        # only initiate
        if self.PATH_n[k]["d"] != {}:
          if self.PATH_n[k]["d"]["pathinfo"]["lsp"]["create"] == True:
            #if self.PATH_d[k]["initiate"] == True:
            wk = self.CM.get_one_pathdef(k)
            if wk == None:
              if self.PATH_n[p]["wk"]["status"] != "Init":
                qpri = ( 100, self.get_C_cnt())
                self.C_Queue.put((qpri,{
                  "type": "DELETE_PATH",
                  "comptype": k,
                }))

      # BW check
      #-----------------------------------------
      allbwinfo   = self.BM.getallbw()
      cpath       = set()
      for lk in list(allbwinfo.keys()):
        v = allbwinfo.get(lk)
        if v!= None:
          reducebw = v["sumpbw"] - v["maxbw"]
          #reducebw = 5 # debug
          if ( reducebw > 0  ):
            wkbw = 0
            for pd in sorted_pathdef2:
              if pd.name in self.PATH_d.keys():
                if self.PATH_d[pd.name]["bw"] > 0:
                  if lk in self.PATH_d[pd.name]["link_set"]:
                    cpath.add(pd.name)
                    wkbw += self.PATH_d[pd.name]["bw"]
                    if wkbw > reducebw:
                      break
      for p in cpath:
        qpri = ( 100, self.get_C_cnt())
        self.C_Queue.put((qpri,{
            "type": "RECOMPUTEX",
            "comptype": p,
        }))

      #-----------------------------------------
      # optm
      #-----------------------------------------
      self.log.info("[PATH] timer optimize start")
      for k in self.PATH_n.keys():
        # only initiate LSP
        #if "pathinfo" in self.PATH_n[k]["d"].keys():
        #self.log.info("[PATH] timer optimize :" + str(k))
        if self.PATH_n[k]["d"] != {}:

          if self.PATH_n[k]["d"]["pathinfo"]["lsp"]["create"] == True:
            #now      = int(time.time() * 1000)
            pathinfo = self.PATH_n[k]["c"]["pathdef"]
            optmtime = 10

            # up
            if self.PATH_n[k]["d"]["pathinfo"]["lsp"]["ope"] == 2:
              calctime = int(self.PATH_n[k]["c"]["updatetime"])
              optmtime = pathinfo.optm
            else:
              calctime = int(self.PATH_n[k]["w"]["calctime"])

            if optmtime != 0:
              optmtime = max(optmtime * 1000, 10000)
              if now - calctime > optmtime:
                self.log.info("[PATH] timer optimize :" + str(k) + " compute start")
                qpri = ( 100, self.get_C_cnt())
                self.C_Queue.put((qpri,{
                  "type": "RECOMPUTEX",
                  "comptype": k,
                }))
              else:
                self.log.info("[PATH] timer optimize :" + str(k) + " compute skip")
            else:
              self.log.info("[PATH] timer optimize :" + str(k) + " compute skip")

        elif self.PATH_n[k]["w"] != {}:
          #self.log.info("[PATH] timer optimize :" + str(self.PATH_n[k]["w"]))
          optmtime = 10
          calctime = int(self.PATH_n[k]["w"]["pathinfo"]["calctime"])
          optmtime = max(optmtime * 1000, 10000)
          if now - calctime > optmtime:
            self.log.info("[PATH] timer optimize :" + str(k) + " compute start")
            qpri = ( 100, self.get_C_cnt())
            self.C_Queue.put((qpri,{
              "type": "RECOMPUTEX",
              "comptype": k,
            }))
          else:
            self.log.info("[PATH] timer optimize :" + str(k) + " compute skip")

      #-----------------------------------------
      # force recalc ( 300 sec over for INIT)
      #-----------------------------------------
      self.log.info("[PATH] force recalc start")
      for k in self.PATH_n.keys():
        if self.PATH_n[k]["w"] != {}:
          if self.PATH_n[k]["w"]["status"] == "Init":
            calc = int(self.PATH_n[k]["w"]["pathinfo"]["calctime"])
            if now - calctime > 300000:
              self.log.info("[PATH] force recalc :" + str(k))
              self.PATH_n[k]["w"] = {}

      #-----------------------------------------
      # New 
      #-----------------------------------------
      self.log.info("[PATH] optimize new start")
      for pd in sorted_pathdef1:
        #self.log.info("[PATH] optimize new: " + str(pd.name))
        src = pd.src
        wknew = False
        for s in src:
          if s in self.PCC.keys():
            if self.PCC[s]["state"] == "sync":
              wknew = True
        if wknew != True:
          self.log.info("[PATH] optimize new: " + str(pd.name) + " compute skip")
          continue

        wknew = False
        if pd.name not in self.PATH_n.keys():
          wknew = True
        else:
          #self.log.info("[PATH] optimize new:" + str(self.PATH_n[pd.name]))
          if "pathinfo" in self.PATH_n[pd.name]["d"].keys():
            if self.PATH_n[pd.name]["d"]["pathinfo"]["lsp"]["ope"] != 2:
              if self.PATH_n[pd.name]["w"] == {}:
                wknew = True
          else:
            if self.PATH_n[pd.name]["w"] == {}:
              wknew = True

        if wknew == True:
          self.log.info("[PATH] optimize new: " + str(pd.name) + " compute start")
        #if ( ( pd.name not in self.PATH_n.keys() ) 
        #  or ( self.PATH_n[pd.name]["w"] == None ) ):
        #  if pd.name not in self.PATH.keys():
          qpri = ( 100, self.get_C_cnt())
          self.C_Queue.put((qpri,{
            "type": "RECOMPUTEX",
            "comptype": pd.name,
          }))
        else:
          self.log.info("[PATH] optimize new: " + str(pd.name) + " compute skip")


  def remove_link(self, G, bw1, bw2, link_set):
    # bw check
    bwdiff = max(bw1 - bw2, 0)

    wkG = G
    for _, _, k in list(wkG.edges(keys=True)):
      bwflg = False
      if ( k in link_set ):
        bwflg = self.BM.chkbw(k,bwdiff)
      else:
        bwflg = self.BM.chkbw(k,bw1)
      if bwflg == False:
        wkG.remove_edge(k[0],k[1],k)

    return wkG

  def recompute_path_for_change(self):

    self.log.info("[COMPUTE] recompute path check start")

    normalized = self.CM.get_all_pathdef()
    paths_sorted = sorted(normalized, key=lambda p: (-p.pri, -p.bw))

    wk_G = {}

    # Link down check
    for pd in paths_sorted:
      wk_skip_flg = False

      self.log.info("[COMPUTE] compute check " + str(pd.name))

      if pd.name in self.PATH.keys():
        link_of_p     = self.PATH[pd.name]["link_set"]
        under_of_p    = self.PATH[pd.name]["underlay"]
        if under_of_p not in wk_G.keys():
          wk_G[under_of_p] = self.GM.get_one_graph(under_of_p)
        wk_del_flg = False
        for lkey in link_of_p:
          if not wk_G[under_of_p].has_edge(lkey[0],lkey[1],lkey):
            wk_del_flg = True
            break
        if wk_del_flg == False:
          wk_skip_flg = True

      if wk_skip_flg:
        self.log.info("[COMPUTE] compute skip")
        if pd.name in self.PATH.keys():
          self.PATH[pd.name]["opttime"] = int(time.time() * 1000)
        continue

      else:
        self.log.info("[COMPUTE] compute exec")
        qpri = ( 50, self.get_C_cnt() )
        self.C_Queue.put((qpri,{
          "type": "RECOMPUTEX",
          "comptype": pd.name
        }))

  def handle_event(self,ev):

    self.log.debug("[COMPUTE] HANDLE EVENT")

    ev_time = int(time.time() * 1000)

    #print(ev)

    ev_t = ev["type"]
    self.log.debug("[COMPUTE] event start "  + ev_t)

    if ev_t == "PATH_CONFIG":
      cid   = ev["diff"]["id"]
      difft = ev["diff"]["type"]
      if ( difft == DiffType.DEL ):
        pass
      elif ( difft == DiffType.ADD ):
        pass
      elif ( difft == DiffType.MOD ):
        qpri = ( 100, self.C_cnt())
        self.C_Queue.put((qpri,{
          #"type": "RECOMPUTE4",
          "type": "RECOMPUTEX",
          "comptype": cid,
        }))
  
    elif ev_t == "RECOMPUTEX":
      self.compute_pathsX(ev["comptype"])

    #elif ev_t == "DELETE_PATH":
    #  self.delete_p_from_d(ev["comptype"])
    elif ev_t == "DELETE_PATH_WK":
      self.delete_p_from_wk(ev["comptype"])

    elif ev_t == "UPDATE_PATH_D":
      #self.update_path_d(ev["comptype"], ev["detail"], ev["pcc"])
      self.update_path_d(ev["comptype"], ev["detail"], ev["pcc"])

    elif ev_t == "DELETE_PATH_D":
      self.delete_p_from_d(ev["comptype"])

    elif ev_t == "DELETE_PATH_FOR_PCC":
      self.delete_path_for_pcc(ev["pcc"])

    elif ev_t == "DELETE_PATH":
      self.delete_p(ev["comptype"])

    elif ev_t == "NWSYNC":
      self.do_set_link_set()

    #else:
    #  print("ev_t:" + str(ev_t))

  def set_link_set(self,p, info):
    wk = set()
    #print(info)
    u  = info["c"]["pathdef"].underlay
    wkflg = True
    for e in info["d"]["pathinfo"]["ero"]:
      lk = self.GM.check_edge(e, u)
      if lk == None:
        wkflg = False
        break
      else:
        wk.add(lk)

    if wkflg = True:
      self.PATH_n[p]["d"]["pathinfo"]["link_set"] = wk
    else:
      qpri = ( 100, self.get_C_cnt())
      self.C_Queue.put((qpri,{
          "type": "RECOMPUTEX",
          "comptype": p,
      }))
      


  def do_set_link_set(self):
    for p in self.PATH_n.keys():
      self.set_link_set(p, self.PATH_n[p])


  def delete_wkbw_base_p(self, p):
    if p in list(self.PATHN_n.keys()):
      for lk in self.PATH[p]["wk"]["link_set"]:
        self.BM.delwkpbw(lk,p)

  def update_path_d(self, p, info, pcc):

    remove = info["lsp"]["remove"]
    #if p not in self.PATH_d.keys():
    #  self.PATH_d[p] = {}
    if p not in self.PATH_n.keys():
      self.PATH_n[p] = {}
      self.PATH_n[p]["c"] = {}
      self.PATH_n[p]["d"] = {}
      self.PATH_n[p]["w"] = {}

    if remove == True:
      #self.PATH_n.pop(p)
      self.delete_wkbw_base_p(p)
      self.BM.delwkpbw(lk,p)
      self.PATH_n.pop(p)
      self.log.info("[PATH] path delete: " + str(p))
    else:
      self.PATH_n[p]["c"]["src"]    = pcc
      self.PATH_n[p]["c"]["updatetime"]  = int(time.time() * 1000)
      #self.PATH_n[p]["c"]["opttime"]     = int(time.time() * 1000)
      self.PATH_n[p]["d"]["pathinfo"] = info
      self.PATH_n[p]["d"]["pathinfo"]["bw"] = int(info["lsp"]["bandwidth"])
      self.PATH_n[p]["d"]["link_set"] = set()

      if self.PATH_n[p]["d"]["pathinfo"]["lsp"]["create"] == True:
        self.PATH_n[p]["c"]["pathdef"] = self.CM.get_one_pathdef(p)
        bgpls_status = self.GM.get_bgpls_active()
        if ((bgpls_status == True)
          and (self.PATH_n[p]["c"]["pathdef"] != None)):
          self.set_link_set(p, self.PATH_n[p])
 
      #self.PATH_d[p]["pathinfo"] = info
      #self.PATH_d[p]["updatetime"]  = int(time.time() * 1000)
      #self.PATH_d[p]["opttime"]     = int(time.time() * 1000)
      #self.PATH_d[p]["pcc"]         = pcc

      self.log.info("[PATH] path update: " + str(self.PATH_n[p] ))
      srpid  = info["srp"]["srpid"]

      #name   = info["lsp"]["name"]
      #remove = info["lsp"]["remove"]

      if srpid != 0:
        if ( self.PATH_n[p]["w"] != []):
          self.PATH_n[p]["w"]["status"] = "Done"

      #self.PATH_n[p]["w"]["status"] = "Done"

      #if name in self.PATH.keys():
      #if name in self.PATH.keys():
      #  if srpid == self.PATH[p]["srpid"]:
      #    self.log.info("[PATH] delete from wk due to report " + name)
      #    self.delete_p_from_wk(name)


  def delete_p_from_both(self, p): # force delete
    if p in list(self.PATH_d.keys()):
      self.PATH_d.pop(p)
    if p in list(self.PATH.keys()):
      self.PATH.pop(p)

  #------------------
  def delete_p(self, p): 
    if p in list(self.PATH_n.keys()):
      srpid = self.get_srpid()
      #wk_results["srpid"]  = srpid
      self.PATH_n[p]["wk"]["status"] = "Init"
      self.pcep_queue.put({
        "type": "PATH DELETE",
        "src" :  self.PATH_n[p]["c"]["src"],
        "name":  p,
        "detail"  : self.PATH_n[p]["d"]["pathinfo"],
        "srpid": srpid
      })

  #def delete_p_from_d(self, p):
  #  if p in list(self.PATH_d.keys()):
  #    if self.PATH_d[p]["initiate"] == True:
  #      if p not in self.PATH.keys():
  #        for lk in self.PATH_d[p]["link_set"]:
  #          self.BM.delpbw(lk,p)
  #        self.PATH_d.pop(p)
  def delete_p_from_d(self, p):
    if p in list(self.PATH_d.keys()):
      self.PATH_d.pop(p)

  def delete_p_from_wk(self, p):
    if p in list(self.PATH.keys()):
      for lk in self.PATH[p]["link_set"]:
        self.BM.delwkpbw(lk,p)
      self.PATH.pop(p)

  def debug_move_p(self):
    for p in list(self.PATH.keys()):
      self.PATH_d[p] = self.PATH[p]

      for lk in list(self.PATH_d[p]["link_set"]):
        self.BM.addpbw(lk,self.PATH_d[p]["bw"],p)

      for lk in list(self.PATH[p]["link_set"]):
        self.BM.delwkpbw(lk,p)

      self.PATH.pop(p)

  #-----------------------------------------------
  # compute
  #-----------------------------------------------
  def compute_pathsX(self, pid):
    self.log.info("[COMPUTE] computeX rtn start for " + pid)

    #if self.PATH.get(pid) != None:
    #  self.log.info("[COMPUTE] skip there is other candidate")
    #  return
    if pid in self.PATH_n.keys():
      if self.PATH_n[pid]["w"] != {}:
        if self.PATH_n[pid]["w"]["status"] == "Init":
          self.log.info("[COMPUTE] skip there is other candidate(d status = init")
          return

    cpathinfo  = None
    if pid in self.PATH_n.keys():
      if self.PATH_n[pid]["d"] != {}:
        if self.PATH_n[pid]["d"]["lsp"]["ope"] == 2:
          cpathinfo = self.PATH_n[pid]["d"]["pathinfo"]
      if cpathinfo == None:
        if self.PATH_n[pid]["w"] != {}:
          cpathinfo = self.PATH_n[pid]["w"]["pathinfo"]

    pd = self.CM.get_one_pathdef(pid)

    self.log.debug("[COMPUTE] path name: "  + str(pid))
    self.log.debug("[COMPUTE] path def:  "  + str(pd))

    wkG         = None
    wkG_t       = None
    wk_skip_flg = False

    #graph / linkinfo:
    ( wkG, wkG_t ) = self.GM.get_one_graph_infos(pd.underlay)
    bef_g_time  = wkG_t

    #bwdiff = 0
    if cpathinfo != None:
      removed_wkG  = self.remove_link(
        wkG, pd.bw, cpathinfo["bw"], cpathinfo["link_set"]
      )
    else:
      #new
      removed_wkG  = self.remove_link(
                       wkG, pd.bw, 0, set()
      )

    removed_rwkG = removed_wkG.reverse(copy=False)
    wk_results = {}

    self.log.info("[COMPUTE] compute start for " + pd.name)

    p, link_set  = self.build_paths(pd, removed_wkG, removed_rwkG)

    self.log.info("[COMPUTE] compute end for" + pd.name)

    wk_results["time"]      = wkG_t
    wk_results["initiate"]  = True
    wk_results["opttime"]   = int(time.time() * 1000)
    wk_results["calctime"]   = int(time.time() * 1000)
    wk_results["underlay"]  = pd.underlay
    wk_results["bw"]        = pd.bw
    wk_results["detail"]    = p
    wk_results["src"]       = pd.src
    wk_results["link_set"]  = link_set

    aft_g_time  = self.GM.get_last_g_time()

    if bef_g_time != aft_g_time:
      self.log.info("[COMPUTE] NW change happend during compute")
      return
    else:
      if pd.name in self.PATH_n.keys():
        if ( wk_results["detail"] == self.PATH_n[pd.name]["w"]["pathinfo"]["detail"] ):
          self.log.info("[COMPUTE] calc results same as bef")
          self.PATH_n[pd.name]["w"]["pathinfo"]["calctime"] = int(time.time() * 1000)
          return

      self.log.info("[COMPUTE] calc results")
      self.log.info("[COMPUTE] new path" + str(wk_results))

    # add
    srpid = self.get_srpid()
    wk_results["srpid"]  = srpid

    for wk_l in wk_results["link_set"]:
      if cpathinfo != None:
        if wk_l in cpathinfo["link_set"]:
          bwdiff = max(0, wk_results["bw"] - cpathinfo["bw"] )
          self.BM.addwkpbw(wk_l, bwdiff,pd.name)
        else:
          self.BM.addwkpbw(wk_l, wk_results["bw"],pd.name)
      else:
        self.BM.addwkpbw(wk_l, wk_results["bw"],pd.name)

    if pd.name not in self.PATH.keys():
      self.PATH[pd.name] = {}

    if pd.name not in self.PATH_n.keys():
      self.PATH_n[pd.name] = {}
      self.PATH_n[pd.name]["c"] = {}
      self.PATH_n[pd.name]["d"] = {}
      self.PATH_n[pd.name]["w"] = {}

    self.PATH[pd.name] = wk_results

    self.PATH_n[pd.name]["c"]["pathdef"]  = pd
    self.PATH_n[pd.name]["c"]["src"]      = pd.src
    self.PATH_n[pd.name]["w"]["pathinfo"] = wk_results
    self.PATH_n[pd.name]["w"]["status"]   = "Init"
    self.PATH_n[pd.name]["w"]["pathinfo"]["calctime"] = int(time.time() * 1000)

    #print(type(wk_results["detail"][pd.src]))
    #print(wk_results["detail"][pd.src])

    if wk_results["detail"][pd.src[0]] != {}: 
      self.pcep_queue.put({
        "type": "PATH UPDATE",
        "src" :  pd.src[0],
        "name":  pd.name,
        "detail"  : self.PATH_n[pd.name]["w"]["pathinfo"]
      })
    else:
      self.PATH_n[pd.name]["w"]["status"]  = "No path"

    self.log.info("[PATH] last_pathinfo: " + pd.name)
    self.log.info("[PATH] last pathinfo: " + str(self.PATH_n[pd.name]))

    #return True

  def build_paths(self, pd, Gc, Gc_rev):
    print(Gc)
    results  = {}
    link_set = set()
    src_set  = []
    dst_set  = []

    if ( pd.type == "p2p" ):

      for src in pd.src:
        results[src] = {}
        dist_src = nx.single_source_dijkstra_path_length( Gc, src, weight="cost" )

        for dst in pd.dst:
          if src == dst:
            continue

          best = dist_src.get(dst)
          if best is None:
            continue

          dist_dst = nx.single_source_dijkstra_path_length( Gc_rev, dst, weight="cost" )
          paths = self.k_physical_paths_visited(
            Gc, src, dst, pd.K, pd.delta, dist_dst, best, pd.mode
          )

          results[src][dst]   = paths
          for r in paths:
            for r2 in r["links"]:
              link_set.add(r2)
    else:
    # p2mp

      print(pd)

      src = pd.src[0]
      (dest_dist, parent0) = self.dijkstra_to_dests(Gc, src, pd.dst)
      o_dests = sorted(dest_dist, key=lambda d: dest_dist[d], reverse=True)

      if len(o_dests) == 0:
        print("o_dests")
        return {}

      tree_edges = set()
      tree_nodes = set()

      # base tree ( to farest node )
      first = o_dests[0]
      path = self.build_one_path(parent0, src, first)

      for u, v, key in path:
        tree_edges.add((u, v, key))
        tree_nodes.add(u)
        tree_nodes.add(v)


      # join to tree for 2nd- node
      for d in o_dests[1:]:
        if d in tree_nodes:
          continue

        dist2, parent2 = self.dijkstra_from_tree(Gc, tree_nodes, tree_edges)

        # back to tree from d
        u = d
        path = []

        #if u in tree_nodes:
        #  continue

        while u not in tree_nodes:
          pi, key = parent2[u]
          path.append((pi, u, key))
          u = pi

        path.reverse()

        for e in path:
          tree_edges.add(e)
          tree_nodes.add(e[0])
          tree_nodes.add(e[1])

      for e in tree_edges:
        link_set.add(e[2])

      tree = self.tree_edges_to_json(tree_edges, src)

      results = tree

    return results, link_set


  def ecmp_paths(self, results, k):
    if not results:
        return []
    best = results[0]["cost"]
    ecmp = [r for r in results if r["cost"] == best]
    return ecmp[:k]

  def k_paths(self, results, k):
    return results[:k]

  def k_physical_paths_visited(self, DG, src, dst, K, delta, dist_dst, best, mode):
    results   = []
    sum_links = []

    def dfs(u, cost, path, links, visited):
        # prune
        if cost + dist_dst.get(u, float("inf")) > best + delta:
            return

        if u == dst:
            results.append({
                "cost": cost,
                "path": path + [u],
                #"links": links.copy()
                "links": list(links)
            })
            return

        for _, v, k, data in DG.out_edges(u, keys=True, data=True):
            if v in visited:
                continue

            visited.add(v)
            path.append(u)
            links.append(k)

            dfs(
                v,
                cost + data["cost"],
                path,
                links,
                visited
            )

            links.pop()
            path.pop()
            visited.remove(v)

    dfs(src, 0, [], [], {src})

    results.sort(key=lambda x: (x["cost"], len(x["path"])))

    if ( mode == "ecmp" ):
      R = self.ecmp_paths(results, K)
    else:
      R = self.k_paths(results,K)

    return R

  def dijkstra_to_dests(self, G, src, dests):
    dests = set(dests)
    found = {}
    parent = {}
    dist = {src: 0}
    pq = [(0, src)]

    while pq and dests:
      cost_u, u = heapq.heappop(pq)

      if cost_u > dist[u]:
        continue

      if u in dests:
        found[u] = cost_u
        dests.remove(u)
        # continue since there are another dests

      for v, keydict in G[u].items():
        for key, attr in keydict.items():
          w = attr["cost"]
          new = cost_u + w
          if v not in dist or new < dist[v]:
              dist[v] = new
              parent[v] = (u, key)
              heapq.heappush(pq, (new, v))

    return found, parent

  def build_one_path(self, parent, src, dest):
    path = []
    u = dest
    while u != src:
        p, key = parent[u]
        path.append((p, u, key))
        u = p
    return list(reversed(path))

  def dijkstra_from_tree(self, G, tree_nodes, tree_edges):

    dist = {}
    parent = {}
    pq = []

    for n in tree_nodes:
        dist[n] = 0
        pq.append((0, n))

    heapq.heapify(pq)

    while pq:
        cost_u, u = heapq.heappop(pq)
        if cost_u > dist[u]:
            continue

        for v, keydict in G[u].items():
            for key, attr in keydict.items():
                w = attr["cost"]

                #if (u, v, key) in tree_edges: #
                #    #w *= 0.5
                #    w *= 1.0

                new = cost_u + w
                if v not in dist or new < dist[v]:
                    dist[v] = new
                    parent[v] = (u, key)
                    heapq.heappush(pq, (new, v))

    return dist, parent

  def tree_edges_to_json(self, tree_edges, src):
    adj = self.build_adj(tree_edges)
    return {src: self.build_tree_json(adj, src)}

  def build_adj(self, tree_edges):
    adj = defaultdict(list)
    for u, v, key in tree_edges:
        adj[u].append((v, key))
        adj[v].append((u, key))
    return adj

  def build_tree_json(self, adj, u, parent=None):
    node  = {}
    node["children"]  = []

    #for v, key, cost in adj[u]:
    for v, key in adj[u]:
        if v == parent:
            continue

        a = self.build_tree_json(adj, v, u)

        if a == None:
          node["children"].append({
              "node" : v,
              "link" : key,
          })
        else:
          node["children"].append({
              "node" : v,
              "link" : key,
              **a
          })

    if node["children"] == []:
      return None

    return node

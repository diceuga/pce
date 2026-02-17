#import logging
#log = logging.getLogger("bgpls")

class BWManager:
  # - link
  #   - maxbw:bw
  #   - unrsvbw: bw
  #   - paths
  #     - name:bw
  #   - wkpaths
  #     - name:bw
  #   - sumpbw: bw
  #   - sumwkpbw: bw

  def __init__(self, log):
    self.log = log
    self.BWdb = {}

  def addpbw(self, lkey, bw, pname):
    self.BWdb[lkey]["paths"][pname] = bw
    self.BWdb[lkey]["sumpbw"] += bw

  def addwkpbw(self, lkey, bw, pname):
    self.BWdb[lkey]["wkpaths"][pname] = bw
    self.BWdb[lkey]["sumwkpbw"] += bw

  def delpbw(self, lkey, bw, pname):
    if lkey in self.BWdb.keys():
      self.BWdb[lkey]["paths"].pop(pname)
      self.BWdb[lkey]["sumpbw"] -= bw

  def delwkpbw(self, lkey, pname):
    if lkey in self.BWdb.keys():
      wkbw = self.BWdb[lkey]["wkpaths"][pname]
      self.BWdb[lkey]["wkpaths"].pop(pname)
      self.BWdb[lkey]["sumwkpbw"] -= wkbw

  def updbw(self, lkey, bw, mbw):
    if lkey not in self.BWdb.keys():
       self.BWdb[lkey]            = {}
       self.BWdb[lkey]["maxbw"]   = mbw
       self.BWdb[lkey]["unrsvbw"] = bw
       self.BWdb[lkey]["sumpbw"]  = 0
       self.BWdb[lkey]["paths"]   = {}
       self.BWdb[lkey]["sumwkpbw"]  = 0
       self.BWdb[lkey]["wkpaths"]   = {}
    else:
       self.BWdb[lkey]["unrsvbw"] = bw
       self.BWdb[lkey]["maxbw"]   = mbw

  def delbw(self, lkey):
    if lkey in self.BWdb.keys():
       self.BWdb.pop(lkey)

  #def updpbw(self, lkey, pkey, bw):
  #  if lkey in self.BWdb.keys():
  #    self.BWdb[lkey]["unrsvbw"] = bw

  def chkbw(self, lkey, bw):
    ebw = min(self.BWdb[lkey]["unrsvbw"], self.BWdb[lkey]["maxbw"] - self.BWdb[lkey]["sumpbw"])
    if ( ebw - bw ) >= 0:
      return True
    else:
      return False

  def getbw(self, lkey):
    if lkey in self.BWdb.keys():
      return self.BWdb[lkey]["unrsvbw"]
    else:
      return None

  def getallbw(self):
    return self.BWdb

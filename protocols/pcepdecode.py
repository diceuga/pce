# protocols/pcepdecode.py
import struct
import socket

OBJ_OPEN = 1

TLV_STATEFUL_PCE_CAP = 16
TLV_SR_PCE_CAP       = 26
TLV_PCE_INITIATE_CAP = 27
TLV_PCECC_CAP        = 32

def decode_pcep_tlvs(buf: bytes):
  off = 0
  end = len(buf)
  #tlvs = []
  res = {}
  res["tlvs"] = {}
  resb = b''

  while off + 4 <= end:
    t, l = struct.unpack("!HH", buf[off:off+4])
    tlvh = buf[off:off+4]
    off += 4

    if off + l > end:
      raise ValueError("PCEP TLV length overflow")

    v = buf[off:off+l]
    off += l

    #tlvs.append({
    #      "type": t,
    #      "length": l,
    #      "value": v
    #})

    if t == TLV_STATEFUL_PCE_CAP:
        flags = struct.unpack("!I", v)[0]
        res["tlvs"]["stateful"] = {
            "capable": True,
            "flags": {
                "LSP-Update":       bool(flags & (1 << 0)),
                "INCLUDE-DB-V":     bool(flags & (1 << 1)),
                "LSP-Instantiate":  bool(flags & (1 << 2)),
                "Triggered-ReSync": bool(flags & (1 << 3)),
                "Delta-LSP-Sync":   bool(flags & (1 << 4)),
                "Triggerd-InitSync":bool(flags & (1 << 5)),
                "P2MP-Capa"        :bool(flags & (1 << 6)),
            }
        }
        resb += tlvh + v


    elif t == TLV_SR_PCE_CAP:
        flags = struct.unpack("!I", v)[0]
        res["tlvs"]["sr"] = {
            "capable": True,
            "naive": bool(flags & (1 << 0)),
            "msd": flags >> 16
        }
        resb += tlvh + v
        #resb += v

    #elif t == TLV_PCE_INITIATE_CAP:
    #    res["tlvs"]["pce_initiate"] = {
    #        "capable": True
    #    }

    #elif t == TLV_PCECC_CAP:
    #    flags = struct.unpack("!I", v)[0]
    #    res["tlvs"]["pcecc"] = {
    #        "capable": True,
    #        "flags": flags
    #    }
    elif t == 34:
      #resb += v
      resb += tlvh + v
      pass
    elif t == 35:
      #resb += v
      resb += tlvh + v
      pass
    elif t == 7:
      resb += tlvh + v
      #resb += v
      pass
    elif t == 60:
      resb += tlvh + v

    else:
        #res["tlvs"][f"unknown_{t}"] = v
        print("unknow tlv type:" + str(t)) 

  return res, resb

  #return tlvs

def decode_pcep_report_tlvs(buf):
  off     = 0
  end     = len(buf)
  tlvs    = []

  while off + 4 <= end:
    t, l = struct.unpack("!HH", buf[off:off+4])
    off += 4

    if off + l > end:
        raise ValueError("PCEP Report TLV overflow")

    v = buf[off:off+l]
    off += l

    tlvs.append({
        "type": t,
        "length": l,
        "value": v
    })

  return tlvs

OBJ_LSP        = 32
OBJ_SRP        = 33
OBJ_ERO        = 7
OBJ_BANDWIDTH  = 5

TLV_LSP_NAME   = 17   # Symbolic Path Name
TLV_SRP_ID     = 20   # SRP-ID TLV (ベンダ実装により異なる場合あり)

def decode_pcep_report(payload: bytes):
    #print(payload)
    off = 0
    res = {
        "lsp": {},
        "srp": {},
        "ero": [],
        "tlvs": {}
    }

    while off + 4 <= len(payload):
        obj_class, obj_type, length = struct.unpack("!BBH", payload[off:off+4])
        body = payload[off+4:off+length]
        #print(body)
        off += length

        # ---- LSP Object ----
        if obj_class == OBJ_LSP:
            #flags = struct.unpack("!I", body[0:4])[0]
            #res["lsp"]["delegated"] = bool(flags & (1 << 31))
            #res["lsp"]["sync"]      = bool(flags & (1 << 30))
            #res["lsp"]["remove"]    = bool(flags & (1 << 29))
            #res["lsp"]["plsp_id"] = struct.unpack("!I", body[4:8])[0]
            fix   = struct.unpack("!HH", body[0:4])
            #print(fix[0])
            #print(fix[0] << 4)
            #print(fix[1])
            #print(fix[1] >> 12)

            plsp_id = (fix[0] << 4) + ( fix[1] >> 12 )
            #status = struct.unpack("!I", body[8:12])[0]

            delegate = bool(fix[1] & 1)
            sync     = bool(fix[1] & (1 << 1))
            remove   = bool(fix[1] & (1 << 2))
            admin    = bool(fix[1] & (1 << 3))
            ope      = (fix[1] >> 4) & 7
            create   = bool(fix[1] & (1 << 7))
            p2mp     = bool(fix[1] & (1 << 8))
            frag     = bool(fix[1] & (1 << 9))
            erocomp  = bool(fix[1] & (1 << 10))
            pce      = bool(fix[1] & (1 << 11))


            res["lsp"]["sync"]  = sync
            res["lsp"]["delegete"]  = delegate
            res["lsp"]["remove"]  = remove
            res["lsp"]["admin"]  = admin
            res["lsp"]["ope"]  = ope
            res["lsp"]["create"] = create
            res["lsp"]["p2mp"] = p2mp
            res["lsp"]["frag"] = frag
            res["lsp"]["erocmp"] = erocomp
            res["lsp"]["pce"] = pce
            res["lsp"]["flags"] = fix[1]
            res["lsp"]["plsp_id"] = plsp_id
            #res["lsp"]["status"] = status

            #rest = body[12:]
            rest = body[4:]

            #tlvs = decode_pcep_report_tlvs(body[8:])
            # 最低 8 byte までは fixed
            tlvs = decode_pcep_report_tlvs(rest)

            print(tlvs)

            for tlv in tlvs:
                if tlv["type"] == TLV_LSP_NAME:
                    res["lsp"]["name"] = tlv["value"].decode(errors="ignore")
                else:
                    res["tlvs"].setdefault("lsp", []).append(tlv)

        # ---- SRP Object ----
        elif obj_class == OBJ_SRP:
            flags = struct.unpack("!I", body[0:4])[0]
            res["srp"]["LSP-remove"]   = flags & 1
            res["srp"]["LSP-Con req"]  = flags & (1 < 1)


            res["srp"]["flags"]  = struct.unpack("!I", body[0:4])[0]
            res["srp"]["srp_id"] = struct.unpack("!I", body[4:8])[0]
            tlvs = decode_pcep_report_tlvs(body[8:])
            res["srp"]["tlvs"] = tlvs

        # ---- ERO Object ----
        elif obj_class == OBJ_ERO:
            ero = []
            p = 0
            while p + 2 <= len(body):
                sub_t = body[p]
                sub_l = body[p+1]
                sub_v = body[p+2:p+2+sub_l]
                p += 2 + sub_l

                if sub_t == 1 and sub_l >= 6:
                    addr = socket.inet_ntoa(sub_v[0:4])
                    plen = sub_v[4]
                    ero.append(f"{addr}/{plen}")
                elif sub_t == 2 and sub_l >= 4:
                    addr = socket.inet_ntoa(sub_v[0:4])
                    ero.append(addr)
                else:
                    ero.append(f"unknown({sub_t})")

            res["ero"] = ero

        # ---- Bandwidth ----
        elif obj_class == OBJ_BANDWIDTH:
            if len(body) == 4:
                res["lsp"]["bandwidth"] = struct.unpack("!f", body)[0]

    return res



def decode_pcep_open(payload):
  off     = 0
  res     = {}
  resopen = b''
  #print(payload)
  while off + 4 <= len(payload):
    obj_class, obj_type, length = struct.unpack("!BBH", payload[off:off+4])
    body = payload[off+4:off+length]
    off += length

    resopen += body[0:4]

    if obj_class == OBJ_OPEN:
      # OPEN Object Body:
      #  0               1               2               3
      # +---------------+---------------+---------------+---------------+
      # |Ver| Flags     | Keepalive     | DeadTimer     | SID           |
      # +---------------+---------------+---------------+---------------+
      ver_flags = body[0]
      keepalive = body[1]
      deadtimer = body[2]
      sid = body[3]

      res["version"] = (ver_flags >> 5) & 0x07
      res["flags"] = ver_flags & 0x1F
      res["keepalive"] = keepalive
      res["deadtimer"] = deadtimer
      res["sid"] = sid

      tlvs,tlvb = decode_pcep_tlvs(body[4:])
      #print(tlvs)
      res["tlvs"] = tlvs


    else:
      print("unknown open obj_class" + str(obj_class))

    return res, resopen + tlvb
  

def decode_pcrpt(payload: bytes):
    """
    ここでは最小構成で:
      - LSP object
      - PLSP-ID
      - PCC Router-ID
    くらいを抜く想定
    """
    return {
        "pcc": "10.0.0.1",
        "plsp_id": 100,
        "state": "UP",
        "ero": ["1.1.1.1", "2.2.2.2"]
    }


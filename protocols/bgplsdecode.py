import struct
from typing import Dict, Tuple, List
import logging

log = logging.getLogger()

AFI_BGPLS = 16388
SAFI_BGPLS = 71
AFI_BGP_LS = 16388
SAFI_BGP_LS = 71

PROTO_ISIS = 2
PROTO_OSPF = 3

NLRI_NODE   = 1
NLRI_LINK   = 2
NLRI_PREFIX = 3


def decode_bgp_update(payload: bytes):
    #print("BGP-LS-DECODE    bgp_update")
    off = 0
    result = {
        "withdraw": [],
        "announce": [],
        "ls_attributes": {},
        "path_attributes": [],
    }

    # --- Withdrawn Routes ---
    wlen = struct.unpack("!H", payload[off:off+2])[0]
    off += 2
    if wlen:
        result["withdraw"].append(payload[off:off+wlen])
        off += wlen

    # --- Path Attributes ---
    pattr_len = struct.unpack("!H", payload[off:off+2])[0]
    off += 2
    pattr_end = off + pattr_len

    #attrs = payload[pos:pos+plen]
    #pend = pos 
    #pos += plen

    ap = 0
    while off < pattr_end:
      attr, off = decode_path_attribute(payload, off)
      #print(attr)
      if attr:
        #result["path_attributes"].append(attr)
        if attr["type"] == 14:  # MP_REACH_NLRI
          result["announce"].extend(attr["nlri"])
        elif attr["type"] == 15:  # MP_UNREACH_NLRI
          result["withdraw"].extend(attr["nlri"])
        elif attr["type"] == 29:  # MP_UNREACH_NLRI
          result["ls_attributes"] = attr["ls_attr"]
        else:
          result["path_attributes"].append(attr)

    return result

def decode_path_attribute(buf: bytes, off: int) -> Tuple[Dict, int]:
    flags = buf[off]
    attr_type = buf[off + 1]
    off += 2

    if flags & 0x10:  # Extended Length
        length = struct.unpack("!H", buf[off:off+2])[0]
        off += 2
    else:
        length = buf[off]
        off += 1

    value = buf[off:off+length]
    off += length

    #print(f"[ATTR] type={attr_type} flags=0x{flags:02x} len={length}")

    if attr_type in (14, 15):  # MP_REACH / MP_UNREACH
        return decode_mp_nlri(attr_type, value), off
    if attr_type == 29:
        return {
          "type": 29,
          "ls_attr": decode_ls_attributes(value)}, off
        #return decode_ls_attributes(value), off

    return {"type": attr_type}, off



TLV_IGP_METRIC           = 1095
TLV_TE_METRIC            = 1092
TLV_MAX_LINK_BW          = 1089
TLV_MAX_RESERVABLE_BW    = 1090
TLV_UNRESERVED_BW        = 1091

def decode_ls_attributes(buf: bytes):
    """
    Decode BGP-LS Attribute (type 29)
    return: dict
    """
    #print("Decode BGP-LS Attribute (type 29)")
    #print(buf)
    off = 0
    attrs = {}

    while off + 4 <= len(buf):
        tlv_type, tlv_len = struct.unpack("!HH", buf[off:off+4])
        off += 4

        value = buf[off:off+tlv_len]
        off += tlv_len

        #print(tlv_type)

        if tlv_type == TLV_IGP_METRIC:
            if tlv_len == 2:
                attrs["igp_metric"] = struct.unpack("!H", value)[0]

        elif tlv_type == TLV_TE_METRIC:
            if tlv_len == 4:
                attrs["te_metric"] = struct.unpack("!I", value)[0]

        elif tlv_type == TLV_MAX_LINK_BW:
            # RFC9552: 4-octet IEEE float
            #print(bytes(value))
            if tlv_len == 4:
                attrs["max_link_bw"] = int(struct.unpack("!f", value)[0]) * 8

        elif tlv_type == TLV_MAX_RESERVABLE_BW:
            if tlv_len == 4:
                attrs["max_reservable_bw"] = int(struct.unpack("!f", value)[0]) * 8

        elif tlv_type == TLV_UNRESERVED_BW:
            # 8 priorities × 4 octets
            if tlv_len == 32:
                attrs["unreserved_bw"] = list(
                    struct.unpack("!8f", value)
                )
                for i in range(len(attrs["unreserved_bw"])):
                  attrs["unreserved_bw"][i] = int(attrs["unreserved_bw"][i]) * 8
                 

        else:
          log.info("[BGPLS] unsupport ls attribute tlv: " + str(tlv_type))
        # 他の TLV は無視（将来拡張）

    return attrs


def decode_mp_nlri(attr_type: int, value: bytes) -> Dict:
    #print("[BGP-LS-DECODE decode mp nlri start")
    off = 0
    afi, safi = struct.unpack("!HB", value[:3])
    off += 3

    if afi != AFI_BGP_LS or safi != SAFI_BGP_LS:
        return {"type": attr_type, "nlri": []}

    if attr_type == 14:  # MP_REACH
        nh_len = value[off]
        off += 1 + nh_len
        snpa = value[off]
        off += 1  # SNPA count

    nlris = []
    #print("NLRI LEN:" + str(len(value)))
    while off < len(value):
        #print("OFFSET:" + str(off))
        #nlri, off = decode_bgpls_nlri(value, off)
        #print("----------------------------- BGP LS NLRI DECODE START")
        nlri, off = decode_bgp_ls_nlri(value, off)
        #print("----------------------------- BGP LS NLRI DECODE END")
        #print("OFFSET:" + str(off))
        nlris.append(nlri)

    return {
        "type": attr_type,
        "afi": afi,
        "safi": safi,
        "nlri": nlris,
    }

#def decode_bgpls_nlri(buf: bytes, off: int) -> Tuple[Dict, int]:
#    nlri_type = buf[off]
#    length = struct.unpack("!H", buf[off+1:off+3])[0]
#    off += 3
#
#    nlri_data = buf[off:off+length]
#    off += length
#
#    protocol_id = nlri_data[0]
#    identifier = struct.unpack("!Q", nlri_data[1:9])[0]
#
#    tlvs = decode_tlvs(nlri_data[9:])
#
#    nlri = {
#        "nlri_type": nlri_type,
#        "protocol_id": protocol_id,
#        "identifier": identifier,
#        #"tlvs": tlvs,
#    }
#
#    # NLRI Type ごとの詳細解析（外出し）
#    nlri["detail"] = decode_nlri_detail(nlri_type, protocol_id, tlvs)
#
#    return nlri, off

def decode_bgp_ls_nlri(buf: bytes, off: int):
    start = off
    end = len(buf)

    if off + 13 > end:
      raise ValueError("NLRI header truncated")

    # NLRI Type (2 bytes)
    nlri_type, nlri_len = struct.unpack("!HH", buf[off:off+4])
    off += 4

    nlri_end = start + 4 + nlri_len

    # Protocol-ID (1 byte)
    protocol_id = buf[off]
    off += 1

    # Identifier (8 bytes)
    identifier = struct.unpack("!Q", buf[off:off+8])[0]
    off += 8

    # TLVs
    tlvs, a  = decode_tlvs(buf[off:nlri_end])

    off = nlri_end

    nlri = {
        "nlri_type": nlri_type,
        "protocol_id": protocol_id,
        "identifier": identifier,
        #"tlvs": tlvs,
    }
    #print("=================")
    #print(nlri)
    #print("=================")

    nlri["detail"] = decode_nlri_detail(nlri_type, protocol_id, tlvs)

    if ( tlvs != [] ):
      print("##### remain TLVs")
      print(tlvs)

    return nlri, off

#CONTAINER_TLVS = {256, 257, 258, 259}

def decode_tlvs(buf: bytes) -> List[Tuple[int, bytes]]:
    off = 0
    end = len(buf)
    tlvs = []
    res = []

    while off < end:
        if off + 4 > end:
          raise ValueError("TLV header truncated")

        t, l = struct.unpack("!HH", buf[off:off+4])
        off += 4

        if off + l > end:
          raise ValueError("TLV length overflow")

        value = buf[off:off+l]
        off += l

        tlv = {"type": t}

        # --- Descriptor TLVs ---
        if t in (256, 257):
          tlv["value"] = value
          tlv["subtlvs"] = decode_subtlvs(value)

        # --- value-only TLVs ---
        else:
          tlv["value"] = value

        tlvs.append(tlv)


    return tlvs, off

def decode_subtlvs(buf: bytes):
    off = 0
    end = len(buf)
    subs = []

    while off + 4 <= end:
        t, l = struct.unpack("!HH", buf[off:off+4])
        off += 4

        if off + l > end:
            raise ValueError("subTLV overflow")

        v = buf[off:off+l]
        off += l

        subs.append({"type": t, "value": v})

    return subs


def decode_nlri_detail(nlri_type: int, protocol_id: int, tlvs):
    if protocol_id == PROTO_OSPF:
      return decode_ospf_nlri(nlri_type, tlvs)
    elif protocol_id == PROTO_ISIS:
      return decode_isis_nlri(nlri_type, tlvs)
    else:
      return {}

def decode_ospf_nlri(nlri_type: int, tlvs):
    #print("ospf decode")
    info = {"protocol": "OSPF"}

    #print("-----")
    #print(tlvs)
    #print("-----")
    i = 0
    while i < len(tlvs):
    #for tlv in tlvs:
       tlv = tlvs[i]
       t = tlv.get("type")
       v = tlv.get("value")
       s = tlv.get("subtlvs")

    #for t, v in tlvs:
       if t == 263:
         info["mt_id"] = struct.unpack("!H", v)[0]
         tlvs.pop(i)
       elif t == 256:  # Local Node Descriptors
         info["local_node"]  = decode_ospf_node_descriptor(s)
         tlvs.pop(i)
       elif t == 257:
         info["remote_node"] = decode_ospf_node_descriptor(s)
         tlvs.pop(i)
       elif t == 258:
         print("***********************:LINK ID")
         info["link_identifier"] = v
         tlvs.pop(i)
       elif t == 259:
         #info["ipv4_interface_address"] = v
         info["ipv4_interface_address"] = socket.inet_ntoa(v)
         tlvs.pop(i)
       elif t == 260:
         #info["ipv4_neighbor_address"] = v
         info["ipv4_neighbor_address"] = socket.inet_ntoa(v)
         tlvs.pop(i)
       elif t == 265:
         info["ip_reachability_info"] = {}
         pl = struct.unpack("!B", v[0:1])[0]
         pr = v[1:]

         info["ip_reachability_info"]["prefix_len"] = pl
         pra = 4 - ( pl / 8 )
         if pra == 3:
             pr = pr + b'\x00\x00\x00'
         elif pra == 2:
             pr = pr + b'\x00\x00'
         elif pra == 1:
             pr = pr + b'\x00'
         #print(pr)
         info["ip_reachability_info"]["prefix"] = socket.inet_ntoa(pr)
         tlvs.pop(i)
       else:
         i += 1

    return info

def decode_ospf_link_descriptor(buf: bytes):
    res = {}
    for t, v in decode_tlvs(buf):
        if t == 514:  # Link Local ID
            res["local_link_id"] = struct.unpack("!I", v)[0]
        elif t == 515:  # Link Remote ID
            res["remote_link_id"] = struct.unpack("!I", v)[0]
        elif t == 516:  # IPv4 Interface Addr
            res["ipv4_if"] = ".".join(map(str, v))
        elif t == 517:  # IPv4 Neighbor Addr
            res["ipv4_nbr"] = ".".join(map(str, v))
        else:
            res[f"unknown_{t}"] = v
    return res

import socket
def decode_ospf_node_descriptor(tlvs):
    res = {}
    #print("ospf node")
    #print(tlvs)

    for st in tlvs:
        t = st["type"]
        v = st["value"]

    #for t, v in decode_tlvs(buf):
        #print(t,v)
        if t == 512:  # Router-ID
            #res["router_id"] = ".".join(map(str, v))
            #res["router_id"] = socket.inet_ntoa(v)
            res["asn"] = struct.unpack("!I", v)[0]
        elif t == 513:  # Interface Address
            res["interface_addr"] = ".".join(map(str, v))
        elif t == 514:  # ASN
            res["area_id"] = socket.inet_ntoa(v)
            #res["asn"] = struct.unpack("!I", v)[0]
        elif t == 515:  # Router-ID
            #res["router_id"] = ".".join(map(str, v))
            res["ipv4_router_id"] = socket.inet_ntoa(v)
        else:
            res[f"unknown_{t}"] = v
    return res


def decode_isis_nlri(nlri_type: int, tlvs):
    return {
        "protocol": "ISIS",
        "raw_tlvs": tlvs,
    }

def decode_ospf_prefix_descriptor(buf: bytes):
    res = {}
    off = 0

    # Prefix Length
    plen = buf[off]
    off += 1

    # Prefix
    byte_len = (plen + 7) // 8
    prefix = buf[off:off+byte_len]
    off += byte_len

    addr = prefix.ljust(4, b"\x00")
    res["prefix"] = ".".join(map(str, addr))
    res["length"] = plen

    return res

#def decode_ospf_bgp_ls_tlvs(tlvs):
#    decoded = {}
#
#    for t, v in tlvs:
#        if t == 256:
#            decoded["local_node"] = decode_ospf_node_descriptor(v)
#            print(decoded["local_node"])
#        elif t == 257:
#            decoded["remote_node"] = decode_ospf_node_descriptor(v)
#        elif t == 258:
#            decoded["link"] = decode_ospf_link_descriptor(v)
#        elif t == 259:
#            decoded["ipv4_interface"] = ".".join(map(str, v))
#        elif t == 260:
#            decoded["ipv4_neighbor"] = ".".join(map(str, v))
#        elif t == 263:
#            decoded["prefix"] = decode_ospf_prefix_descriptor(v)
#        else:
#            decoded[f"unknown_{t}"] = v
#
#    return decoded






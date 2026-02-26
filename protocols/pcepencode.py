# protocols/pcepencode.py
import struct
import socket
import itertools

# PCEP Object-Class
OBJ_SRP       = 33
OBJ_LSP       = 32
OBJ_ERO       = 7
OBJ_BANDWIDTH = 5

# PCEP Message-Type
PCEP_PCINITIATE = 11
PCEP_PCUPDATE   = 10

_srp_id_gen = itertools.count(1)

def _pcep_obj_header(obj_class, obj_type, body_len):
    length = 4 + body_len
    return struct.pack("!BBH", obj_class, obj_type, length)

def build_srp_object(srpid):
    #srp_id = next(_srp_id_gen)
    flag   = 0
    body = struct.pack("!II", flag, srpid)
    return _pcep_obj_header(OBJ_SRP, 1, len(body)) + body

def build_lsp_object(name: str, delegated=True, create=True):
    flags = 0
    if delegated:
        flags |= (1 << 0)   # D
    if create:
        flags |= (1 << 1)   # C (Initiate / Create 用フラグ扱い)

    plsp_id = 0  # 新規作成時は 0
    status = 0

    fixed = struct.pack("!HH", plsp_id, flags)

    # Symbolic Path Name TLV (Type=17)
    name_b = name.encode()
    tlv = struct.pack("!HH", 17, len(name_b)) + name_b

    body = fixed + tlv
    return _pcep_obj_header(OBJ_LSP, 1, len(body)) + body

def build_ero_object_from_path(path_nodes):
    """
    path_nodes: ['100.127.0.101', '100.127.0.102', '100.127.0.103']
    """
    subs = b""
    for ip in path_nodes:
        print(ip)
        subs += struct.pack("!BB", 1, 8) + socket.inet_aton(ip[3]) + struct.pack("!BB", 32, 0)  # IPv4 address subobject

    return _pcep_obj_header(OBJ_ERO, 1, len(subs)) + subs

def build_bandwidth_object(bw_mbps: float):
    # RFC5440: IEEE float
    body = struct.pack("!f", float(bw_mbps))
    return _pcep_obj_header(OBJ_BANDWIDTH, 1, len(body)) + body

def build_pcinitiate_from_path(path_update: dict,srpid):
    """
    PATH UPDATE(dict) -> PCInitiate payload
    """
    name = path_update["name"]
    if name == '2': # p2mp skip
      return None
    detail = path_update["detail"]["detail"]

    # --- p2p path 抽出（1本だけ取る最小実装） ---
    src = list(detail.keys())[0]
    dst = list(detail[src].keys())[0]
    print(src)
    print(dst)
    print(detail[src][dst])
    #for d in detail[src][dst]:
    #  path_nodes.append(
    #print(detail[src][dst]["links"])
    path_nodes = detail[src][dst][0]["links"]
    print("HOGE")
    print(path_nodes)

    objs = b""
    objs += build_srp_object(srpid)
    objs += build_lsp_object(name=name, delegated=True, create=True)
    objs += build_ero_object_from_path(path_nodes)
    objs += build_bandwidth_object(path_update["detail"].get("bw", 0))

    return objs


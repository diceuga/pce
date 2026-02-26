# utils/logging.py
import logging
import os
import time
from logging.handlers import WatchedFileHandler

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE  = os.getenv("LOG_FILE", "/var/log/dag/app.log")

class NanoFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ns = time.time_ns()
        sec = ns // 1_000_000_000
        nsec = ns % 1_000_000_000
        usec = (ns % 1_000_000_000) // 1_000


        t = time.localtime(sec)
        if datefmt:
            base = time.strftime(datefmt, t)
        else:
            base = time.strftime("%Y-%m-%d %H:%M:%S", t)

        return f"{base}.{usec:06d}"

class PrefixFilter(logging.Filter):
    def __init__(self, prefix):
        self.prefix = prefix

    def filter(self, record):
        return record.getMessage().startswith(self.prefix)


class ExcludePrefixFilter(logging.Filter):
    def __init__(self, prefixes):
        self.prefixes = prefixes

    def filter(self, record):
        msg = record.getMessage()
        return not any(msg.startswith(p) for p in self.prefixes)

#def log_comp(level, msg):
#  logging.log(level, "[COMPUTE] " + msg)


def setup_logging(base_dir):
    log_dir  = os.path.join(base_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    log_path       = os.path.join(log_dir, "log.txt")
    complog_path   = os.path.join(log_dir, "comp.txt")
    bgplslog_path  = os.path.join(log_dir, "bgp.txt")
    graphlog_path  = os.path.join(log_dir, "graph.txt")
    pceplog_path   = os.path.join(log_dir, "pcep.txt")

    #level = getattr(logging, LOG_LEVEL, logging.INFO)
    level = getattr(logging, "INFO", logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        #fmt="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        fmt="%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter = NanoFormatter(
      "%(asctime)s %(levelname)-5s %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S"
    )

    # logrotate 対応（外部でローテートされる前提）
    logfile = WatchedFileHandler(log_path)
    logfile.setFormatter(formatter)
    logfile.setLevel(level)
    logfile.addFilter(ExcludePrefixFilter(["[BGPLS]", "[COMPUTE]", "[GRAPH]", "[PCEP]"]))

    complogfile = WatchedFileHandler(complog_path)
    complogfile.setFormatter(formatter)
    complogfile.setLevel(level)
    complogfile.addFilter(PrefixFilter("[COMPUTE]"))

    bgplslogfile = WatchedFileHandler(bgplslog_path)
    bgplslogfile.setFormatter(formatter)
    bgplslogfile.setLevel(level)
    bgplslogfile.addFilter(PrefixFilter("[BGPLS]"))

    graphlogfile = WatchedFileHandler(graphlog_path)
    graphlogfile.setFormatter(formatter)
    graphlogfile.setLevel(level)
    graphlogfile.addFilter(PrefixFilter("[GRAPH]"))

    pceplogfile = WatchedFileHandler(pceplog_path)
    pceplogfile.setFormatter(formatter)
    pceplogfile.setLevel(level)
    pceplogfile.addFilter(PrefixFilter("[PCEP]"))

    # コンソール出力（任意）
    #console_handler = logging.StreamHandler()
    #console_handler.setFormatter(formatter)
    #console_handler.setLevel(level)

    root.addHandler(logfile)
    root.addHandler(bgplslogfile)
    root.addHandler(complogfile)
    root.addHandler(graphlogfile)
    root.addHandler(pceplogfile)
    #root.addHandler(console_handler)


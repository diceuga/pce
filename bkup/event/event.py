# event/event.py
from dataclasses import dataclass
import time

@dataclass
class Event:
    type: str
    payload: dict
    ts: float = time.time()


# utils/diff.py
from enum import Enum, auto


class DiffType(Enum):
    ADD = auto()
    DEL = auto()
    MOD = auto()


def diff_dict(old: dict, new: dict):
    """
    dict -> dict の差分を返す
    key 単位で ADD / DEL / MOD を検出
    """
    diffs = []

    old_keys = set(old.keys())
    new_keys = set(new.keys())

    # ADD
    for k in new_keys - old_keys:
        diffs.append({
            "type": DiffType.ADD,
            "id": k,
            "new": new[k],
        })

    # DEL
    for k in old_keys - new_keys:
        diffs.append({
            "type": DiffType.DEL,
            "id": k,
            "old": old[k],
        })

    # MOD
    for k in old_keys & new_keys:
        if old[k] != new[k]:
            diffs.append({
                "type": DiffType.MOD,
                "id": k,
                "old": old[k],
                "new": new[k],
            })

    return diffs


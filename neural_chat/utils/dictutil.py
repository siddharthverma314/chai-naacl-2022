from typing import List, Union
from flatten_dict import flatten, unflatten
import torch


def collate(dicts: List[dict]) -> dict:
    keys = flatten(dicts[0]).keys()
    new_dict = {}
    for k in keys:
        new_dict[k] = torch.cat([flatten(d)[k] for d in dicts])
    return unflatten(new_dict)


def uncollate(tbl: dict) -> List[dict]:
    tbl = {k: list(v) for k, v in flatten(tbl).items()}
    return [
        unflatten({k: v[i].unsqueeze(0) for k, v in tbl.items()})
        for i in range(len(next(iter(tbl.values()))))
    ]

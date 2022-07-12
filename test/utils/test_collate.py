from typing import List
from neural_chat.utils import collate, uncollate
import torch


def tolist(tbl: List[dict]):
    return [{k: list(v.squeeze()) for k, v in t.items()} for t in tbl]


def test_collate():
    BATCH_SIZE = 100
    for _ in range(100):
        tbl = [
            {k: torch.rand((1, 3)) for k in ("a", "b", "c")} for _ in range(BATCH_SIZE)
        ]
        assert tolist(uncollate(collate(tbl))) == tolist(tbl)


def test_collate_2():
    for _ in range(100):
        tbl = {k: torch.rand((100, 3)) for k in ("a", "b", "c")}
        after = collate(uncollate(tbl))
        for k in tbl.keys():
            assert torch.all(tbl[k] == after[k])

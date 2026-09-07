"""Token-position helpers shared by attention implementations."""

from __future__ import annotations

from typing import List

import torch


def _find_longest_run_positions(mask_1d: torch.Tensor) -> List[int]:
    """mask_1d: [S] bool; returns positions of the longest contiguous True run."""
    idx = mask_1d.nonzero(as_tuple=False).view(-1).tolist()
    if not idx:
        return []
    best_run: List[int] = []
    cur: List[int] = [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            if len(cur) > len(best_run):
                best_run = cur
            cur = [i]
    if len(cur) > len(best_run):
        best_run = cur
    return best_run

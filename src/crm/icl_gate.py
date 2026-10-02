"""ICLGate — hard safety gate (Stage 2)."""
from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Optional
from .pem_store import PEMStore

@dataclass
class ICLGateResult:
    tripped: bool
    similarity: float
    forced_action: Optional[int] = None

class ICLGate:
    def __init__(self, store: PEMStore, threshold: float = 0.94, verify_action: int = 0):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0,1]")
        self.store = store
        self.threshold = threshold
        self.verify_action = verify_action

    def check(self, query: torch.Tensor) -> ICLGateResult:
        sim = self.store.max_cosine(query)
        if sim != sim:  # NaN
            return ICLGateResult(tripped=True, similarity=sim, forced_action=self.verify_action)
        if sim >= self.threshold:
            return ICLGateResult(tripped=True, similarity=sim, forced_action=self.verify_action)
        return ICLGateResult(tripped=False, similarity=sim)

    def force_verify_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # set all but verify_action to -inf
        out = logits.clone()
        mask = torch.ones_like(out, dtype=torch.bool)
        mask[..., self.verify_action] = False
        out = out.masked_fill(mask, float("-inf"))
        return out

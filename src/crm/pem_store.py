"""PEMStore — failure-embedding store (Stage 2)."""
from __future__ import annotations
import torch
from typing import Optional

class PEMStore:
    def __init__(self, dim: int = 768):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._embeddings: list[torch.Tensor] = []

    def store(self, emb: torch.Tensor) -> None:
        if not torch.is_tensor(emb):
            raise TypeError("emb must be a torch.Tensor")
        if emb.dim() != 1 or emb.shape[0] != self.dim:
            raise ValueError(f"emb must be 1D of size {self.dim}")
        if not torch.isfinite(emb).all():
            raise ValueError("emb must be finite")
        norm = emb.norm()
        if norm == 0:
            raise ValueError("zero-norm embedding rejected")
        self._embeddings.append(emb / norm)

    def max_cosine(self, query: torch.Tensor) -> float:
        if not torch.is_tensor(query):
            raise TypeError("query must be a torch.Tensor")
        if query.dim() != 1 or query.shape[0] != self.dim:
            raise ValueError(f"query must be 1D of size {self.dim}")
        if not torch.isfinite(query).all():
            raise ValueError("query must be finite")
        qnorm = query.norm()
        if qnorm == 0 or len(self._embeddings) == 0:
            return float("nan")
        q = query / qnorm
        sims = [torch.dot(q, e).item() for e in self._embeddings]
        return max(sims)

    def __len__(self) -> int:
        return len(self._embeddings)

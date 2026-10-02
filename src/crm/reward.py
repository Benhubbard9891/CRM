"""Parametric multi-objective reward (CRM v4.1, Stage 4).

Blueprint formula (per step, per sequence)::

    r_t = w_q*DQ_t - w_c*(Cost(a_t)/Bn_t) - w_f*I[ECU fail]
          - w_p*I[PEM trip] - w_e*I[escalated]

Blueprint-explicit weights: w_q=10.0, w_p=4.0, w_e=8.0. The blueprint gives no
values for w_c and w_f, so they are required constructor arguments with no
invented defaults — the caller must choose them explicitly. See
``spec/parametric_reward_v4_1.yaml``.

The ``I[PEM trip]`` term integrates directly with Stage 2: feed
``ICLGateResult.tripped`` as ``pem_tripped``. Quality delta, action cost, ECU
failure, and escalation are caller-provided — their estimators are deferred.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Blueprint-explicit defaults.
W_Q_DEFAULT = 10.0
W_P_DEFAULT = 4.0
W_E_DEFAULT = 8.0

BUDGET_FLOOR = 1e-4  # blueprint-explicit: Bn_t clamped >= 1e-4


@dataclass(frozen=True)
class RewardWeights:
    """The five reward weights.

    w_q, w_p, w_e default to the blueprint values. w_c and w_f have no
    blueprint values and must be supplied explicitly.
    """

    w_c: float
    w_f: float
    w_q: float = W_Q_DEFAULT
    w_p: float = W_P_DEFAULT
    w_e: float = W_E_DEFAULT

    def __post_init__(self) -> None:
        for name in ("w_q", "w_c", "w_f", "w_p", "w_e"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not torch.isfinite(torch.tensor(float(v))):
                raise ValueError(f"{name} must be finite, got {v!r}")
            if v < 0:
                raise ValueError(
                    f"{name} must be non-negative; a negative weight would "
                    f"invert its incentive/disincentive, got {v}"
                )


class ParametricReward:
    """Computes r_t per sequence from caller-provided step signals."""

    def __init__(self, weights: RewardWeights, budget_floor: float = BUDGET_FLOOR) -> None:
        if not isinstance(weights, RewardWeights):
            raise ValueError(f"weights must be RewardWeights, got {type(weights).__name__}")
        if not budget_floor > 0:
            raise ValueError(f"budget_floor must be positive, got {budget_floor}")
        self.weights = weights
        self.budget_floor = float(budget_floor)

    @staticmethod
    def _float_vector(name: str, value: Tensor, batch: int) -> Tensor:
        v = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if v.numel() != batch:
            raise ValueError(f"{name} must hold B={batch} values, got {v.numel()}")
        if not torch.isfinite(v).all():
            raise ValueError(f"{name} must be finite")
        return v

    @staticmethod
    def _bool_vector(name: str, value: Tensor, batch: int) -> Tensor:
        v = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
        if v.numel() != batch:
            raise ValueError(f"{name} must hold B={batch} flags, got {v.numel()}")
        return v

    def __call__(
        self,
        *,
        quality_delta: Tensor,
        action_cost: Tensor,
        budget: Tensor,
        ecu_failed: Tensor,
        pem_tripped: Tensor,
        escalated: Tensor,
    ) -> Tensor:
        """Compute r_t for a batch.

        Args:
            quality_delta: [B] DQ_t quality delta per sequence.
            action_cost:   [B] Cost(a_t) per sequence.
            budget:        [B] Bn_t remaining budget; clamped to >= floor.
            ecu_failed:    [B] bool I[ECU fail].
            pem_tripped:   [B] bool I[PEM trip] (use ICLGateResult.tripped).
            escalated:     [B] bool I[escalated].

        Returns:
            [B] float32 rewards.
        """
        dq = torch.as_tensor(quality_delta, dtype=torch.float32).reshape(-1)
        if not torch.isfinite(dq).all():
            raise ValueError("quality_delta must be finite")
        batch = dq.numel()
        cost = self._float_vector("action_cost", action_cost, batch)
        bn = self._float_vector("budget", budget, batch).clamp_min(self.budget_floor)
        ecu = self._bool_vector("ecu_failed", ecu_failed, batch)
        pem = self._bool_vector("pem_tripped", pem_tripped, batch)
        esc = self._bool_vector("escalated", escalated, batch)

        w = self.weights
        return (
            w.w_q * dq
            - w.w_c * (cost / bn)
            - w.w_f * ecu.to(torch.float32)
            - w.w_p * pem.to(torch.float32)
            - w.w_e * esc.to(torch.float32)
        )

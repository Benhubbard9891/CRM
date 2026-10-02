"""CRM v4.1 — Cognitive Resource Manager package."""

from .pem_store import PEMStore
from .icl_gate import ICLGate, ICLGateResult
from .state_vector import StateVectorBuilder
from .reward import RewardWeights, ParametricReward
from .ppo import ppo_update, compute_gae
from .telemetry import PassRecord, SafetyTelemetry
from .reference import (
    ReferencePolicy,
    ReferenceEmbedder,
    ReferenceCostModel,
    ReferenceQualityEstimator,
    ReferenceFailureDetector,
)
from .rollout import collect_rollout, train_reference

__all__ = [
    "PEMStore",
    "ICLGate",
    "ICLGateResult",
    "StateVectorBuilder",
    "RewardWeights",
    "ParametricReward",
    "ppo_update",
    "compute_gae",
    "PassRecord",
    "SafetyTelemetry",
    "ReferencePolicy",
    "ReferenceEmbedder",
    "ReferenceCostModel",
    "ReferenceQualityEstimator",
    "ReferenceFailureDetector",
    "collect_rollout",
    "train_reference",
]

__version__ = "4.1.0"

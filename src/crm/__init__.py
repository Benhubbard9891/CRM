"""CRM v4.1 construction: PEM failure store, ICL hard gate, state vector.

Implements the staged build from the Master Architecture Blueprint.
Each stage is deterministic, fail-closed, and covered by verification tests.
"""

from .icl_gate import ICLGate, ICLGateResult
from .pem_store import PEMStore
from .ppo import (
    PPOConfig,
    PPOUpdateStats,
    RolloutBatch,
    clipped_surrogate_loss,
    compute_gae,
    exact_kl_divergence,
    pem_override_frequency,
    policy_entropy,
    ppo_total_loss,
    ppo_update,
    value_loss,
)
from .reward import ParametricReward, RewardWeights
from .reference import (
    REFERENCE_LR,
    REFERENCE_TASK_TAXONOMY,
    ActionVocabulary,
    PEMPpopulationPolicy,
    QualityDeltaEstimator,
    ReferenceCostModel,
    ReferenceFailureDetector,
    ReferencePolicy,
    ReferenceTextEmbedder,
    apply_budget_dynamics,
    export_telemetry_jsonl,
    mcdropout_uncertainty,
)
from .rollout import EnvObservation, RolloutEnv, ScriptedEnv, collect_rollout, train_reference
from .telemetry import PassRecord, SafetyTelemetry
from .state_vector import (
    BUDGET_CEILING,
    BUDGET_FLOOR,
    DIM_NAMES,
    STATE_DIM,
    UNCERTAINTY_DIM,
    StateVectorBuilder,
    shannon_entropy,
)

__all__ = [
    "PEMStore",
    "ICLGate",
    "ICLGateResult",
    "StateVectorBuilder",
    "shannon_entropy",
    "ParametricReward",
    "RewardWeights",
    "PPOConfig",
    "PPOUpdateStats",
    "RolloutBatch",
    "compute_gae",
    "clipped_surrogate_loss",
    "value_loss",
    "policy_entropy",
    "exact_kl_divergence",
    "ppo_total_loss",
    "ppo_update",
    "pem_override_frequency",
    "PassRecord",
    "SafetyTelemetry",
    "REFERENCE_LR",
    "REFERENCE_TASK_TAXONOMY",
    "ActionVocabulary",
    "PEMPpopulationPolicy",
    "QualityDeltaEstimator",
    "ReferenceCostModel",
    "ReferenceFailureDetector",
    "ReferencePolicy",
    "ReferenceTextEmbedder",
    "apply_budget_dynamics",
    "export_telemetry_jsonl",
    "mcdropout_uncertainty",
    "EnvObservation",
    "RolloutEnv",
    "ScriptedEnv",
    "collect_rollout",
    "train_reference",
    "STATE_DIM",
    "UNCERTAINTY_DIM",
    "DIM_NAMES",
    "BUDGET_FLOOR",
    "BUDGET_CEILING",
]

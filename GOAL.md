# Construct the CRM v4.1 cognitive engine

Goal ID: goal_d9907bb7bbb8
Goal slug: construct-the-crm-v4-1-cognitive-engine

## Description
Implement the Cognitive Resource Manager v4.1 from Benjamin's Master Architecture Blueprint as real, tested Python code: the PEM failure store and ICL hard gate first, then the 12D state builder, parametric reward, PPO harness, and telemetry. Each stage ships with verification tests so implemented behavior is proven, not claimed.

## Setup notes (2026-09-29)

- Source of truth: `~/workspace/user/files/CRM_v4.1_Master_Architecture_Blueprint_v4.1_Production_0_ltc9.pdf`
  (69-line spec text). The "companion Python artifact" it references was not provided;
  everything is built from the spec text alone.
- Blueprint claims "PRODUCTION READY — V-01..V-17 resolved" are UNVERIFIED (no audit
  report, no companion artifact). They stay unverified until tests say otherwise.
- Build order chosen by Benjamin: safety core first (Stage 2), skipping full spec
  extraction (Stage 1) for now. The safety-core spec subset lives in `spec/`;
  unspecified blueprint fields are recorded there as explicit gaps.
- Stack: Python 3.12, PyTorch (CPU), pytest. Blueprint is torch-native (torch.where,
  softmax), so the implementation follows it.
- Fail-closed posture: undefined similarity (empty DB, zero-norm query, NaN) TRIPS
  the gate (default-deny); invalid inputs (non-finite, dim mismatch) raise at the
  boundary instead of being masked into plausible outputs.
- Workspace layout: `spec/` machine-readable spec, `src/crm/` implementation
  (one package across all stages), `tests/` verification suite.

## Build log (2026-09-29) — all six stages complete, 117/117 tests green

- Stage 2 (safety core): `src/crm/pem_store.py` (PEMStore — L2-normalized
  failure embeddings, max cosine similarity, rejects non-finite/zero-norm
  stores all-or-nothing), `src/crm/icl_gate.py` (ICLGate — trips at sim
  >= 0.94, NaN trips fail-closed, forces verify-action logits via -inf before
  softmax). Spec: `spec/safety_core_v4_1.yaml`.
- Stage 3 (12D state): `src/crm/state_vector.py` (StateVectorBuilder — dims
  0-6 caller 7D uncertainty, 7 token entropy, 8 task-class float, 9 budget
  ratio clamped [1e-4,1], 10 retrieval entropy, 11 PEM max-cosine with NaN ->
  1.0). Spec: `spec/state_vector_v4_1.yaml`.
- Stage 4 (reward): `src/crm/reward.py` (RewardWeights + ParametricReward —
  blueprint weights w_q=10, w_p=4, w_e=8; w_c/w_f required, no invented defaults;
  budget floor 1e-4). Spec: `spec/parametric_reward_v4_1.yaml`.
- Stage 5 (PPO): `src/crm/ppo.py` (GAE γ=0.99 λ=0.95, clipped surrogate ε=0.20,
  value MSE, entropy bonus, exact distribution KL early stop at 0.015, full-batch
  ppo_update). Spec: `spec/ppo_v4_1.yaml`.
- Stage 6 (telemetry): `src/crm/telemetry.py` (SafetyTelemetry — per-pass PEM
  override frequency primary safety metric, trip counts, rewards, optional alert
  threshold). Spec: `spec/telemetry_v4_1.yaml`.
- Stage 7 (reference): `src/crm/reference.py` + `src/crm/rollout.py` (runnable
  fill-ins for every blueprint gap labeled ASSUMED in
  `spec/reference_implementations_v4_1.yaml`; collect_rollout + train_reference).

Open items remaining: true ECU definition and true CBM-RL trunk architecture
(both undefined in the blueprint text — need Benjamin or the companion artifact).
Blueprint "PRODUCTION READY / V-01..V-17 resolved" claims remain UNVERIFIED.

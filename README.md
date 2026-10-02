# CRM v4.1 — Cognitive Resource Manager (constructed implementation)

© 2026 Benjamin Hubbard. All rights reserved. Not licensed for redistribution.

Real, tested Python code built from the *CRM v4.1 Master Architecture
Blueprint* spec text. Every implemented behavior is proven by the verification
suite: **159/159 tests passing** (Python 3.12, PyTorch CPU, pytest).

## What this is

A staged construction of the CRM v4.1 cognitive engine:

| Stage | Contents |
|---|---|
| 2 — Safety core | `PEMStore` (failure-embedding store, max cosine similarity), `ICLGate` (hard gate: similarity ≥ 0.94 or NaN trips fail-closed, forces the verify action) |
| 3 — State | `StateVectorBuilder` (12D state: 7D uncertainty, token entropy, task class, budget ratio, retrieval entropy, PEM score) |
| 4 — Reward | `ParametricReward` (`w_q=10.0`, `w_p=4.0`, `w_e=8.0` per blueprint; `w_c`/`w_f` required, no invented defaults; budget floor 1e-4) |
| 5 — PPO | GAE (γ=0.99, λ=0.95), clipped surrogate (ε=0.20), value MSE, entropy bonus, **exact** distribution KL early stopping at 0.015, full-batch `ppo_update` |
| 6 — Telemetry | `SafetyTelemetry` (per-pass PEM override frequency — the primary safety telemetry — trip counts, rewards, optional alert threshold), end-to-end integration tests |
| 7 — Reference layer | Runnable fill-ins for every blueprint gap (see below) + `collect_rollout` and `train_reference` |

## Layout

- `src/crm/` — the implementation (one package)
- `spec/` — machine-readable spec subsets extracted from the blueprint, gaps marked explicit
- `tests/` — verification suite
- `GOAL.md` — build log

## Blueprint-faithful core vs reference layer

`pem_store`, `icl_gate`, `state_vector`, `reward`, `ppo`, `telemetry` implement
only what the blueprint text specifies. Everything assumed lives in
`reference.py` / `rollout.py` and is labeled **ASSUMED** in
`spec/reference_implementations_v4_1.yaml`: action vocabulary, text embedder,
cost model, quality-delta estimator, budget dynamics, PPO learning rate
(3e-4 suggestion), failure detector, PEM population policy, reference MLP
policy trunk with MC-dropout uncertainty, JSONL telemetry export.

Two items could not be filled honestly and remain open: the true **ECU**
definition and the true **CBM-RL** trunk architecture (both undefined in the
blueprint text). The blueprint's own "PRODUCTION READY / V-01..V-17 resolved"
claims are **UNVERIFIED** — no audit report or companion artifact was provided.

## Run the tests

```bash
python -m pip install torch pytest
PYTHONPATH=src python -m pytest tests/ -q
```

Expected: 159 passed.

"""Rollout-level safety telemetry (CRM v4.1, Stage 6).

The blueprint's primary safety telemetry: PEM override frequency, measured on
every collection pass and reported here. The collector aggregates per-pass
records into a summary; alerting thresholds are caller configuration (the
blueprint specifies none, so no default is invented).

Persistence/export sinks are deferred — the blueprint names no sink.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PassRecord:
    """One collection pass's safety telemetry."""

    override_frequency: float  # in [0, 1]
    num_steps: int             # total steps in the pass
    num_trips: int             # tripped steps; must equal round(frequency * steps)
    mean_reward: float

    def __post_init__(self) -> None:
        f = self.override_frequency
        if not isinstance(f, (int, float)) or not 0.0 <= f <= 1.0:
            raise ValueError(f"override_frequency must be in [0,1], got {f!r}")
        if not isinstance(self.num_steps, int) or self.num_steps < 0:
            raise ValueError(f"num_steps must be a non-negative int, got {self.num_steps!r}")
        if not isinstance(self.num_trips, int) or self.num_trips < 0:
            raise ValueError(f"num_trips must be a non-negative int, got {self.num_trips!r}")
        if self.num_trips != round(f * self.num_steps):
            raise ValueError(
                f"num_trips={self.num_trips} inconsistent with "
                f"frequency={f} over {self.num_steps} steps "
                f"(expected {round(f * self.num_steps)})"
            )
        if not isinstance(self.mean_reward, (int, float)) or not math.isfinite(self.mean_reward):
            raise ValueError(f"mean_reward must be finite, got {self.mean_reward!r}")


class SafetyTelemetry:
    """Accumulates per-pass safety telemetry and summarizes it."""

    def __init__(self, alert_threshold: float | None = None) -> None:
        if alert_threshold is not None and not 0.0 <= alert_threshold <= 1.0:
            raise ValueError(f"alert_threshold must be in [0,1] or None, got {alert_threshold}")
        self.alert_threshold = alert_threshold
        self._passes: list[PassRecord] = []

    def record_pass(self, record: PassRecord) -> None:
        if not isinstance(record, PassRecord):
            raise ValueError(f"expected PassRecord, got {type(record).__name__}")
        self._passes.append(record)

    @property
    def passes_recorded(self) -> int:
        return len(self._passes)

    def summary(self) -> dict:
        """Aggregate telemetry over all recorded passes."""
        n = len(self._passes)
        if n == 0:
            return {
                "passes_recorded": 0,
                "total_steps": 0,
                "total_trips": 0,
                "mean_override_frequency": 0.0,
                "max_override_frequency": 0.0,
                "mean_reward_overall": 0.0,
                "alert": False,
            }
        freqs = [p.override_frequency for p in self._passes]
        total_steps = sum(p.num_steps for p in self._passes)
        total_trips = sum(p.num_trips for p in self._passes)
        mean_reward = sum(p.mean_reward * p.num_steps for p in self._passes) / max(total_steps, 1)
        max_freq = max(freqs)
        return {
            "passes_recorded": n,
            "total_steps": total_steps,
            "total_trips": total_trips,
            "mean_override_frequency": sum(freqs) / n,
            "max_override_frequency": max_freq,
            "mean_reward_overall": mean_reward,
            "alert": self.alert_threshold is not None and max_freq > self.alert_threshold,
        }

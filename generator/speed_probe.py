#!/usr/bin/env python3
"""Lightweight timing recorder for step-100 generation.

The generator already contains instrumentation hooks; this module turns those
hooks into durable CSV/JSON/Markdown artifacts when speed auditing is enabled.
It is intentionally dependency-free so lab runs can collect timing data even in
minimal Python environments.
"""

from __future__ import annotations

import csv
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


ENABLE_ENV = "DATA_SYS_SPEED_AUDIT"
DIR_ENV = "DATA_SYS_SPEED_AUDIT_DIR"
EXPERIMENT_ENV = "DATA_SYS_SPEED_AUDIT_EXPERIMENT"
FLUSH_SECONDS_ENV = "DATA_SYS_SPEED_AUDIT_FLUSH_SECONDS"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def speed_audit_enabled() -> bool:
    return _truthy(os.environ.get(ENABLE_ENV))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_output_dir(base_dir: str | os.PathLike[str] | None) -> Path:
    root = Path(base_dir or ".").resolve() / "reports" / "speed_audit"
    return root / datetime.now().strftime("%Y%m%d_%H%M%S")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass
class ComponentStats:
    component: str
    calls: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0
    min_seconds: float = field(default_factory=lambda: float("inf"))

    def add(self, elapsed_seconds: float) -> None:
        self.calls += 1
        self.total_seconds += elapsed_seconds
        self.max_seconds = max(self.max_seconds, elapsed_seconds)
        self.min_seconds = min(self.min_seconds, elapsed_seconds)

    def as_row(self) -> Dict[str, Any]:
        average = self.total_seconds / self.calls if self.calls else 0.0
        min_seconds = 0.0 if self.min_seconds == float("inf") else self.min_seconds
        return {
            "component": self.component,
            "calls": self.calls,
            "total_seconds": round(self.total_seconds, 6),
            "avg_seconds": round(average, 6),
            "max_seconds": round(self.max_seconds, 6),
            "min_seconds": round(min_seconds, 6),
        }


@dataclass
class TimingToken:
    recorder: "SpeedAuditRecorder"
    component: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_perf: float = field(default_factory=time.perf_counter)
    start_wall: str = field(default_factory=_utc_now)

    def finish(self, *, extra_metadata: Optional[Dict[str, Any]] = None) -> float:
        elapsed = time.perf_counter() - self.start_perf
        metadata = dict(self.metadata)
        # Several generator hooks update `_sp.units` after the timed work knows
        # how many rows/items were produced. Preserve those late annotations.
        for attr in ("category", "units", "note"):
            if attr in self.__dict__:
                metadata[attr] = _jsonable(self.__dict__[attr])
        if extra_metadata:
            metadata.update(extra_metadata)
        self.recorder.record_duration(
            self.component,
            elapsed,
            metadata=metadata,
            started_at=self.start_wall,
        )
        return elapsed


class SpeedAuditRecorder:
    """Collects component timings and persists an audit report."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        experiment: str = "step100",
        flush_seconds: float = 60.0,
        slow_sample_limit: int = 500,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.flush_seconds = max(1.0, float(flush_seconds))
        self.slow_sample_limit = int(slow_sample_limit)
        self.created_at = _utc_now()
        self._started_perf = time.perf_counter()
        self._last_flush_perf = self._started_perf
        self._stats: Dict[str, ComponentStats] = {}
        self._samples: list[Dict[str, Any]] = []
        self._final_metadata: Dict[str, Any] = {}
        self._finalized = False

    def start(self, component: str, **metadata: Any) -> TimingToken:
        return TimingToken(self, component=str(component), metadata=_jsonable(metadata))

    @contextmanager
    def track(self, component: str, **metadata: Any) -> Iterator[TimingToken]:
        token = self.start(component, **metadata)
        try:
            yield token
        finally:
            token.finish()

    def record_duration(
        self,
        component: str,
        elapsed_seconds: float,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        started_at: Optional[str] = None,
    ) -> None:
        component = str(component)
        elapsed_seconds = max(0.0, float(elapsed_seconds))
        stats = self._stats.setdefault(component, ComponentStats(component))
        stats.add(elapsed_seconds)
        self._record_sample(component, elapsed_seconds, metadata or {}, started_at)
        if time.perf_counter() - self._last_flush_perf >= self.flush_seconds:
            self.flush()

    def _record_sample(
        self,
        component: str,
        elapsed_seconds: float,
        metadata: Dict[str, Any],
        started_at: Optional[str],
    ) -> None:
        row = {
            "component": component,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "started_at": started_at or "",
            "finished_at": _utc_now(),
            "metadata_json": json.dumps(_jsonable(metadata), sort_keys=True),
        }
        self._samples.append(row)
        self._samples.sort(key=lambda item: float(item["elapsed_seconds"]), reverse=True)
        if len(self._samples) > self.slow_sample_limit:
            del self._samples[self.slow_sample_limit :]

    def flush(self) -> None:
        self._last_flush_perf = time.perf_counter()
        self._write_component_summary()
        self._write_slow_samples()
        self._write_summary()
        self._write_markdown()

    def finalize(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self._finalized:
            return
        if metadata:
            self._final_metadata.update(_jsonable(metadata))
        self._finalized = True
        self.flush()

    def _component_rows(self) -> list[Dict[str, Any]]:
        rows = [stats.as_row() for stats in self._stats.values()]
        rows.sort(key=lambda row: float(row["total_seconds"]), reverse=True)
        return rows

    def _write_csv(self, path: Path, rows: Iterable[Dict[str, Any]], fieldnames: list[str]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        tmp_path.replace(path)

    def _write_component_summary(self) -> None:
        self._write_csv(
            self.output_dir / "component_summary.csv",
            self._component_rows(),
            ["component", "calls", "total_seconds", "avg_seconds", "max_seconds", "min_seconds"],
        )

    def _write_slow_samples(self) -> None:
        self._write_csv(
            self.output_dir / "slow_samples.csv",
            self._samples,
            ["component", "elapsed_seconds", "started_at", "finished_at", "metadata_json"],
        )

    def _summary_payload(self) -> Dict[str, Any]:
        rows = self._component_rows()
        accounted = sum(float(row["total_seconds"]) for row in rows)
        wall = time.perf_counter() - self._started_perf
        return {
            "experiment": self.experiment,
            "created_at": self.created_at,
            "updated_at": _utc_now(),
            "output_dir": str(self.output_dir),
            "wall_seconds": round(wall, 6),
            "accounted_component_seconds": round(accounted, 6),
            "component_count": len(rows),
            "sample_count": len(self._samples),
            "final_metadata": self._final_metadata,
            "top_components": rows[:20],
            "slowest_samples": self._samples[:50],
        }

    def _write_summary(self) -> None:
        path = self.output_dir / "summary.json"
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(self._summary_payload(), indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def _write_markdown(self) -> None:
        payload = self._summary_payload()
        lines = [
            "# Step 100 Speed Audit",
            "",
            f"- Experiment: `{payload['experiment']}`",
            f"- Updated: `{payload['updated_at']}`",
            f"- Wall seconds: `{payload['wall_seconds']}`",
            f"- Accounted component seconds: `{payload['accounted_component_seconds']}`",
            "",
            "## Top Components",
            "",
            "| Component | Calls | Total s | Avg s | Max s |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in payload["top_components"][:15]:
            lines.append(
                f"| `{row['component']}` | {row['calls']} | {row['total_seconds']} | "
                f"{row['avg_seconds']} | {row['max_seconds']} |"
            )
        lines.extend(["", "## Slowest Samples", "", "| Component | Seconds | Metadata |", "|---|---:|---|"])
        for row in payload["slowest_samples"][:15]:
            metadata = row.get("metadata_json", "{}")
            lines.append(f"| `{row['component']}` | {row['elapsed_seconds']} | `{metadata}` |")
        (self.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


_RECORDER_CACHE: Dict[tuple[str, str], SpeedAuditRecorder] = {}


def get_env_speed_recorder(
    *,
    default_experiment: str = "step100",
    base_dir: str | os.PathLike[str] | None = None,
) -> SpeedAuditRecorder | None:
    if not speed_audit_enabled():
        return None
    experiment = os.environ.get(EXPERIMENT_ENV, default_experiment)
    output_dir = os.environ.get(DIR_ENV)
    path = Path(output_dir).resolve() if output_dir else _default_output_dir(base_dir)
    cache_key = (experiment, str(path))
    recorder = _RECORDER_CACHE.get(cache_key)
    if recorder is None:
        flush_seconds = float(os.environ.get(FLUSH_SECONDS_ENV, "60"))
        recorder = SpeedAuditRecorder(path, experiment=experiment, flush_seconds=flush_seconds)
        _RECORDER_CACHE[cache_key] = recorder
    return recorder

#!/usr/bin/env python3
"""Optional memory audit recorder for long generation runs.

The memory probe is deliberately opt-in. It records coarse snapshots around
major phases so we can identify accidental state growth without paying runtime
cost in normal benchmark-candidate runs.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:  # pragma: no cover - optional dependency path
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


ENABLE_ENV = "DATA_SYS_MEMORY_AUDIT"
DIR_ENV = "DATA_SYS_MEMORY_AUDIT_DIR"
EXPERIMENT_ENV = "DATA_SYS_MEMORY_AUDIT_EXPERIMENT"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def audit_enabled() -> bool:
    return _truthy(os.environ.get(ENABLE_ENV))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_output_dir() -> Path:
    return Path.cwd().resolve() / "reports" / "memory_audit" / datetime.now().strftime("%Y%m%d_%H%M%S")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass
class AuditTarget:
    name: str
    obj: Any
    category: str = ""
    note: str = ""
    overlap_group: str = ""


def audit_target(
    name: str,
    obj: Any,
    category: str = "",
    note: str = "",
    overlap_group: str = "",
) -> AuditTarget:
    return AuditTarget(str(name), obj, str(category), str(note), str(overlap_group))


def graph_audit_targets(graph: Any) -> list[AuditTarget]:
    targets = [audit_target("graph", graph, "graph_runtime", "graph runtime object", "graph")]
    for attr in (
        "edges",
        "_edges",
        "active_edges",
        "temporal_history",
        "edge_history",
        "nodes",
        "communities",
    ):
        if hasattr(graph, attr):
            targets.append(audit_target(f"graph.{attr}", getattr(graph, attr)))
    return targets


def _object_size_bytes(obj: Any) -> int:
    if obj is None:
        return 0
    try:
        if hasattr(obj, "memory_usage"):
            usage = obj.memory_usage(deep=True)
            if hasattr(usage, "sum"):
                return int(usage.sum())
            return int(usage)
    except Exception:
        pass
    try:
        if hasattr(obj, "nbytes"):
            return int(obj.nbytes)
    except Exception:
        pass
    try:
        if hasattr(obj, "get_total_buffer_size"):
            return int(obj.get_total_buffer_size())
    except Exception:
        pass
    try:
        return int(sys.getsizeof(obj))
    except Exception:
        return 0


def _object_count(obj: Any) -> int:
    if obj is None:
        return 0
    try:
        return int(len(obj))
    except Exception:
        return 1


class MemoryAuditRecorder:
    def __init__(self, output_dir: str | os.PathLike[str], *, experiment: str = "step100") -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.created_at = _utc_now()
        self.rows: list[Dict[str, Any]] = []
        self._process_rss_samples: list[int] = []

    def record_snapshot(
        self,
        phase: str,
        targets: Iterable[AuditTarget],
        *,
        note: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        sample_kind: str = "checkpoint",
    ) -> None:
        rss_bytes = 0
        if psutil is not None:
            try:
                rss_bytes = int(psutil.Process(os.getpid()).memory_info().rss)
            except Exception:
                rss_bytes = 0
        metadata_json = json.dumps(_jsonable(metadata or {}), sort_keys=True)
        for target in targets:
            row = {
                "timestamp": _utc_now(),
                "experiment": self.experiment,
                "phase": str(phase),
                "sample_kind": str(sample_kind),
                "target": target.name,
                "category": target.category,
                "overlap_group": target.overlap_group,
                "object_count": _object_count(target.obj),
                "estimated_bytes": _object_size_bytes(target.obj),
                "process_rss_bytes": rss_bytes,
                "plateau_hit": False,
                "note": note or target.note,
                "metadata_json": metadata_json,
            }
            self.rows.append(row)
        self.flush()

    def record_process(
        self,
        phase: str,
        *,
        sample_kind: str = "process_sample",
        note: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rss_bytes = 0
        if psutil is not None:
            try:
                rss_bytes = int(psutil.Process(os.getpid()).memory_info().rss)
            except Exception:
                rss_bytes = 0
        self._process_rss_samples.append(rss_bytes)
        recent = [value for value in self._process_rss_samples[-6:] if value > 0]
        plateau_hit = False
        if len(recent) >= 6:
            low = min(recent)
            high = max(recent)
            plateau_hit = (high - low) <= max(1, int(high * 0.05))
        row = {
            "timestamp": _utc_now(),
            "experiment": self.experiment,
            "phase": str(phase),
            "sample_kind": str(sample_kind),
            "target": "__process__",
            "category": "process",
            "overlap_group": "process",
            "object_count": 1,
            "estimated_bytes": rss_bytes,
            "process_rss_bytes": rss_bytes,
            "plateau_hit": plateau_hit,
            "note": note,
            "metadata_json": json.dumps(_jsonable(metadata or {}), sort_keys=True),
        }
        self.rows.append(row)
        self.flush()
        return row

    def flush(self) -> None:
        self._write_csv()
        self._write_summary()
        self._write_markdown()

    def _write_csv(self) -> None:
        fields = [
            "timestamp",
            "experiment",
            "phase",
            "sample_kind",
            "target",
            "category",
            "overlap_group",
            "object_count",
            "estimated_bytes",
            "process_rss_bytes",
            "plateau_hit",
            "note",
            "metadata_json",
        ]
        path = self.output_dir / "snapshots.csv"
        tmp_path = path.with_suffix(".csv.tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
        tmp_path.replace(path)

    def _summary_payload(self) -> Dict[str, Any]:
        latest_by_target: Dict[str, Dict[str, Any]] = {}
        for row in self.rows:
            latest_by_target[row["target"]] = row
        largest = sorted(
            latest_by_target.values(),
            key=lambda row: int(row.get("estimated_bytes") or 0),
            reverse=True,
        )
        return {
            "experiment": self.experiment,
            "created_at": self.created_at,
            "updated_at": _utc_now(),
            "output_dir": str(self.output_dir),
            "snapshot_rows": len(self.rows),
            "latest_target_count": len(latest_by_target),
            "largest_latest_targets": largest[:30],
        }

    def _write_summary(self) -> None:
        path = self.output_dir / "summary.json"
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(self._summary_payload(), indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def _write_markdown(self) -> None:
        payload = self._summary_payload()
        lines = [
            "# Step 100 Memory Audit",
            "",
            f"- Experiment: `{payload['experiment']}`",
            f"- Updated: `{payload['updated_at']}`",
            f"- Snapshot rows: `{payload['snapshot_rows']}`",
            "",
            "## Largest Latest Targets",
            "",
            "| Target | Count | Estimated MB | RSS MB | Phase |",
            "|---|---:|---:|---:|---|",
        ]
        for row in payload["largest_latest_targets"][:20]:
            estimated_mb = int(row.get("estimated_bytes") or 0) / (1024 * 1024)
            rss_mb = int(row.get("process_rss_bytes") or 0) / (1024 * 1024)
            lines.append(
                f"| `{row['target']}` | {row['object_count']} | {estimated_mb:.2f} | "
                f"{rss_mb:.2f} | `{row['phase']}` |"
            )
        (self.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


_RECORDER_CACHE: Dict[tuple[str, str], MemoryAuditRecorder] = {}


def get_env_audit_recorder(*, default_experiment: str = "step100") -> MemoryAuditRecorder | None:
    if not audit_enabled():
        return None
    experiment = os.environ.get(EXPERIMENT_ENV, default_experiment)
    output_dir = Path(os.environ.get(DIR_ENV) or _default_output_dir()).resolve()
    cache_key = (experiment, str(output_dir))
    recorder = _RECORDER_CACHE.get(cache_key)
    if recorder is None:
        recorder = MemoryAuditRecorder(output_dir, experiment=experiment)
        _RECORDER_CACHE[cache_key] = recorder
    return recorder

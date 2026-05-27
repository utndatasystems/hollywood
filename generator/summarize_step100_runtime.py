#!/usr/bin/env python3
"""Summarize step-100 runtime, profiling, resume, and quality guardrail status."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_parse_error": line[:200]})
    return rows


def _read_csv(path: Path) -> list[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _latest_matching_dir(root: Path, required_file: str) -> Optional[Path]:
    if (root / required_file).exists():
        return root
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir() and (path / required_file).exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_speed(base_dir: Path, speed_dir: Optional[Path]) -> Dict[str, Any]:
    root = speed_dir or base_dir / "reports" / "speed_audit"
    audit_dir = _latest_matching_dir(root, "component_summary.csv")
    if audit_dir is None:
        return {"exists": False, "root": str(root), "message": "No speed audit artifacts found."}
    component_rows = _read_csv(audit_dir / "component_summary.csv")
    slow_rows = _read_csv(audit_dir / "slow_samples.csv")
    summary = _read_json(audit_dir / "summary.json")
    component_rows.sort(key=lambda row: _float(row.get("total_seconds")), reverse=True)
    slow_rows.sort(key=lambda row: _float(row.get("elapsed_seconds")), reverse=True)
    return {
        "exists": True,
        "dir": str(audit_dir),
        "summary": summary,
        "top_components": component_rows[:20],
        "slow_samples": slow_rows[:20],
    }


def _event_timestamp(event: Dict[str, Any]) -> Optional[float]:
    for key in ("time", "timestamp", "ts", "wall_time", "created_at"):
        value = event.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _load_progress(base_dir: Path) -> Dict[str, Any]:
    progress_path = base_dir / "decision_logs" / "movie_generation_progress.jsonl"
    events = _read_jsonl(progress_path)
    timestamps = [stamp for event in events if (stamp := _event_timestamp(event)) is not None]
    movie_like = [
        event
        for event in events
        if str(event.get("event", "")).lower() in {"movie_complete", "movie_completed", "movie_done"}
        or "movie_id" in event
        and str(event.get("event", "")).lower().endswith("complete")
    ]
    completed_ids = {
        event.get("movie_id")
        for event in movie_like
        if event.get("movie_id") not in (None, "")
    }
    elapsed_seconds = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
    completed = len(completed_ids) or len(movie_like)
    return {
        "exists": progress_path.exists(),
        "path": str(progress_path),
        "event_count": len(events),
        "last_event": events[-1] if events else None,
        "completed_movie_events": completed,
        "elapsed_seconds_from_log": round(elapsed_seconds, 3),
        "movies_per_second_from_log": round(completed / elapsed_seconds, 6) if elapsed_seconds > 0 else 0.0,
    }


def _load_resume(base_dir: Path) -> Dict[str, Any]:
    manifest_path = base_dir / "_step100_resume" / "manifest.json"
    payload = _read_json(manifest_path)
    return {
        "exists": manifest_path.exists(),
        "path": str(manifest_path),
        "status": payload.get("status"),
        "last_completed_year": payload.get("last_completed_year"),
        "last_completed_sequence_index": payload.get("last_completed_sequence_index"),
        "movie_count": payload.get("movie_count") or payload.get("produced_movie_count"),
        "start_year": payload.get("start_year"),
        "end_year": payload.get("end_year"),
        "fingerprint": payload.get("run_compatibility_fingerprint") or payload.get("fingerprint"),
    }


def _find_sanity_report(base_dir: Path) -> Optional[Path]:
    candidates = [
        base_dir / "reports" / "step100_sanity_report.json",
        base_dir / "reports" / "step100_sanity.json",
        base_dir / "step100_sanity_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    report_root = base_dir / "reports"
    if report_root.exists():
        matches = sorted(report_root.glob("*sanity*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def _load_quality(base_dir: Path) -> Dict[str, Any]:
    report_path = _find_sanity_report(base_dir)
    if report_path is None:
        return {"exists": False, "message": "No sanity report JSON found yet."}
    payload = _read_json(report_path)
    gates = payload.get("gates") or payload.get("quality_gates") or {}
    movie = payload.get("movie") or {}
    cast = payload.get("cast") or {}
    directors = payload.get("directors") or {}
    companies = payload.get("companies") or {}
    keywords = payload.get("keywords") or {}
    awards = payload.get("awards") or {}
    selected_metrics = {
        "movie_rows": movie.get("rows"),
        "unique_titles": movie.get("unique_titles"),
        "duplicate_titles": movie.get("duplicate_titles"),
        "duplicated_tagline_rate_pct": movie.get("duplicated_tagline_rate_pct"),
        "cast_reuse_ratio": cast.get("reuse_ratio"),
        "unique_cast_people": cast.get("unique_people"),
        "blank_character_description_rate_pct": cast.get("blank_character_description_rate_pct"),
        "unique_directors": directors.get("unique_directors"),
        "company_gini": companies.get("gini"),
        "unique_companies": companies.get("unique_companies"),
        "keyword_zero_exact_topic_rate_pct": keywords.get("zero_exact_topic_rate_pct"),
        "awards_movie_share_pct": awards.get("movies_with_awards_rate_pct"),
        "awards_win_share_pct": awards.get("win_rate_pct"),
    }
    return {
        "exists": True,
        "path": str(report_path),
        "gates": gates,
        "selected_metrics": {key: value for key, value in selected_metrics.items() if value is not None},
    }


def _load_runtime_config(base_dir: Path) -> Dict[str, Any]:
    paths: list[Path] = []
    env_path = os.environ.get("DATA_SYS_PIPELINE_CONFIG")
    if env_path:
        paths.append(Path(env_path))
    paths.extend([base_dir / "benchmark_candidate_profile.json", base_dir / "pipeline_profile.json"])
    for path in paths:
        if path.exists():
            payload = _read_json(path)
            runtime = payload.get("runtime", {})
            return {
                "exists": True,
                "path": str(path),
                "disabled_secondary_tables": runtime.get("disabled_secondary_tables", []),
                "disabled_global_tables": runtime.get("disabled_global_tables", []),
                "disabled_post_loop_tables": runtime.get("disabled_post_loop_tables", []),
            }
    return {"exists": False}


def build_summary(base_dir: Path, speed_dir: Optional[Path] = None) -> Dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "base_dir": str(base_dir),
        "speed_audit": _load_speed(base_dir, speed_dir),
        "progress": _load_progress(base_dir),
        "resume": _load_resume(base_dir),
        "runtime_config": _load_runtime_config(base_dir),
        "quality": _load_quality(base_dir),
    }


def _format_table(rows: Iterable[Dict[str, Any]], columns: list[str], limit: int) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in list(rows)[:limit]:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def write_outputs(summary: Dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "step100_runtime_summary.json"
    md_path = out_dir / "step100_runtime_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    speed = summary["speed_audit"]
    progress = summary["progress"]
    resume = summary["resume"]
    runtime = summary["runtime_config"]
    quality = summary["quality"]
    lines = [
        "# Step 100 Runtime Summary",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Base dir: `{summary['base_dir']}`",
        f"- Resume status: `{resume.get('status')}`; last completed year: `{resume.get('last_completed_year')}`",
        f"- Progress events: `{progress.get('event_count')}`; movie-complete events: `{progress.get('completed_movie_events')}`",
        f"- Movies/sec from log: `{progress.get('movies_per_second_from_log')}`",
        "",
        "## Runtime Profile",
        "",
        f"- Config found: `{runtime.get('exists')}`",
        f"- Disabled secondary tables: `{runtime.get('disabled_secondary_tables', [])}`",
        f"- Disabled global tables: `{runtime.get('disabled_global_tables', [])}`",
        f"- Disabled post-loop tables: `{runtime.get('disabled_post_loop_tables', [])}`",
        "",
        "## Speed Audit",
        "",
    ]
    if speed.get("exists"):
        lines.append(f"- Audit dir: `{speed.get('dir')}`")
        speed_summary = speed.get("summary") or {}
        lines.append(f"- Wall seconds: `{speed_summary.get('wall_seconds')}`")
        lines.append(f"- Accounted component seconds: `{speed_summary.get('accounted_component_seconds')}`")
        lines.extend(["", "### Top Components", ""])
        lines.extend(_format_table(speed.get("top_components", []), ["component", "calls", "total_seconds", "avg_seconds", "max_seconds"], 15))
        lines.extend(["", "### Slow Samples", ""])
        lines.extend(_format_table(speed.get("slow_samples", []), ["component", "elapsed_seconds", "metadata_json"], 10))
    else:
        lines.append(f"- `{speed.get('message')}`")
    lines.extend(["", "## Quality Guardrails", ""])
    if quality.get("exists"):
        lines.append(f"- Sanity report: `{quality.get('path')}`")
        gates = quality.get("gates") or {}
        if gates:
            failed = [name for name, value in gates.items() if value is False]
            lines.append(f"- Failed gates: `{failed}`")
        metrics = quality.get("selected_metrics") or {}
        for key, value in metrics.items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append(f"- `{quality.get('message')}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, help="Run directory to summarize.")
    parser.add_argument("--out-dir", default=None, help="Directory for step100_runtime_summary outputs.")
    parser.add_argument("--speed-dir", default=None, help="Explicit speed-audit directory or root.")
    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else base_dir / "reports"
    speed_dir = Path(args.speed_dir).resolve() if args.speed_dir else None
    summary = build_summary(base_dir, speed_dir=speed_dir)
    json_path, md_path = write_outputs(summary, out_dir)
    print(f"Runtime summary written: {json_path}")
    print(f"Runtime summary written: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

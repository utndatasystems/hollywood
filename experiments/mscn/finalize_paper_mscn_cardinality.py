#!/usr/bin/env python3
"""Finalize a paper MSCN cardinality run.

This script is intentionally lightweight and dependency-free. It reads the
3-seed MSCN report/prediction layout produced by the public MSCN runner
and writes a paper-collection bundle with:

* an artifact audit,
* seed-level and aggregate Q-error tables,
* a recommended reporting policy,
* handlebar and signed-bias SVG plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


WORKLOADS = [
    ("job_light", "JOB-Light"),
    ("job_exact", "JOB"),
    ("job_complex", "JOB-Complex"),
]


@dataclass(frozen=True)
class PredictionRow:
    workload: str
    workload_label: str
    seed: str
    query_id: str
    prediction: float
    actual: float
    q_error: float


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return None
    if text.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if text.lower() in {"-inf", "-infinity"}:
        return -math.inf
    try:
        return float(text)
    except ValueError:
        return None


def q_error(prediction: float, actual: float) -> float:
    if actual <= 0 or prediction <= 0:
        return math.inf
    return max(prediction / actual, actual / prediction)


def percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct / 100.0
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[lower]
    frac = pos - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def summarize(values: list[float]) -> dict[str, float | int | None]:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p90": percentile(finite, 90),
        "p95": percentile(finite, 95),
        "p99": percentile(finite, 99),
        "max": max(finite),
        "min": min(finite),
        "p25": percentile(finite, 25),
        "p75": percentile(finite, 75),
        "variance": statistics.pvariance(finite) if len(finite) > 1 else 0.0,
    }


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "inf" if number > 0 else "-inf"
    if abs(number) >= 1000 or (0 < abs(number) < 0.001):
        return f"{number:.6g}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_prediction_csv(path: Path, workload: str, label: str, seed: str) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            actual = parse_float(record.get("actual"))
            prediction = parse_float(record.get("prediction_for_q_error"))
            if prediction is None:
                prediction = parse_float(record.get("prediction"))
            qe = parse_float(record.get("q_error"))
            if actual is None or prediction is None or actual <= 0:
                continue
            if qe is None or qe < 1:
                qe = q_error(prediction, actual)
            if not math.isfinite(qe):
                continue
            rows.append(
                PredictionRow(
                    workload=workload,
                    workload_label=label,
                    seed=seed,
                    query_id=str(record.get("query_id", "")),
                    prediction=float(prediction),
                    actual=float(actual),
                    q_error=float(qe),
                )
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fieldnames})


def svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def x_jitter(key: str, spread: float = 16.0) -> float:
    total = sum(ord(ch) * (idx + 1) for idx, ch in enumerate(key))
    return ((total % 1000) / 999.0 - 0.5) * 2.0 * spread


def render_handlebar_svg(path: Path, predictions: list[PredictionRow], *, max_log: float | None = None) -> None:
    workloads = WORKLOADS
    seeds = sorted({row.seed for row in predictions}, key=lambda s: int(s))
    max_q = max((row.q_error for row in predictions if math.isfinite(row.q_error)), default=10.0)
    y_max_log = float(max_log) if max_log is not None else max(1.0, math.ceil(math.log10(max_q)))
    width = 1080
    panel_h = 340
    height = 84 + panel_h * len(workloads)
    left = 92
    right = 42
    plot_w = width - left - right
    top_pad = 54
    bottom_pad = 78
    colors = {"1": "#2563eb", "2": "#059669", "3": "#7c3aed"}

    def y_for(plot_top: float, plot_h: float, q: float) -> float:
        log_value = math.log10(max(1.0, min(q, 10.0**y_max_log)))
        return plot_top + (y_max_log - log_value) / y_max_log * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.small{font-size:12px;fill:#4b5563}.grid{stroke:#e5e7eb}.axis{stroke:#111827}</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{left}" y="32" font-size="22" font-weight="700">MSCN Cardinality Q-Error by Seed</text>',
        f'<text x="{left}" y="55" class="small">Whisker=min/max, box=IQR, line=median, diamond=mean. Y axis is log10(q-error), capped at 1e{y_max_log:g}.</text>',
    ]
    for panel_idx, (workload, label) in enumerate(workloads):
        panel_y = 76 + panel_idx * panel_h
        plot_top = panel_y + top_pad
        plot_h = panel_h - top_pad - bottom_pad
        parts.append(f'<text x="{left}" y="{panel_y + 24}" font-size="18" font-weight="700">{label}</text>')
        for tick in range(0, int(y_max_log) + 1):
            y = y_for(plot_top, plot_h, 10.0**tick)
            parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" class="grid"/>')
            parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" class="small">1e{tick}</text>')
        parts.append(f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{plot_top + plot_h}" class="axis"/>')
        parts.append(f'<line x1="{left}" y1="{plot_top + plot_h}" x2="{width - right}" y2="{plot_top + plot_h}" class="axis"/>')
        for seed_idx, seed in enumerate(seeds):
            group = [row for row in predictions if row.workload == workload and row.seed == seed]
            if not group:
                continue
            stats = summarize([row.q_error for row in group])
            x = left + (seed_idx + 0.5) * plot_w / len(seeds)
            color = colors.get(seed, "#374151")
            q_min = float(stats["min"])
            q25 = float(stats["p25"])
            q50 = float(stats["median"])
            q75 = float(stats["p75"])
            q_max = float(stats["max"])
            q_mean = float(stats["mean"])
            y_min = y_for(plot_top, plot_h, q_min)
            y25 = y_for(plot_top, plot_h, q25)
            y50 = y_for(plot_top, plot_h, q50)
            y75 = y_for(plot_top, plot_h, q75)
            y_max = y_for(plot_top, plot_h, q_max)
            y_mean = y_for(plot_top, plot_h, q_mean)
            parts.append(f'<line x1="{x:.2f}" y1="{y_max:.2f}" x2="{x:.2f}" y2="{y_min:.2f}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<line x1="{x - 22:.2f}" y1="{y_max:.2f}" x2="{x + 22:.2f}" y2="{y_max:.2f}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<line x1="{x - 22:.2f}" y1="{y_min:.2f}" x2="{x + 22:.2f}" y2="{y_min:.2f}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<rect x="{x - 32:.2f}" y="{min(y25, y75):.2f}" width="64" height="{max(abs(y75 - y25), 2):.2f}" fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<line x1="{x - 32:.2f}" y1="{y50:.2f}" x2="{x + 32:.2f}" y2="{y50:.2f}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<polygon points="{x:.2f},{y_mean - 7:.2f} {x + 7:.2f},{y_mean:.2f} {x:.2f},{y_mean + 7:.2f} {x - 7:.2f},{y_mean:.2f}" fill="#fff" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<text x="{x:.2f}" y="{plot_top + plot_h + 30:.2f}" text-anchor="middle" font-size="13" font-weight="700">seed {svg_escape(seed)}</text>')
            parts.append(f'<text x="{x:.2f}" y="{plot_top + plot_h + 47:.2f}" text-anchor="middle" class="small">n={len(group)}, med={fmt(q50)}</text>')
            parts.append(f'<text x="{x:.2f}" y="{plot_top + plot_h + 63:.2f}" text-anchor="middle" class="small">p95={fmt(stats["p95"])}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def render_signed_svg(path: Path, predictions: list[PredictionRow], *, cap_log10: float = 5.0) -> None:
    workloads = WORKLOADS
    seeds = sorted({row.seed for row in predictions}, key=lambda s: int(s))
    width = 1080
    panel_h = 360
    height = 92 + panel_h * len(workloads)
    left = 92
    right = 42
    plot_w = width - left - right
    top_pad = 60
    bottom_pad = 86
    colors = {"1": "#2563eb", "2": "#059669", "3": "#7c3aed"}

    def signed_value(row: PredictionRow) -> float:
        direction = 1.0 if row.prediction >= row.actual else -1.0
        return direction * math.log10(max(1.0, row.q_error))

    def y_for(plot_top: float, plot_h: float, value: float) -> float:
        clipped = max(-cap_log10, min(cap_log10, value))
        return plot_top + (cap_log10 - clipped) / (2.0 * cap_log10) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.small{font-size:12px;fill:#4b5563}.grid{stroke:#e5e7eb}.zero{stroke:#111827;stroke-width:1.5}.axis{stroke:#111827}</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
        f'<text x="{left}" y="32" font-size="22" font-weight="700">MSCN Signed Cardinality Q-Error Bias</text>',
        f'<text x="{left}" y="55" class="small">Above zero = overestimate, below zero = underestimate. Axis is signed log10(q-error), clipped at 1e{cap_log10:g}.</text>',
    ]
    for panel_idx, (workload, label) in enumerate(workloads):
        panel_y = 84 + panel_idx * panel_h
        plot_top = panel_y + top_pad
        plot_h = panel_h - top_pad - bottom_pad
        parts.append(f'<text x="{left}" y="{panel_y + 24}" font-size="18" font-weight="700">{label}</text>')
        for tick in range(-int(cap_log10), int(cap_log10) + 1):
            y = y_for(plot_top, plot_h, float(tick))
            cls = "zero" if tick == 0 else "grid"
            label_tick = "0" if tick == 0 else (f"+1e{tick}" if tick > 0 else f"-1e{abs(tick)}")
            parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" class="{cls}"/>')
            parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" class="small">{label_tick}</text>')
        parts.append(f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{plot_top + plot_h}" class="axis"/>')
        parts.append(f'<line x1="{left}" y1="{plot_top + plot_h}" x2="{width - right}" y2="{plot_top + plot_h}" class="axis"/>')
        for seed_idx, seed in enumerate(seeds):
            group = [row for row in predictions if row.workload == workload and row.seed == seed]
            if not group:
                continue
            x = left + (seed_idx + 0.5) * plot_w / len(seeds)
            color = colors.get(seed, "#374151")
            signed_values = sorted(signed_value(row) for row in group)
            q25 = percentile(signed_values, 25) or 0.0
            q50 = statistics.median(signed_values)
            q75 = percentile(signed_values, 75) or 0.0
            parts.append(f'<rect x="{x - 32:.2f}" y="{y_for(plot_top, plot_h, q75):.2f}" width="64" height="{max(y_for(plot_top, plot_h, q25) - y_for(plot_top, plot_h, q75), 2):.2f}" fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<line x1="{x - 32:.2f}" y1="{y_for(plot_top, plot_h, q50):.2f}" x2="{x + 32:.2f}" y2="{y_for(plot_top, plot_h, q50):.2f}" stroke="{color}" stroke-width="3"/>')
            for row in group:
                signed = signed_value(row)
                clipped = abs(signed) > cap_log10
                px = x + x_jitter(f"{workload}:{seed}:{row.query_id}", 24.0)
                py = y_for(plot_top, plot_h, signed)
                radius = 4.2 if clipped else 3.4
                stroke = ' stroke="#dc2626" stroke-width="1.2"' if clipped else ""
                parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius}" fill="{color}" fill-opacity="0.62"{stroke}/>')
                direction = "over" if row.prediction >= row.actual else "under"
                parts.append(f'<title>seed {seed} {label} {svg_escape(row.query_id)} {direction} q={fmt(row.q_error)} actual={fmt(row.actual)} pred={fmt(row.prediction)}</title>')
            under = sum(1 for row in group if row.prediction < row.actual)
            over = sum(1 for row in group if row.prediction > row.actual)
            med_q = statistics.median(row.q_error for row in group)
            parts.append(f'<text x="{x:.2f}" y="{plot_top + plot_h + 31:.2f}" text-anchor="middle" font-size="13" font-weight="700">seed {svg_escape(seed)}</text>')
            parts.append(f'<text x="{x:.2f}" y="{plot_top + plot_h + 48:.2f}" text-anchor="middle" class="small">n={len(group)}, med q={fmt(med_q)}</text>')
            parts.append(f'<text x="{x:.2f}" y="{plot_top + plot_h + 65:.2f}" text-anchor="middle" class="small">under={under}, over={over}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def audit_artifacts(run_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    model_seeds = [str(seed) for seed in report.get("model_seeds", [])]
    required_top = [
        "paper_mscn_report.json",
        "training_rows_manifest.json",
        "sample_tables_manifest.json",
        "coverage_audit.json",
        "encoder_manifest.json",
        "generation/synthetic_labels.csv",
    ]
    audit: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "top_level_files": {},
        "seeds": {},
    }
    for rel in required_top:
        path = run_dir / rel
        audit["top_level_files"][rel] = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
    for seed in model_seeds:
        seed_dir = run_dir / "model_seeds" / f"seed_{seed}"
        seed_entry: dict[str, Any] = {
            "model_exists": (seed_dir / "paper_mscn_model.pt").exists(),
            "summary_exists": (seed_dir / "qerror_summaries.json").exists(),
            "predictions": {},
        }
        for workload, _label in WORKLOADS:
            path = seed_dir / "predictions" / f"{workload}.csv"
            rows = 0
            if path.exists():
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    rows = sum(1 for _ in csv.DictReader(handle))
            seed_entry["predictions"][workload] = {
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "rows": rows,
            }
        audit["seeds"][seed] = seed_entry
    audit["complete"] = all(item["exists"] for item in audit["top_level_files"].values()) and all(
        seed["model_exists"]
        and seed["summary_exists"]
        and all(pred["exists"] and pred["rows"] > 0 for pred in seed["predictions"].values())
        for seed in audit["seeds"].values()
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--signed-cap-log10", type=float, default=5.0)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "final_mscn_cardinality_bundle")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = read_json(run_dir / "paper_mscn_report.json")
    seeds = [str(seed) for seed in report.get("model_seeds", [])]

    predictions: list[PredictionRow] = []
    for seed in seeds:
        for workload, label in WORKLOADS:
            predictions.extend(
                read_prediction_csv(
                    run_dir / "model_seeds" / f"seed_{seed}" / "predictions" / f"{workload}.csv",
                    workload,
                    label,
                    seed,
                )
            )

    seed_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    policy_notes: list[str] = []
    aggregate = report.get("aggregate") or {}
    for workload, label in WORKLOADS:
        entry = aggregate.get(workload) or {}
        seed_stats = {str(row.get("seed")): row for row in entry.get("seeds", [])}
        for seed, stats in sorted(seed_stats.items(), key=lambda item: int(item[0])):
            seed_rows.append(
                {
                    "workload": label,
                    "seed": seed,
                    "count": stats.get("count"),
                    "mean": stats.get("mean"),
                    "median": stats.get("median"),
                    "p90": stats.get("p90"),
                    "p95": stats.get("p95"),
                    "p99": stats.get("p99"),
                    "max": stats.get("max"),
                }
            )
        best_by_median_seed, best_by_median = min(
            ((seed, float(stats["median"])) for seed, stats in seed_stats.items()),
            key=lambda item: item[1],
        )
        best_by_p95_seed, best_by_p95 = min(
            ((seed, float(stats["p95"])) for seed, stats in seed_stats.items()),
            key=lambda item: item[1],
        )
        med_of_meds = float(entry.get("median_of_medians"))
        med_p95 = float(entry.get("median_p95"))
        best_median_delta = (best_by_median / med_of_meds - 1.0) * 100.0
        best_p95_delta = (best_by_p95 / med_p95 - 1.0) * 100.0
        aggregate_rows.append(
            {
                "workload": label,
                "query_count": next(iter(seed_stats.values())).get("count") if seed_stats else "",
                "median_of_seed_medians": med_of_meds,
                "median_of_seed_p95": med_p95,
                "best_seed_by_median": best_by_median_seed,
                "best_seed_median": best_by_median,
                "best_seed_median_delta_pct": best_median_delta,
                "best_seed_by_p95": best_by_p95_seed,
                "best_seed_p95": best_by_p95,
                "best_seed_p95_delta_pct": best_p95_delta,
                "recommended_policy": "median_of_seed_summaries",
            }
        )
        if abs(best_median_delta) > 10.0 or abs(best_p95_delta) > 20.0:
            policy_notes.append(
                f"{label}: best seed materially changes results "
                f"(median delta {best_median_delta:.1f}%, p95 delta {best_p95_delta:.1f}%)."
            )

    audit = audit_artifacts(run_dir, report)
    recommendation = {
        "recommended_table_policy": "median_of_seed_summaries",
        "reason": (
            "Use the three-seed aggregate rather than the best seed. The best seed is an optimistic "
            "model-selection result and materially improves at least one workload/tail metric."
        ),
        "notes": policy_notes,
    }

    write_csv(
        out_dir / "mscn_seed_metrics.csv",
        seed_rows,
        ["workload", "seed", "count", "mean", "median", "p90", "p95", "p99", "max"],
    )
    write_csv(
        out_dir / "mscn_aggregate_table_policy.csv",
        aggregate_rows,
        [
            "workload",
            "query_count",
            "median_of_seed_medians",
            "median_of_seed_p95",
            "best_seed_by_median",
            "best_seed_median",
            "best_seed_median_delta_pct",
            "best_seed_by_p95",
            "best_seed_p95",
            "best_seed_p95_delta_pct",
            "recommended_policy",
        ],
    )
    (out_dir / "artifact_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out_dir / "table_policy_recommendation.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    render_handlebar_svg(out_dir / "mscn_seed_qerror_handlebars.svg", predictions)
    render_signed_svg(out_dir / "mscn_signed_qerror_bias_by_seed.svg", predictions, cap_log10=args.signed_cap_log10)

    readme = [
        "# Final MSCN Cardinality Bundle",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Source run: `{run_dir}`",
        "",
        "Recommended reporting policy: `median_of_seed_summaries`.",
        "",
        "Reason: choosing the single best seed is optimistic and materially changes JOB/JOB-Complex tail results.",
        "",
        "Files:",
        "- `artifact_audit.json`: verifies training/eval artifacts are present.",
        "- `mscn_aggregate_table_policy.csv`: paper table candidates and seed-selection deltas.",
        "- `mscn_seed_metrics.csv`: full seed-level Q-error metrics.",
        "- `mscn_seed_qerror_handlebars.svg`: per-seed Q-error distributions.",
        "- `mscn_signed_qerror_bias_by_seed.svg`: over/under-estimation bias by seed.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "audit_complete": audit["complete"], "recommendation": recommendation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

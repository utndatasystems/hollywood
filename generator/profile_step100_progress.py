from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    progress_path = base_dir / "decision_logs" / "20260501_014006_movie_generation_progress.jsonl"
    start_ts = float(sys.argv[1]) if len(sys.argv) > 1 else 1777625700.0

    movies: dict[int, list[dict]] = collections.defaultdict(list)
    latest: dict | None = None
    start: dict | None = None
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except Exception:
                continue
            ts = event.get("timestamp")
            if not isinstance(ts, (int, float)) or float(ts) < start_ts:
                continue
            if event.get("event") == "step100_titles_ready":
                start = event
            if event.get("event") in {"movie_stage", "movie_started", "movie_completed"}:
                latest = event
                seq = event.get("seq_idx")
                if seq is not None:
                    movies[int(seq)].append(event)

    stage_deltas: dict[str, list[float]] = collections.defaultdict(list)
    movie_totals: list[float] = []
    for events in movies.values():
        events.sort(key=lambda e: (float(e.get("timestamp", 0.0) or 0.0), float(e.get("elapsed_sec", -1.0) or -1.0)))
        prev_elapsed = 0.0
        max_elapsed = 0.0
        for event in events:
            if event.get("event") != "movie_stage":
                continue
            elapsed = event.get("elapsed_sec")
            if not isinstance(elapsed, (int, float)):
                continue
            elapsed = float(elapsed)
            stage_deltas[str(event.get("stage"))].append(max(0.0, elapsed - prev_elapsed))
            prev_elapsed = max(prev_elapsed, elapsed)
            max_elapsed = max(max_elapsed, elapsed)
        if max_elapsed:
            movie_totals.append(max_elapsed)

    print("latest=", json.dumps(latest or {}, ensure_ascii=False))
    if start and latest and latest.get("seq_idx"):
        elapsed = float(latest["timestamp"]) - float(start["timestamp"])
        seq = int(latest["seq_idx"])
        print(
            "overall "
            f"elapsed_sec={elapsed:.1f} seq={seq} movies_per_sec={seq / elapsed:.4f} "
            f"sec_per_movie={elapsed / seq:.2f} movies_per_hour={3600.0 * seq / elapsed:.1f}"
        )
    if movie_totals:
        print(
            "movie_elapsed "
            f"samples={len(movie_totals)} mean={statistics.mean(movie_totals):.2f}s "
            f"median={statistics.median(movie_totals):.2f}s p90={percentile(movie_totals, 0.9):.2f}s "
            f"p99={percentile(movie_totals, 0.99):.2f}s"
        )

    rows: list[tuple[float, float, float, int, str]] = []
    for stage, values in stage_deltas.items():
        if len(values) < 20:
            continue
        rows.append((statistics.mean(values), statistics.median(values), sum(values), len(values), stage))

    print("slowest_stage_deltas")
    for mean, median, total, count, stage in sorted(rows, reverse=True)[:30]:
        print(f"{stage:35s} mean={mean:7.4f}s median={median:7.4f}s total={total:9.1f}s n={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

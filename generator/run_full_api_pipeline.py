from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from model_defaults import model_for_role

DEFAULT_MODEL = model_for_role("entity_gen")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolved_profile(requested: str, n_movies: int) -> str:
    if requested != "auto":
        return requested
    return "smoke" if int(n_movies) <= 250 else "standard"


def _smoke_counts(n_movies: int) -> dict[str, int]:
    movies = max(1, int(n_movies))
    return {
        "n_titles": movies,
        "n_persons": max(300, int(round(movies * 3.0))),
        "n_companies": max(45, int(round(movies * 0.20))),
        "n_keywords": max(260, int(round(movies * 2.6))),
        "n_characters": max(260, int(round(movies * 2.6))),
    }


def _standard_counts(n_movies: int) -> dict[str, int]:
    movies = max(1, int(n_movies))
    return {
        "n_titles": movies,
        "n_persons": max(1200, int(round(movies * 2.25))),
        "n_companies": max(120, int(round(movies * 0.15))),
        "n_keywords": max(350, int(round(movies * 0.19))),
        "n_characters": max(1500, int(round(movies * 3.93))),
    }


def _resolved_counts(args: argparse.Namespace, profile: str) -> dict[str, int | None]:
    counts: dict[str, int | None] = {
        "n_titles": args.n_titles,
        "n_persons": args.n_persons,
        "n_companies": args.n_companies,
        "n_keywords": args.n_keywords,
        "n_characters": args.n_characters,
    }
    if profile == "smoke":
        defaults = _smoke_counts(args.n_movies)
    else:
        defaults = _standard_counts(args.n_movies)
    for key, value in defaults.items():
        if counts[key] is None:
            counts[key] = int(value)
    return counts


def _resolved_years(args: argparse.Namespace, profile: str) -> tuple[int | None, int | None]:
    start_year = args.start_year
    end_year = args.end_year
    if (start_year is None) != (end_year is None):
        raise ValueError("start-year and end-year must be provided together")
    if start_year is None and profile == "standard":
        return 1950, 2025
    return start_year, end_year


def _format_count(value: int | None) -> str:
    return str(int(value)) if value is not None else "(runner auto)"


def _build_runner_env(
    *,
    run_id: str,
    log_dir: Path,
    profile: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DATA_SYS_OVERLOAD_PROFILE"] = profile
    env["DATA_SYS_LLM_USAGE_LOG"] = str(log_dir / f"{run_id}_llm_usage.jsonl")

    if args.llm_max_attempts is not None:
        env["DATA_SYS_LLM_MAX_ATTEMPTS"] = str(max(1, int(args.llm_max_attempts)))

    if args.llm_timeout_sec is not None:
        env["DATA_SYS_LLM_TIMEOUT_SEC"] = str(max(5.0, float(args.llm_timeout_sec)))

    if args.llm_base_delay_sec is not None:
        env["DATA_SYS_LLM_BASE_DELAY_SEC"] = str(max(0.0, float(args.llm_base_delay_sec)))

    if args.llm_max_delay_sec is not None:
        env["DATA_SYS_LLM_MAX_DELAY_SEC"] = str(max(0.0, float(args.llm_max_delay_sec)))
    # Keep smoke runs on the provider defaults unless the caller overrides them.
    # Required bootstrap artifacts are too important to fail fast on transient 503s.

    if args.person_enrich_batch_size is not None:
        env["DATA_SYS_PERSON_ENRICH_BATCH_SIZE"] = str(max(1, int(args.person_enrich_batch_size)))
    elif profile == "smoke":
        env["DATA_SYS_PERSON_ENRICH_BATCH_SIZE"] = "24"

    if args.person_enrich_outer_retries is not None:
        env["DATA_SYS_PERSON_ENRICH_OUTER_RETRIES"] = str(max(1, int(args.person_enrich_outer_retries)))
    elif profile == "smoke":
        env["DATA_SYS_PERSON_ENRICH_OUTER_RETRIES"] = "2"

    if args.latent_batch_size is not None:
        env["DATA_SYS_LATENT_BATCH_SIZE"] = str(max(1, int(args.latent_batch_size)))
    elif profile == "smoke":
        env["DATA_SYS_LATENT_BATCH_SIZE"] = "24"

    if args.latent_max_retries is not None:
        env["DATA_SYS_LATENT_MAX_RETRIES"] = str(max(1, int(args.latent_max_retries)))
    elif profile == "smoke":
        env["DATA_SYS_LATENT_MAX_RETRIES"] = "3"

    return env


def _run_command(cmd: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(shlex.quote(part) for part in cmd)}\n")
        for name in (
            "DATA_SYS_OVERLOAD_PROFILE",
            "DATA_SYS_LLM_MAX_ATTEMPTS",
            "DATA_SYS_LLM_TIMEOUT_SEC",
            "DATA_SYS_LLM_BASE_DELAY_SEC",
            "DATA_SYS_LLM_MAX_DELAY_SEC",
            "DATA_SYS_PERSON_ENRICH_BATCH_SIZE",
            "DATA_SYS_PERSON_ENRICH_OUTER_RETRIES",
            "DATA_SYS_LATENT_BATCH_SIZE",
            "DATA_SYS_LATENT_MAX_RETRIES",
            "DATA_SYS_LLM_USAGE_LOG",
        ):
            if env.get(name):
                log.write(f"@env {name}={env[name]}\n")
        log.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            try:
                print(line, end="")
            except UnicodeEncodeError:
                encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
                safe_line = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
                print(safe_line, end="")
            log.write(line)
            log.flush()
        exit_code = process.wait()
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="High-level launcher for a fresh Mirage API pipeline run.")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--n-movies", type=int, default=200000)
    parser.add_argument("--until-step", type=int, default=130)
    parser.add_argument("--profile", choices=["auto", "smoke", "standard"], default="auto")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--n-persons", type=int, default=None)
    parser.add_argument("--n-companies", type=int, default=None)
    parser.add_argument("--n-keywords", type=int, default=None)
    parser.add_argument("--n-characters", type=int, default=None)
    parser.add_argument("--n-titles", type=int, default=None)
    parser.add_argument("--llm-max-attempts", type=int, default=None)
    parser.add_argument("--llm-timeout-sec", type=float, default=None)
    parser.add_argument("--llm-base-delay-sec", type=float, default=None)
    parser.add_argument("--llm-max-delay-sec", type=float, default=None)
    parser.add_argument("--person-enrich-batch-size", type=int, default=None)
    parser.add_argument("--person-enrich-outer-retries", type=int, default=None)
    parser.add_argument("--latent-batch-size", type=int, default=None)
    parser.add_argument("--latent-max-retries", type=int, default=None)
    parser.add_argument("--run-calibration", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    log_dir = base_dir / "_runner_logs" / "full_api_pipeline"
    run_id = f"{int(args.n_movies)}movies_{_timestamp()}"
    log_path = log_dir / f"{run_id}.log"
    python = sys.executable
    profile = _resolved_profile(str(args.profile), int(args.n_movies))
    counts = _resolved_counts(args, profile)
    start_year, end_year = _resolved_years(args, profile)
    runner_env = _build_runner_env(run_id=run_id, log_dir=log_dir, profile=profile, args=args)
    usage_log = Path(runner_env["DATA_SYS_LLM_USAGE_LOG"])

    print("=" * 72)
    print("MIRAGE HIGH-LEVEL API RUNNER")
    print("=" * 72)
    print(f"Base dir:      {base_dir}")
    print(f"Movies:        {int(args.n_movies)}")
    print(f"Until step:    {int(args.until_step)}")
    print(f"Profile:       {profile}")
    print(f"Year span:     {start_year if start_year is not None else '(generator default)'} -> {end_year if end_year is not None else '(generator default)'}")
    print(f"Persons:       {_format_count(counts['n_persons'])}")
    print(f"Companies:     {_format_count(counts['n_companies'])}")
    print(f"Keywords:      {_format_count(counts['n_keywords'])}")
    print(f"Characters:    {_format_count(counts['n_characters'])}")
    print(f"Titles:        {_format_count(counts['n_titles'])}")
    print(f"Model:         {args.model}")
    print(f"Calibration:   {bool(args.run_calibration)}")
    print(f"LLM attempts:  {runner_env.get('DATA_SYS_LLM_MAX_ATTEMPTS', '(provider default)')}")
    print(f"LLM delay:     {runner_env.get('DATA_SYS_LLM_BASE_DELAY_SEC', '(provider default)')} -> {runner_env.get('DATA_SYS_LLM_MAX_DELAY_SEC', '(provider default)')}")
    print(f"Person batch:  {runner_env.get('DATA_SYS_PERSON_ENRICH_BATCH_SIZE', '(script default)')}")
    print(f"Latent batch:  {runner_env.get('DATA_SYS_LATENT_BATCH_SIZE', '(script default)')}")
    print(f"Log file:      {log_path}")
    print(f"Usage log:     {usage_log}")
    print("=" * 72)

    started = time.time()

    if bool(args.run_calibration):
        calibration_cmd = [
            python,
            str(base_dir / "calibrate_llm_prompts.py"),
            "--base-dir",
            str(base_dir),
            "--model",
            str(args.model),
        ]
        _run_command(calibration_cmd, cwd=base_dir, log_path=log_path, env=runner_env)

    pipeline_cmd = [
        python,
        str(base_dir / "run_pipeline.py"),
        "--fresh",
        "--n-movies",
        str(int(args.n_movies)),
        "--until-step",
        str(int(args.until_step)),
        "--model",
        str(args.model),
    ]
    if start_year is not None and end_year is not None:
        pipeline_cmd.extend(["--start-year", str(int(start_year)), "--end-year", str(int(end_year))])
    for flag_name, cli_name in (
        ("n_persons", "--n-persons"),
        ("n_companies", "--n-companies"),
        ("n_keywords", "--n-keywords"),
        ("n_characters", "--n-characters"),
        ("n_titles", "--n-titles"),
    ):
        value = counts[flag_name]
        if value is not None:
            pipeline_cmd.extend([cli_name, str(int(value))])
    _run_command(pipeline_cmd, cwd=base_dir, log_path=log_path, env=runner_env)

    elapsed = time.time() - started
    print("=" * 72)
    print(f"Completed in {elapsed / 60.0:.1f} minutes")
    print(f"Log file: {log_path}")
    print(f"Usage log: {usage_log}")
    print("=" * 72)


if __name__ == "__main__":
    main()

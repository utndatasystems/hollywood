from __future__ import annotations

import faulthandler
import os
import sys
import time

from pipeline_runtime import resolve_workspace
from world_state import WorldState


def main() -> int:
    faulthandler.enable(all_threads=True)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.getenv("DATA_SYS_PIPELINE_CONFIG") or os.path.join(base_dir, "v18_config.json")
    if not os.path.exists(config_path):
        config_path = None
    workspace = resolve_workspace(
        script_dir=base_dir,
        data_dir=base_dir,
        output_dir=base_dir,
        config_path=config_path,
    )
    print(f"[probe] base_dir={base_dir}", flush=True)
    print(f"[probe] config_path={config_path}", flush=True)
    start = time.time()
    world = WorldState(base_dir, seed=42, config_path=config_path, workspace=workspace)
    print("[probe] calling world.load()", flush=True)
    world.load()
    print(
        "[probe] loaded "
        f"persons={len(world.persons)} companies={len(world.companies)} "
        f"keywords={len(world.keywords)} titles={len(world.title_bank)} "
        f"characters={len(world.character_bank)} elapsed={time.time() - start:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

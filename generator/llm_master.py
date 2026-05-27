from __future__ import annotations

"""LLM action executor for WorldState mutations.

This rewrite keeps the public API stable but fixes several structural problems:

- duration-based effects restore cleanly and invalidate stale caches
- person-row mutations refresh the year cache used by assembly.py
- genre-weight actions are converted into the *additive deltas* currently
  consumed by sample_movie_concept(), instead of pretending the pipeline uses
  multiplicative semantics end-to-end
- cascade edge changes are routed through apply_world_patches so they participate
  in temporal versioning rather than mutating active edges in-place forever
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from temporal_evolution_api import (
    PatchApplyReport,
    _clamp,
    _company_exists,
    _find_person_row,
    _person_exists,
    _safe_int,
    _safe_str,
    _sync_sim_cache_for_persons,
    _invalidate_year_cache,
    apply_world_patches,
)

try:
    from contracts import GENRES, COUNTRIES, GENRE_WEIGHTS, COUNTRY_WEIGHTS
except Exception:
    GENRES = []
    COUNTRIES = []
    GENRE_WEIGHTS = {}
    COUNTRY_WEIGHTS = {}


@dataclass
class ActionReport:
    applied: int = 0
    skipped: int = 0
    errors: int = 0
    skipped_reasons: List[str] = field(default_factory=list)

    def skip(self, reason: str) -> None:
        self.skipped += 1
        self.skipped_reasons.append(str(reason))

    def log_error(self, reason: str) -> None:
        self.errors += 1
        self.skipped_reasons.append(f"ERROR: {reason}")


_LEGACY_OPS = {
    "retire_person",
    "career_stage_transition",
    "dissolve_company",
    "company_tier_transition",
    "genre_trend_shift",
    "set_yearly_max",
    "person_latent_delta",
    "company_latent_delta",
    "edge_add",
    "edge_update",
    "edge_expire",
}


class LLMMasterClass:
    def __init__(self, world):
        self.world = world
        self._ensure_world_fields()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, llm_response: dict, year: int) -> ActionReport:
        rep = ActionReport()
        if not isinstance(llm_response, dict):
            rep.skip("llm_response is not a dict")
            return rep
        actions = llm_response.get("actions", [])
        if not isinstance(actions, list):
            rep.skip("'actions' is not a list")
            return rep

        for call in actions:
            if not isinstance(call, dict):
                rep.skip("action_call is not a dict")
                continue
            action = _safe_str(call.get("action"), "").strip()
            params = call.get("params") or {}
            if not isinstance(params, dict):
                params = {}
            try:
                self._dispatch(action, params, year, rep)
            except Exception as exc:
                rep.log_error(f"Exception in action '{action}': {exc}")
        return rep

    def apply_expiry(self, current_year: int) -> None:
        remaining = []
        for effect in list(self.world.active_effects):
            if _safe_int(effect.get("expires_year"), 9999) <= current_year:
                self._restore_effect(effect)
            else:
                remaining.append(effect)
        self.world.active_effects = remaining

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, action: str, params: dict, year: int, rep: ActionReport) -> None:
        if action == "career_pause":
            self._career_pause(params, year, rep)
        elif action == "boost_person":
            self._boost_person(params, year, rep)
        elif action == "change_genre_affinity":
            self._change_genre_affinity(params, rep)
        elif action == "change_style_tags":
            self._change_style_tags(params, rep)
        elif action == "change_director_specialty":
            self._change_director_specialty(params, rep)
        elif action == "merge_companies":
            self._merge_companies(params, year, rep)
        elif action == "adjust_company_specialty":
            self._adjust_company_specialty(params, rep)
        elif action == "adjust_genre_weight":
            self._adjust_genre_weight(params, year, rep)
        elif action == "adjust_country_weight":
            self._adjust_country_weight(params, year, rep)
        elif action == "award_prestige_shift":
            self._award_prestige_shift(params, rep)
        elif action == "cascade_edge_change":
            self._cascade_edge_change(params, year, rep)
        elif action == "create_world_event":
            self._create_world_event(params, year, rep)
        elif action in _LEGACY_OPS:
            op = {"op": action}
            op.update(params)
            patch_rep = apply_world_patches(self.world, [op], from_year=year, to_year=year + 1)
            self._merge_patch_report_into_action_report(patch_rep, rep)
        elif not action:
            rep.skip("empty action name")
        else:
            rep.skip(f"Unknown action: '{action}'")

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _career_pause(self, params: dict, year: int, rep: ActionReport) -> None:
        pid = _safe_int(params.get("person_id"), 0)
        duration = max(1, min(5, _safe_int(params.get("duration_years"), 1)))
        if not _person_exists(self.world, pid):
            rep.skip(f"career_pause: unknown person_id={pid}")
            return
        if pid in self.world.paused_persons:
            rep.skip(f"career_pause: person {pid} is already paused")
            return
        if getattr(self.world, "persons", None) is None:
            rep.skip("career_pause: no persons DataFrame")
            return

        row = _find_person_row(self.world, pid)
        if row is None or len(row) == 0:
            rep.skip(f"career_pause: cannot locate row for person_id={pid}")
            return
        original_ret = _safe_int(row.iloc[0].get("retirement_year"), 2100)

        mask = self.world.persons["person_id"].astype(int) == pid
        self.world.persons.loc[mask, "retirement_year"] = int(year - 1)
        self.world.paused_persons[pid] = {
            "original_retirement_year": int(original_ret),
            "pause_until": int(year + duration),
        }
        self.world.active_effects.append({
            "action": "career_pause_restore",
            "person_id": int(pid),
            "original_retirement_year": int(original_ret),
            "expires_year": int(year + duration),
        })
        _invalidate_year_cache(self.world)
        rep.applied += 1

    def _boost_person(self, params: dict, year: int, rep: ActionReport) -> None:
        pid = _safe_int(params.get("person_id"), 0)
        multiplier = _clamp(params.get("multiplier", 1.5), 1.0, 5.0)
        duration = max(1, min(5, _safe_int(params.get("duration_years"), 2)))
        if not _person_exists(self.world, pid):
            rep.skip(f"boost_person: unknown person_id={pid}")
            return
        if getattr(self.world, "persons", None) is None:
            rep.skip("boost_person: no persons DataFrame")
            return

        mask = self.world.persons["person_id"].astype(int) == pid
        row = self.world.persons[mask]
        if len(row) == 0:
            rep.skip(f"boost_person: cannot locate row for person_id={pid}")
            return

        original_pw = float(row.iloc[0].get("pop_weight", 0.3) or 0.3)
        new_pw = _clamp(original_pw * multiplier, 0.0, 0.99)
        self.world.persons.loc[mask, "pop_weight"] = float(new_pw)
        self.world.person_pop_weight[int(pid)] = float(new_pw)
        self.world.active_effects.append({
            "action": "boost_person_restore",
            "person_id": int(pid),
            "original_pop_weight": float(original_pw),
            "expires_year": int(year + duration),
        })
        _invalidate_year_cache(self.world)
        rep.applied += 1

    def _change_genre_affinity(self, params: dict, rep: ActionReport) -> None:
        pid = _safe_int(params.get("person_id"), 0)
        genre = _safe_str(params.get("genre"), "").strip()
        delta = float(params.get("delta", 0.1) or 0.1)
        if not _person_exists(self.world, pid):
            rep.skip(f"change_genre_affinity: unknown person_id={pid}")
            return
        if not genre:
            rep.skip("change_genre_affinity: empty genre")
            return
        if getattr(self.world, "persons", None) is None:
            rep.skip("change_genre_affinity: no persons DataFrame")
            return

        mask = self.world.persons["person_id"].astype(int) == pid
        row = self.world.persons[mask]
        if len(row) == 0:
            rep.skip(f"change_genre_affinity: cannot locate row for person_id={pid}")
            return
        current = _safe_str(row.iloc[0].get("genre_affinity"), "")
        values = {g.strip() for g in current.replace(",", ";").split(";") if g.strip()}
        if delta > 0:
            values.add(genre)
        elif delta < -0.3 and genre in values and len(values) > 1:
            values.discard(genre)
        self.world.persons.loc[mask, "genre_affinity"] = ";".join(sorted(values))
        self._refresh_person_structural_state(pid)
        rep.applied += 1

    def _change_style_tags(self, params: dict, rep: ActionReport) -> None:
        pid = _safe_int(params.get("person_id"), 0)
        add_tags = params.get("add_tags") or []
        remove_tags = params.get("remove_tags") or []
        if not _person_exists(self.world, pid):
            rep.skip(f"change_style_tags: unknown person_id={pid}")
            return
        if getattr(self.world, "persons", None) is None:
            rep.skip("change_style_tags: no persons DataFrame")
            return

        mask = self.world.persons["person_id"].astype(int) == pid
        row = self.world.persons[mask]
        if len(row) == 0:
            rep.skip(f"change_style_tags: cannot locate row for person_id={pid}")
            return
        current = _safe_str(row.iloc[0].get("style_tags"), "")
        values = {t.strip().lower() for t in current.replace(",", ";").split(";") if t.strip()}
        for tag in add_tags:
            if isinstance(tag, str) and tag.strip():
                values.add(tag.strip().lower())
        for tag in remove_tags:
            if isinstance(tag, str):
                values.discard(tag.strip().lower())
        self.world.persons.loc[mask, "style_tags"] = ";".join(sorted(values))
        self._refresh_person_structural_state(pid)
        rep.applied += 1

    def _change_director_specialty(self, params: dict, rep: ActionReport) -> None:
        pid = _safe_int(params.get("person_id"), 0)
        genres = [str(g).strip() for g in (params.get("genres") or []) if str(g).strip()]
        if not _person_exists(self.world, pid):
            rep.skip(f"change_director_specialty: unknown person_id={pid}")
            return
        if not genres:
            rep.skip("change_director_specialty: empty genres list")
            return
        if getattr(self.world, "persons", None) is None:
            rep.skip("change_director_specialty: no persons DataFrame")
            return

        mask = self.world.persons["person_id"].astype(int) == pid
        self.world.persons.loc[mask, "genre_affinity"] = ";".join(genres)
        self._refresh_person_structural_state(pid)
        rep.applied += 1

    def _merge_companies(self, params: dict, year: int, rep: ActionReport) -> None:
        cid_a = _safe_int(params.get("company_a_id"), 0)
        cid_b = _safe_int(params.get("company_b_id"), 0)
        if cid_a == cid_b:
            rep.skip("merge_companies: company_a_id == company_b_id")
            return
        if not _company_exists(self.world, cid_a):
            rep.skip(f"merge_companies: unknown company_a_id={cid_a}")
            return
        if not _company_exists(self.world, cid_b):
            rep.skip(f"merge_companies: unknown company_b_id={cid_b}")
            return
        if getattr(self.world, "companies", None) is not None:
            mask_b = self.world.companies["company_id"].astype(int) == cid_b
            row_b = self.world.companies[mask_b]
            if len(row_b) == 0:
                rep.skip(f"merge_companies: cannot locate row for company_b_id={cid_b}")
                return
            founded_b = _safe_int(row_b.iloc[0].get("founded_year"), year - 10)
            if year < founded_b:
                rep.skip(f"merge_companies: year {year} < founded_year {founded_b} for company_b_id={cid_b}")
                return
            old_defunct = row_b.iloc[0].get("defunct_year")
            if old_defunct is None or (isinstance(old_defunct, float) and old_defunct != old_defunct):
                self.world.companies.loc[mask_b, "defunct_year"] = int(year)
            else:
                self.world.companies.loc[mask_b, "defunct_year"] = min(int(old_defunct), int(year))

        self.world._merge_families.setdefault(int(cid_a), set()).add(int(cid_b))
        self.world._merge_families.setdefault(int(cid_b), set()).add(int(cid_a))
        if hasattr(self.world, "company_family"):
            self.world.company_family.setdefault(int(cid_a), set()).add(int(cid_b))
            self.world.company_family.setdefault(int(cid_b), set()).add(int(cid_a))
        rep.applied += 1

    def _adjust_company_specialty(self, params: dict, rep: ActionReport) -> None:
        cid = _safe_int(params.get("company_id"), 0)
        genres = [str(g).strip() for g in (params.get("genres") or []) if str(g).strip()]
        if not _company_exists(self.world, cid):
            rep.skip(f"adjust_company_specialty: unknown company_id={cid}")
            return
        if not genres:
            rep.skip("adjust_company_specialty: empty genres list")
            return
        if getattr(self.world, "companies", None) is None:
            rep.skip("adjust_company_specialty: no companies DataFrame")
            return
        mask = self.world.companies["company_id"].astype(int) == cid
        self.world.companies.loc[mask, "specialty_genres"] = ";".join(genres)
        rep.applied += 1

    def _adjust_genre_weight(self, params: dict, year: int, rep: ActionReport) -> None:
        """Store an additive delta because assembly.sample_movie_concept() consumes deltas.

        The prompt/action schema talks about multipliers.  The current generator
        consumes additive deltas.  We bridge that mismatch here instead of letting
        the system silently mean two different things in two different files.
        """
        genre = _safe_str(params.get("genre"), "").strip()
        multiplier = _clamp(params.get("multiplier", 1.0), 0.1, 5.0)
        duration = max(1, min(10, _safe_int(params.get("duration_years"), 3)))
        if not genre:
            rep.skip("adjust_genre_weight: empty genre")
            return
        if GENRES and genre not in GENRES:
            rep.skip(f"adjust_genre_weight: unknown genre '{genre}'")
            return
        base_weight = float(GENRE_WEIGHTS.get(genre, 0.08) or 0.08)
        delta = _clamp(base_weight * (multiplier - 1.0), -0.12, 0.12)
        old = float(self.world.genre_weight_overrides.get(genre, 0.0))
        self.world.genre_weight_overrides[genre] = _clamp(old + delta, -0.20, 0.20)
        self.world.active_effects.append({
            "action": "genre_weight_restore",
            "genre": genre,
            "delta": float(delta),
            "expires_year": int(year + duration),
        })
        rep.applied += 1

    def _adjust_country_weight(self, params: dict, year: int, rep: ActionReport) -> None:
        country = _safe_str(params.get("country"), "").strip()
        multiplier = _clamp(params.get("multiplier", 1.0), 0.1, 10.0)
        duration = max(1, min(10, _safe_int(params.get("duration_years"), 5)))
        if not country:
            rep.skip("adjust_country_weight: empty country")
            return
        if COUNTRIES and country not in COUNTRIES:
            rep.skip(f"adjust_country_weight: unknown country '{country}'")
            return
        # Country weighting is not yet consumed consistently downstream, so keep
        # the semantics literal here.
        self.world.country_weight_overrides[country] = float(multiplier)
        self.world.active_effects.append({
            "action": "country_weight_restore",
            "country": country,
            "expires_year": int(year + duration),
        })
        rep.applied += 1

    def _award_prestige_shift(self, params: dict, rep: ActionReport) -> None:
        ceremony = _safe_str(params.get("ceremony"), "").strip()
        delta = _clamp(params.get("delta", 0.0), -0.5, 0.5)
        if not ceremony:
            rep.skip("award_prestige_shift: empty ceremony name")
            return
        old = float(self.world.award_prestige.get(ceremony, 0.0))
        self.world.award_prestige[ceremony] = _clamp(old + delta, -1.0, 1.0)
        rep.applied += 1

    def _cascade_edge_change(self, params: dict, year: int, rep: ActionReport) -> None:
        pid = _safe_int(params.get("person_id"), 0)
        delta = _clamp(params.get("delta", 0.0), -0.5, 0.5)
        n = max(1, min(20, _safe_int(params.get("n"), 5)))
        if not _person_exists(self.world, pid):
            rep.skip(f"cascade_edge_change: unknown person_id={pid}")
            return
        edge_graph = getattr(self.world, "edge_graph", None)
        if edge_graph is None:
            rep.skip("cascade_edge_change: no edge_graph loaded")
            return
        edges = getattr(edge_graph, "edges", [])
        touching = []
        for edge in edges:
            src = _safe_int(edge.get("src_id"), -1)
            dst = _safe_int(edge.get("dst_id"), -1)
            if pid not in (src, dst):
                continue
            touching.append((float(edge.get("weight", 0.0) or 0.0), {
                "op": "edge_update",
                "edge_type": _safe_str(edge.get("edge_type"), ""),
                "src_id": src,
                "dst_id": dst,
                "delta_weight": float(delta),
                "reason": f"cascade_edge_change around person_id={pid}",
            }))
        if not touching:
            rep.skip(f"cascade_edge_change: no edges found for person_id={pid}")
            return
        touching.sort(key=lambda kv: kv[0], reverse=True)
        patch_rep = apply_world_patches(self.world, [patch for _, patch in touching[:n]], from_year=year, to_year=year + 1)
        self._merge_patch_report_into_action_report(patch_rep, rep)

    def _create_world_event(self, params: dict, year: int, rep: ActionReport) -> None:
        event_type = _safe_str(params.get("event_type"), "unknown").strip() or "unknown"
        narrative = _safe_str(params.get("narrative"), "").strip()
        duration = max(0, _safe_int(params.get("duration_years"), 0))
        affected_id = params.get("affected_entity_id")
        affected_type = _safe_str(params.get("affected_entity_type"), "").strip() or None
        parameter_delta = params.get("parameter_delta") or {}
        if not narrative:
            rep.skip("create_world_event: empty narrative -- event not logged")
            return
        event_id = len(self.world.world_events) + 1
        self.world.world_events.append({
            "event_id": int(event_id),
            "year": int(year),
            "event_type": event_type,
            "description": narrative,
            "duration_years": int(duration),
            "affected_entity_id": int(affected_id) if affected_id is not None else None,
            "affected_entity_type": affected_type,
            "parameter_delta_json": json.dumps(parameter_delta, ensure_ascii=False) if parameter_delta else None,
        })
        rep.applied += 1

    # ------------------------------------------------------------------
    # Expiry / restore
    # ------------------------------------------------------------------

    def _restore_effect(self, effect: dict) -> None:
        action = _safe_str(effect.get("action"), "").strip()
        try:
            if action == "career_pause_restore":
                pid = _safe_int(effect.get("person_id"), 0)
                orig_ret = _safe_int(effect.get("original_retirement_year"), 2100)
                if getattr(self.world, "persons", None) is not None:
                    self.world.persons.loc[self.world.persons["person_id"].astype(int) == pid, "retirement_year"] = int(orig_ret)
                self.world.paused_persons.pop(pid, None)
                _invalidate_year_cache(self.world)

            elif action == "boost_person_restore":
                pid = _safe_int(effect.get("person_id"), 0)
                orig_pw = float(effect.get("original_pop_weight", 0.3) or 0.3)
                if getattr(self.world, "persons", None) is not None:
                    self.world.persons.loc[self.world.persons["person_id"].astype(int) == pid, "pop_weight"] = float(orig_pw)
                self.world.person_pop_weight[pid] = float(orig_pw)
                _invalidate_year_cache(self.world)

            elif action == "genre_weight_restore":
                genre = _safe_str(effect.get("genre"), "")
                delta = float(effect.get("delta", 0.0) or 0.0)
                if genre:
                    old = float(self.world.genre_weight_overrides.get(genre, 0.0))
                    new = old - delta
                    if abs(new) < 1e-9:
                        self.world.genre_weight_overrides.pop(genre, None)
                    else:
                        self.world.genre_weight_overrides[genre] = new

            elif action == "country_weight_restore":
                country = _safe_str(effect.get("country"), "")
                if country:
                    self.world.country_weight_overrides.pop(country, None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_world_fields(self) -> None:
        defaults = {
            "active_effects": [],
            "world_events": [],
            "genre_weight_overrides": {},
            "country_weight_overrides": {},
            "award_prestige": {},
            "paused_persons": {},
            "_merge_families": {},
        }
        for attr, default in defaults.items():
            if not hasattr(self.world, attr):
                setattr(self.world, attr, default.copy() if isinstance(default, dict) else list(default) if isinstance(default, list) else default)

    def _refresh_person_structural_state(self, pid: int) -> None:
        _sync_sim_cache_for_persons(self.world, [int(pid)])
        _invalidate_year_cache(self.world)

    @staticmethod
    def _merge_patch_report_into_action_report(patch_rep: PatchApplyReport, rep: ActionReport) -> None:
        rep.applied += patch_rep.applied
        rep.skipped += patch_rep.skipped
        rep.errors += patch_rep.errors
        rep.skipped_reasons.extend(patch_rep.messages)

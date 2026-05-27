from __future__ import annotations

import json
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


HOT_UNDIRECTED_TYPES = {
    "friendship",
    "rivalry",
    "collaboration",
    "chemistry",
    "clique",
    "former_collaborator",
}
HOT_DIRECTED_TYPES = {"mentorship", "avoid"}
HOT_EDGE_TYPES = HOT_UNDIRECTED_TYPES | HOT_DIRECTED_TYPES

COLD_CP_TYPES = {"brand_fit", "employment", "blacklist", "exclusive_deal"}
COLD_CC_TYPES = {"co_production", "market_rival", "subsidiary"}

UNDIRECTED_TYPES = HOT_UNDIRECTED_TYPES | {"co_production", "market_rival", "subsidiary"}
NEGATIVE_EDGE_TYPES = {"rivalry", "avoid", "blacklist", "market_rival"}

RUNTIME_VERSION = 1
INT32_MIN = -2_147_483_648
INT32_MAX = 2_147_483_647
ACTIVE_FLAG = 1
DELTA_FLUSH_THRESHOLD = 10_000
CLOSURE_FLUSH_THRESHOLD = 10_000
EVENT_FLUSH_THRESHOLD = 20_000
OVERLAY_COMPACT_RATIO = 0.30
OVERLAY_COMPACT_SIZE = 100_000

TYPE_CODE_TO_NAME = [
    "friendship",
    "rivalry",
    "collaboration",
    "chemistry",
    "clique",
    "former_collaborator",
    "mentorship",
    "avoid",
    "brand_fit",
    "employment",
    "blacklist",
    "exclusive_deal",
    "co_production",
    "market_rival",
    "subsidiary",
]
TYPE_NAME_TO_CODE = {name: idx for idx, name in enumerate(TYPE_CODE_TO_NAME)}

HISTORY_SCHEMA = pa.schema(
    [
        ("row_id", pa.int64()),
        ("src_id", pa.int64()),
        ("dst_id", pa.int64()),
        ("src_name", pa.string()),
        ("dst_name", pa.string()),
        ("src_type", pa.string()),
        ("dst_type", pa.string()),
        ("edge_type", pa.string()),
        ("sign", pa.string()),
        ("weight", pa.float32()),
        ("raw_weight", pa.float32()),
        ("reason", pa.string()),
        ("source_batch", pa.string()),
        ("source_kind", pa.string()),
        ("valid_from", pa.int32()),
        ("valid_to", pa.int32()),
        ("community_id", pa.int32()),
        ("_scd2_retired", pa.bool_()),
        ("_overridden_by_rivalry", pa.bool_()),
    ]
)

EVENT_SCHEMA = pa.schema(
    [
        ("event_year", pa.int32()),
        ("event_kind", pa.string()),
        ("row_id", pa.int64()),
        ("prior_row_id", pa.int64()),
        ("src_id", pa.int64()),
        ("dst_id", pa.int64()),
        ("edge_type", pa.string()),
        ("weight", pa.float32()),
        ("valid_from", pa.int32()),
        ("valid_to", pa.int32()),
        ("reason", pa.string()),
        ("source_kind", pa.string()),
    ]
)

CLOSURE_SCHEMA = pa.schema(
    [
        ("row_id", pa.int64()),
        ("valid_to", pa.int32()),
        ("_scd2_retired", pa.bool_()),
    ]
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, "", "nan"):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "nan"):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _norm_year(value: Any) -> int:
    if value in (None, "", "nan"):
        return INT32_MAX
    return _safe_int(value, INT32_MAX)


def _display_year(value: int) -> int | None:
    if value >= INT32_MAX or value <= INT32_MIN:
        return None
    return int(value)


def _normalize_seed32(seed: int) -> int:
    return int(seed) & 0xFFFFFFFF


def _norm_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _edge_sign(edge_type: str, sign: str | None = None) -> str:
    if sign in {"+", "-"}:
        return str(sign)
    return "-" if edge_type in NEGATIVE_EDGE_TYPES else "+"


def _edge_entity_types(edge_type: str) -> tuple[str, str]:
    if edge_type in HOT_EDGE_TYPES:
        return ("person", "person")
    if edge_type in COLD_CP_TYPES:
        return ("company", "person")
    return ("company", "company")


def _canonical_ids(src_id: int, dst_id: int, edge_type: str) -> tuple[int, int]:
    if edge_type in UNDIRECTED_TYPES:
        a = int(src_id)
        b = int(dst_id)
        return (a, b) if a <= b else (b, a)
    return (int(src_id), int(dst_id))


def pack_edge_key(src_id: int, dst_id: int, edge_type: str) -> int:
    src, dst = _canonical_ids(src_id, dst_id, edge_type)
    return (int(src) << 32) | int(dst)


def _active_in_year(valid_from: int, valid_to: int, year: int | None) -> bool:
    if year is None:
        return True
    return int(valid_from) <= int(year) <= int(valid_to)


def _history_writer(path: Path, schema: pa.Schema) -> ipc.RecordBatchFileWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    sink = pa.OSFile(str(path), "wb")
    return ipc.new_file(sink, schema, options=ipc.IpcWriteOptions(compression="lz4"))


@dataclass(slots=True)
class EdgeRecord:
    row_id: int
    src_id: int
    dst_id: int
    edge_type: str
    sign: str
    weight: float
    valid_from: int
    valid_to: int
    src_type: str
    dst_type: str
    src_name: str = ""
    dst_name: str = ""
    reason: str = ""
    source_kind: str = ""
    source_batch: str = ""
    raw_weight: float | None = None
    community_id: int | None = None
    overridden_by_rivalry: bool = False

    @property
    def key(self) -> int:
        return pack_edge_key(self.src_id, self.dst_id, self.edge_type)

    def is_active(self, year: int | None = None) -> bool:
        return _active_in_year(self.valid_from, self.valid_to, year)

    def to_history_row(self) -> dict[str, Any]:
        return {
            "row_id": int(self.row_id),
            "src_id": int(self.src_id),
            "dst_id": int(self.dst_id),
            "src_name": self.src_name,
            "dst_name": self.dst_name,
            "src_type": self.src_type,
            "dst_type": self.dst_type,
            "edge_type": self.edge_type,
            "sign": self.sign,
            "weight": float(self.weight),
            "raw_weight": float(self.raw_weight if self.raw_weight is not None else self.weight),
            "reason": self.reason,
            "source_batch": self.source_batch,
            "source_kind": self.source_kind,
            "valid_from": _display_year(self.valid_from),
            "valid_to": _display_year(self.valid_to),
            "community_id": self.community_id,
            "_scd2_retired": False,
            "_overridden_by_rivalry": bool(self.overridden_by_rivalry),
        }

    def to_export_row(self, name_resolver: Callable[[int, str], str] | None = None) -> dict[str, Any]:
        src_name = self.src_name or (name_resolver(self.src_id, self.src_type) if name_resolver is not None else "")
        dst_name = self.dst_name or (name_resolver(self.dst_id, self.dst_type) if name_resolver is not None else "")
        return {
            "src_id": int(self.src_id),
            "dst_id": int(self.dst_id),
            "src_name": src_name,
            "dst_name": dst_name,
            "src_type": self.src_type,
            "dst_type": self.dst_type,
            "edge_type": self.edge_type,
            "sign": self.sign,
            "weight": float(self.weight),
            "raw_weight": float(self.raw_weight if self.raw_weight is not None else self.weight),
            "reason": self.reason,
            "source_batch": self.source_batch,
            "source_kind": self.source_kind,
            "valid_from": _display_year(self.valid_from),
            "valid_to": _display_year(self.valid_to),
            "community_id": self.community_id,
            "_scd2_retired": False,
            "_overridden_by_rivalry": bool(self.overridden_by_rivalry),
        }


@dataclass(slots=True)
class HotEdgeStore:
    edge_type: str
    directed: bool
    sign: str
    keys: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    valid_from: np.ndarray
    valid_to: np.ndarray
    row_id: np.ndarray
    adj_nodes: np.ndarray | None = None
    adj_indptr: np.ndarray | None = None
    adj_neighbors: np.ndarray | None = None
    adj_weight: np.ndarray | None = None
    adj_valid_from: np.ndarray | None = None
    adj_valid_to: np.ndarray | None = None
    src_nodes: np.ndarray | None = None
    src_indptr: np.ndarray | None = None
    src_order: np.ndarray | None = None

    def lookup_index(self, key: int) -> int | None:
        if len(self.keys) == 0:
            return None
        idx = int(np.searchsorted(self.keys, np.uint64(key)))
        if idx < len(self.keys) and int(self.keys[idx]) == int(key):
            return idx
        return None

    def iter_neighbors(self, person_id: int) -> Iterator[tuple[int, float, int, int]]:
        if self.adj_nodes is None or self.adj_indptr is None or self.adj_neighbors is None:
            return iter(())
        node = int(person_id)
        pos = int(np.searchsorted(self.adj_nodes, np.int32(node)))
        if pos >= len(self.adj_nodes) or int(self.adj_nodes[pos]) != node:
            return iter(())
        start = int(self.adj_indptr[pos])
        end = int(self.adj_indptr[pos + 1])

        def _gen() -> Iterator[tuple[int, float, int, int]]:
            assert self.adj_neighbors is not None
            assert self.adj_weight is not None
            assert self.adj_valid_from is not None
            assert self.adj_valid_to is not None
            for idx in range(start, end):
                yield (
                    int(self.adj_neighbors[idx]),
                    float(self.adj_weight[idx]),
                    int(self.adj_valid_from[idx]),
                    int(self.adj_valid_to[idx]),
                )

        return _gen()

    def iter_from_src(self, src_id: int) -> Iterator[tuple[int, float, int, int]]:
        if self.src_nodes is None or self.src_indptr is None or self.src_order is None:
            return iter(())
        src = int(src_id)
        pos = int(np.searchsorted(self.src_nodes, np.int32(src)))
        if pos >= len(self.src_nodes) or int(self.src_nodes[pos]) != src:
            return iter(())
        start = int(self.src_indptr[pos])
        end = int(self.src_indptr[pos + 1])

        def _gen() -> Iterator[tuple[int, float, int, int]]:
            assert self.src_order is not None
            for ord_idx in range(start, end):
                row_idx = int(self.src_order[ord_idx])
                yield (
                    int(self.dst[row_idx]),
                    float(self.weight[row_idx]),
                    int(self.valid_from[row_idx]),
                    int(self.valid_to[row_idx]),
                )

        return _gen()


@dataclass(slots=True)
class ColdEdgeStore:
    keys: np.ndarray
    edge_type: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    valid_from: np.ndarray
    valid_to: np.ndarray
    row_id: np.ndarray
    sign: np.ndarray

    def lookup_index(self, key: int, edge_type: str) -> int | None:
        if len(self.keys) == 0:
            return None
        left = int(np.searchsorted(self.keys, np.uint64(key), side="left"))
        if left >= len(self.keys) or int(self.keys[left]) != int(key):
            return None
        code = TYPE_NAME_TO_CODE[edge_type]
        right = int(np.searchsorted(self.keys, np.uint64(key), side="right"))
        for idx in range(left, right):
            if int(self.edge_type[idx]) == code:
                return idx
        return None


@dataclass(slots=True)
class OverlayEdgeStore:
    edge_type: str
    directed: bool
    index_mode: str
    keys: array = field(default_factory=lambda: array("Q"))
    src: array = field(default_factory=lambda: array("I"))
    dst: array = field(default_factory=lambda: array("I"))
    weight: array = field(default_factory=lambda: array("f"))
    valid_from: array = field(default_factory=lambda: array("i"))
    valid_to: array = field(default_factory=lambda: array("i"))
    row_id: array = field(default_factory=lambda: array("q"))
    flags: array = field(default_factory=lambda: array("B"))
    slot_by_key: dict[int, int] = field(default_factory=dict)
    node_slots: dict[int, list[int]] = field(default_factory=dict)
    src_slots: dict[int, list[int]] = field(default_factory=dict)
    reason_by_slot: dict[int, str] = field(default_factory=dict)
    source_kind_by_slot: dict[int, str] = field(default_factory=dict)
    source_batch_by_slot: dict[int, str] = field(default_factory=dict)
    raw_weight_by_slot: dict[int, float] = field(default_factory=dict)
    community_id_by_slot: dict[int, int] = field(default_factory=dict)
    overridden_by_slot: dict[int, bool] = field(default_factory=dict)
    inactive_slots: int = 0

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def active_count(self) -> int:
        return len(self.slot_by_key)

    def _append_sidecars(self, slot: int, record: EdgeRecord) -> None:
        if record.reason:
            self.reason_by_slot[int(slot)] = str(record.reason)
        if record.source_kind:
            self.source_kind_by_slot[int(slot)] = str(record.source_kind)
        if record.source_batch:
            self.source_batch_by_slot[int(slot)] = str(record.source_batch)
        if record.raw_weight is not None and float(record.raw_weight) != float(record.weight):
            self.raw_weight_by_slot[int(slot)] = float(record.raw_weight)
        if record.community_id is not None:
            self.community_id_by_slot[int(slot)] = int(record.community_id)
        if record.overridden_by_rivalry:
            self.overridden_by_slot[int(slot)] = True

    def add(self, record: EdgeRecord) -> int:
        slot = len(self.keys)
        key = int(record.key)
        self.keys.append(key)
        self.src.append(int(record.src_id))
        self.dst.append(int(record.dst_id))
        self.weight.append(float(record.weight))
        self.valid_from.append(int(record.valid_from))
        self.valid_to.append(int(record.valid_to))
        self.row_id.append(int(record.row_id))
        self.flags.append(ACTIVE_FLAG)
        self.slot_by_key[key] = int(slot)
        if self.index_mode == "undirected":
            self.node_slots.setdefault(int(record.src_id), []).append(int(slot))
            self.node_slots.setdefault(int(record.dst_id), []).append(int(slot))
        elif self.index_mode == "src":
            self.src_slots.setdefault(int(record.src_id), []).append(int(slot))
        self._append_sidecars(slot, record)
        return int(slot)

    def add_many(self, records: Sequence[EdgeRecord]) -> list[int]:
        if not records:
            return []
        start = len(self.keys)
        count = len(records)
        slots = list(range(start, start + count))
        keys = np.fromiter((int(record.key) for record in records), dtype=np.uint64, count=count)
        src = np.fromiter((int(record.src_id) for record in records), dtype=np.uint32, count=count)
        dst = np.fromiter((int(record.dst_id) for record in records), dtype=np.uint32, count=count)
        weight = np.fromiter((float(record.weight) for record in records), dtype=np.float32, count=count)
        valid_from = np.fromiter((int(record.valid_from) for record in records), dtype=np.int32, count=count)
        valid_to = np.fromiter((int(record.valid_to) for record in records), dtype=np.int32, count=count)
        row_id = np.fromiter((int(record.row_id) for record in records), dtype=np.int64, count=count)
        flags = np.full(count, ACTIVE_FLAG, dtype=np.uint8)

        self.keys.frombytes(np.ascontiguousarray(keys, dtype=np.uint64).tobytes())
        self.src.frombytes(np.ascontiguousarray(src, dtype=np.uint32).tobytes())
        self.dst.frombytes(np.ascontiguousarray(dst, dtype=np.uint32).tobytes())
        self.weight.frombytes(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
        self.valid_from.frombytes(np.ascontiguousarray(valid_from, dtype=np.int32).tobytes())
        self.valid_to.frombytes(np.ascontiguousarray(valid_to, dtype=np.int32).tobytes())
        self.row_id.frombytes(np.ascontiguousarray(row_id, dtype=np.int64).tobytes())
        self.flags.frombytes(np.ascontiguousarray(flags, dtype=np.uint8).tobytes())

        for slot, record in zip(slots, records):
            key = int(record.key)
            self.slot_by_key[key] = int(slot)
            if self.index_mode == "undirected":
                self.node_slots.setdefault(int(record.src_id), []).append(int(slot))
                self.node_slots.setdefault(int(record.dst_id), []).append(int(slot))
            elif self.index_mode == "src":
                self.src_slots.setdefault(int(record.src_id), []).append(int(slot))
            self._append_sidecars(int(slot), record)
        return slots

    def is_active_slot(self, slot: int) -> bool:
        if slot < 0 or slot >= len(self.flags):
            return False
        return bool(int(self.flags[slot]) & ACTIVE_FLAG)

    def lookup_slot(self, key: int) -> int | None:
        slot = self.slot_by_key.get(int(key))
        if slot is None or not self.is_active_slot(int(slot)):
            return None
        return int(slot)

    def tombstone(self, key: int, close_year: int | None = None) -> int | None:
        slot = self.lookup_slot(key)
        if slot is None:
            return None
        self.slot_by_key.pop(int(key), None)
        self.flags[slot] = 0
        if close_year is not None:
            self.valid_to[slot] = int(close_year)
        self.inactive_slots += 1
        return int(slot)

    def _slot_reason(self, slot: int) -> str:
        return self.reason_by_slot.get(int(slot), "")

    def _slot_source_kind(self, slot: int) -> str:
        return self.source_kind_by_slot.get(int(slot), "")

    def _slot_source_batch(self, slot: int) -> str:
        return self.source_batch_by_slot.get(int(slot), "")

    def _slot_raw_weight(self, slot: int) -> float:
        return float(self.raw_weight_by_slot.get(int(slot), float(self.weight[slot])))

    def _slot_community_id(self, slot: int) -> int | None:
        return self.community_id_by_slot.get(int(slot))

    def _slot_overridden(self, slot: int) -> bool:
        return bool(self.overridden_by_slot.get(int(slot), False))

    def edge_record(self, slot: int, *, override_valid_to: int | None = None) -> EdgeRecord:
        valid_to = int(self.valid_to[slot] if override_valid_to is None else override_valid_to)
        return EdgeRecord(
            row_id=int(self.row_id[slot]),
            src_id=int(self.src[slot]),
            dst_id=int(self.dst[slot]),
            edge_type=self.edge_type,
            sign=_edge_sign(self.edge_type),
            weight=float(self.weight[slot]),
            valid_from=int(self.valid_from[slot]),
            valid_to=valid_to,
            src_type=_edge_entity_types(self.edge_type)[0],
            dst_type=_edge_entity_types(self.edge_type)[1],
            reason=self._slot_reason(slot),
            source_kind=self._slot_source_kind(slot),
            source_batch=self._slot_source_batch(slot),
            raw_weight=self._slot_raw_weight(slot),
            community_id=self._slot_community_id(slot),
            overridden_by_rivalry=self._slot_overridden(slot),
        )

    def active_payload(self, key: int, year: int | None) -> dict[str, Any] | None:
        slot = self.lookup_slot(key)
        if slot is None:
            return None
        valid_from = int(self.valid_from[slot])
        valid_to = int(self.valid_to[slot])
        if not _active_in_year(valid_from, valid_to, year):
            return None
        return {
            "weight": float(self.weight[slot]),
            "valid_from": _display_year(valid_from),
            "valid_to": _display_year(valid_to),
            "reason": self._slot_reason(slot),
            "source_kind": self._slot_source_kind(slot),
            "source_batch": self._slot_source_batch(slot),
            "community_id": self._slot_community_id(slot),
            "_overridden_by_rivalry": self._slot_overridden(slot),
        }

    def iter_neighbors(self, person_id: int, year: int | None) -> Iterator[tuple[int, float, int, int]]:
        if self.index_mode != "undirected":
            return iter(())
        pid = int(person_id)

        def _gen() -> Iterator[tuple[int, float, int, int]]:
            for slot in self.node_slots.get(pid, []):
                if not self.is_active_slot(int(slot)):
                    continue
                src = int(self.src[slot])
                dst = int(self.dst[slot])
                other = dst if src == pid else src
                valid_from = int(self.valid_from[slot])
                valid_to = int(self.valid_to[slot])
                if _active_in_year(valid_from, valid_to, year):
                    yield (int(other), float(self.weight[slot]), valid_from, valid_to)

        return _gen()

    def iter_from_src(self, src_id: int, year: int | None) -> Iterator[EdgeRecord]:
        if self.index_mode != "src":
            return iter(())
        sid = int(src_id)

        def _gen() -> Iterator[EdgeRecord]:
            for slot in self.src_slots.get(sid, []):
                if not self.is_active_slot(int(slot)):
                    continue
                valid_from = int(self.valid_from[slot])
                valid_to = int(self.valid_to[slot])
                if _active_in_year(valid_from, valid_to, year):
                    yield self.edge_record(int(slot))

        return _gen()

    def iter_node_ids(self) -> Iterator[int]:
        if self.index_mode != "undirected":
            return iter(())

        def _gen() -> Iterator[int]:
            for node, slots in self.node_slots.items():
                if any(self.is_active_slot(int(slot)) for slot in slots):
                    yield int(node)

        return _gen()

    def iter_src_ids(self) -> Iterator[int]:
        if self.index_mode != "src":
            return iter(())

        def _gen() -> Iterator[int]:
            for node, slots in self.src_slots.items():
                if any(self.is_active_slot(int(slot)) for slot in slots):
                    yield int(node)

        return _gen()

    def iter_active_export_rows(self, name_resolver: Callable[[int, str], str] | None = None) -> Iterator[dict[str, Any]]:
        for slot in self.slot_by_key.values():
            if self.is_active_slot(int(slot)):
                yield self.edge_record(int(slot)).to_export_row(name_resolver)

    def needs_compaction(self) -> bool:
        allocated = len(self.keys)
        if allocated == 0:
            return False
        if allocated >= OVERLAY_COMPACT_SIZE:
            return True
        return self.inactive_slots > 0 and (self.inactive_slots / float(allocated)) >= OVERLAY_COMPACT_RATIO

    def compact(self) -> None:
        if len(self.keys) == 0:
            self.slot_by_key.clear()
            self.node_slots.clear()
            self.src_slots.clear()
            self.reason_by_slot.clear()
            self.source_kind_by_slot.clear()
            self.source_batch_by_slot.clear()
            self.raw_weight_by_slot.clear()
            self.community_id_by_slot.clear()
            self.overridden_by_slot.clear()
            self.inactive_slots = 0
            return

        new_store = OverlayEdgeStore(
            edge_type=self.edge_type,
            directed=self.directed,
            index_mode=self.index_mode,
        )
        for slot in range(len(self.keys)):
            if not self.is_active_slot(slot):
                continue
            new_store.add(self.edge_record(slot))

        self.keys = new_store.keys
        self.src = new_store.src
        self.dst = new_store.dst
        self.weight = new_store.weight
        self.valid_from = new_store.valid_from
        self.valid_to = new_store.valid_to
        self.row_id = new_store.row_id
        self.flags = new_store.flags
        self.slot_by_key = new_store.slot_by_key
        self.node_slots = new_store.node_slots
        self.src_slots = new_store.src_slots
        self.reason_by_slot = new_store.reason_by_slot
        self.source_kind_by_slot = new_store.source_kind_by_slot
        self.source_batch_by_slot = new_store.source_batch_by_slot
        self.raw_weight_by_slot = new_store.raw_weight_by_slot
        self.community_id_by_slot = new_store.community_id_by_slot
        self.overridden_by_slot = new_store.overridden_by_slot
        self.inactive_slots = 0


def _select_bounded_positions(
    weights: np.ndarray,
    valid_mask: np.ndarray,
    limit: int,
    seed: int,
) -> np.ndarray:
    valid_idx = np.flatnonzero(valid_mask)
    if limit <= 0 or valid_idx.size == 0:
        return np.array([], dtype=np.int32)
    if valid_idx.size <= limit:
        return valid_idx.astype(np.int32, copy=False)

    top_count = max(1, min(limit, int(round(limit * 0.70))))
    valid_weights = np.asarray(weights[valid_idx], dtype=np.float32)
    if top_count >= valid_idx.size:
        top_local = np.arange(valid_idx.size, dtype=np.int32)
    else:
        top_local = np.argpartition(valid_weights, -top_count)[-top_count:].astype(np.int32, copy=False)
    top_idx = valid_idx[top_local]
    order = np.argsort(valid_weights[top_local])[::-1]
    top_idx = top_idx[order].astype(np.int32, copy=False)

    explore_count = max(0, limit - len(top_idx))
    if explore_count <= 0 or len(top_idx) >= valid_idx.size:
        return top_idx[:limit].astype(np.int32, copy=False)

    chosen_mask = np.zeros(valid_idx.size, dtype=bool)
    chosen_mask[top_local] = True
    remaining_idx = valid_idx[~chosen_mask]
    if remaining_idx.size <= 0:
        return top_idx[:limit].astype(np.int32, copy=False)

    rng = np.random.RandomState(_normalize_seed32(seed))
    take = min(explore_count, remaining_idx.size)
    explore = rng.choice(remaining_idx, size=take, replace=False)
    return np.concatenate([top_idx, np.asarray(explore, dtype=np.int32)])[:limit].astype(np.int32, copy=False)


def _overlay_index_mode(edge_type: str) -> str:
    if edge_type in HOT_UNDIRECTED_TYPES:
        return "undirected"
    if edge_type in HOT_DIRECTED_TYPES:
        return "src"
    return "none"


def _empty_hot_store(edge_type: str) -> HotEdgeStore:
    return HotEdgeStore(
        edge_type=edge_type,
        directed=edge_type not in UNDIRECTED_TYPES,
        sign=_edge_sign(edge_type),
        keys=np.array([], dtype=np.uint64),
        src=np.array([], dtype=np.uint32),
        dst=np.array([], dtype=np.uint32),
        weight=np.array([], dtype=np.float32),
        valid_from=np.array([], dtype=np.int32),
        valid_to=np.array([], dtype=np.int32),
        row_id=np.array([], dtype=np.int64),
    )


def _empty_cold_store() -> ColdEdgeStore:
    return ColdEdgeStore(
        keys=np.array([], dtype=np.uint64),
        edge_type=np.array([], dtype=np.uint8),
        src=np.array([], dtype=np.uint32),
        dst=np.array([], dtype=np.uint32),
        weight=np.array([], dtype=np.float32),
        valid_from=np.array([], dtype=np.int32),
        valid_to=np.array([], dtype=np.int32),
        row_id=np.array([], dtype=np.int64),
        sign=np.array([], dtype=np.int8),
    )


def _build_adjacency(
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    valid_from: np.ndarray,
    valid_to: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(src) == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
    node_ids = np.concatenate([src.astype(np.int32), dst.astype(np.int32)])
    neighbors = np.concatenate([dst.astype(np.int32), src.astype(np.int32)])
    weights = np.concatenate([weight.astype(np.float32), weight.astype(np.float32)])
    vf = np.concatenate([valid_from.astype(np.int32), valid_from.astype(np.int32)])
    vt = np.concatenate([valid_to.astype(np.int32), valid_to.astype(np.int32)])
    order = np.argsort(node_ids, kind="mergesort")
    node_ids = node_ids[order]
    neighbors = neighbors[order]
    weights = weights[order]
    vf = vf[order]
    vt = vt[order]
    unique_nodes, counts = np.unique(node_ids, return_counts=True)
    indptr = np.zeros(len(unique_nodes) + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts, dtype=np.int32)
    return unique_nodes, indptr, neighbors, weights, vf, vt


def _build_src_offsets(src: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(src) == 0:
        return np.array([], dtype=np.int32), np.array([0], dtype=np.int32), np.array([], dtype=np.int32)
    order = np.argsort(src.astype(np.int32), kind="mergesort")
    ordered_src = src[order].astype(np.int32)
    unique_src, counts = np.unique(ordered_src, return_counts=True)
    indptr = np.zeros(len(unique_src) + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts, dtype=np.int32)
    return unique_src, indptr, order.astype(np.int32)


class EdgeHistorySequence(Sequence[dict[str, Any]]):
    def __init__(self, graph: "GraphRuntime"):
        self.graph = graph

    def __len__(self) -> int:
        return int(self.graph.history_row_count)

    def __getitem__(self, index):
        raise TypeError("Graph history is streamed; random indexing is not supported.")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.graph.iter_temporal_history_rows()


class GraphAdjacencyView(Mapping[int, list[tuple[int, float, int | None, int | None]]]):
    def __init__(self, graph: "GraphRuntime", edge_type: str):
        self.graph = graph
        self.edge_type = edge_type

    def get(self, key: int, default=None):
        if self.edge_type == "friendship":
            rows = list(self.graph.iter_friend_neighbors(int(key), None))
        else:
            rows = list(self.graph.iter_rival_neighbors(int(key), None))
        out = [(nbr, w, _display_year(vf), _display_year(vt)) for (nbr, w, vf, vt) in rows]
        return out if out else default

    def __getitem__(self, key: int):
        rows = self.get(int(key), None)
        if rows is None:
            raise KeyError(key)
        return rows

    def __iter__(self):
        seen = set()
        store = self.graph.hot_stores.get(self.edge_type)
        if store is not None and store.adj_nodes is not None:
            for node in store.adj_nodes.tolist():
                seen.add(int(node))
                yield int(node)
        for node in self.graph._iter_overlay_nodes(self.edge_type):
            if int(node) not in seen:
                yield int(node)

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())


class PairPayloadView(Mapping[tuple[int, int], dict[str, Any]]):
    def __init__(self, graph: "GraphRuntime", edge_type: str):
        self.graph = graph
        self.edge_type = edge_type

    def get(self, key: tuple[int, int], default=None):
        if not isinstance(key, tuple) or len(key) != 2:
            return default
        payload = self.graph.get_active_payload(self.edge_type, int(key[0]), int(key[1]))
        return payload if payload is not None else default

    def __getitem__(self, key: tuple[int, int]) -> dict[str, Any]:
        payload = self.get(key, None)
        if payload is None:
            raise KeyError(key)
        return payload

    def __contains__(self, key: object) -> bool:
        return self.get(key, None) is not None  # type: ignore[arg-type]

    def __iter__(self):
        for row in self.graph.iter_active_rows_for_type(self.edge_type):
            yield _canonical_ids(row["src_id"], row["dst_id"], self.edge_type)

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())


class DirectorEdgeView(Mapping[int, list[dict[str, Any]]]):
    def __init__(self, graph: "GraphRuntime", edge_type: str):
        self.graph = graph
        self.edge_type = edge_type

    def get(self, key: int, default=None):
        if self.edge_type == "mentorship":
            rows = [
                {
                    "actor_id": actor_id,
                    "weight": weight,
                    "valid_from": _display_year(valid_from),
                    "valid_to": _display_year(valid_to),
                }
                for actor_id, weight, valid_from, valid_to in self.graph.get_director_prefs(int(key), None)
            ]
        else:
            rows = [
                {
                    "actor_id": actor_id,
                    "valid_from": _display_year(valid_from),
                    "valid_to": _display_year(valid_to),
                }
                for actor_id, valid_from, valid_to in self.graph.get_director_avoids(int(key), None)
            ]
        return rows if rows else default

    def __getitem__(self, key: int):
        rows = self.get(int(key), None)
        if rows is None:
            raise KeyError(key)
        return rows

    def __iter__(self):
        seen = set()
        store = self.graph.hot_stores.get(self.edge_type)
        if store is not None and store.src_nodes is not None:
            for node in store.src_nodes.tolist():
                seen.add(int(node))
                yield int(node)
        for node in self.graph._iter_overlay_src_nodes(self.edge_type):
            if int(node) not in seen:
                yield int(node)

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())


class GraphAffinityIndexView(Mapping[str, Any]):
    def __init__(self, graph: "GraphRuntime"):
        self.graph = graph
        self._view = {
            "friendships": PairPayloadView(graph, "friendship"),
            "rivalries": PairPayloadView(graph, "rivalry"),
            "director_prefs": DirectorEdgeView(graph, "mentorship"),
            "director_avoids": DirectorEdgeView(graph, "avoid"),
            "company_affinity": {},
            "company_rivalry": {},
            "person_company_affinity": defaultdict(list),
        }

    def __getitem__(self, key: str):
        return self._view[key]

    def __iter__(self):
        return iter(self._view)

    def __len__(self) -> int:
        return len(self._view)

    def get(self, key: str, default=None):
        return self._view.get(key, default)


class EdgeWeightsView(Mapping[tuple[int, int], float]):
    def __init__(self, graph: "GraphRuntime"):
        self.graph = graph

    def get(self, key: tuple[int, int], default=None):
        if not isinstance(key, tuple) or len(key) != 2:
            return default
        payload = self.graph.get_active_payload("friendship", int(key[0]), int(key[1]))
        if payload is None:
            payload = self.graph.get_active_payload("rivalry", int(key[0]), int(key[1]))
        if payload is None:
            return default
        return float(payload.get("weight", default if default is not None else 0.0))

    def __getitem__(self, key: tuple[int, int]):
        value = self.get(key, None)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self):
        seen = set()
        for edge_type in ("friendship", "rivalry"):
            for row in self.graph.iter_active_rows_for_type(edge_type):
                key = _canonical_ids(row["src_id"], row["dst_id"], edge_type)
                if key not in seen:
                    seen.add(key)
                    yield key

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())


class GraphRuntime:
    def __init__(self, base_dir: str | Path, manifest: Mapping[str, Any], name_resolver: Callable[[int, str], str] | None = None):
        self.base_dir = Path(base_dir)
        self.graph_dir = self.base_dir / "graph"
        self.runtime_dir = self.graph_dir / "runtime"
        self.history_dir = self.graph_dir / "history"
        self.manifest = dict(manifest)
        self.name_resolver = name_resolver
        self.history_row_count = int(self.manifest.get("history_rows", 0))
        self._next_row_id = int(self.manifest.get("next_row_id", self.history_row_count + 1))
        self.hot_stores: dict[str, HotEdgeStore] = {}
        self.cold_cp = _empty_cold_store()
        self.cold_cc = _empty_cold_store()
        self._load_runtime_arrays()
        self._overlay_stores: dict[str, OverlayEdgeStore] = {
            edge_type: OverlayEdgeStore(
                edge_type=edge_type,
                directed=edge_type not in UNDIRECTED_TYPES,
                index_mode=_overlay_index_mode(edge_type),
            )
            for edge_type in (HOT_EDGE_TYPES | COLD_CP_TYPES | COLD_CC_TYPES)
        }
        self._base_retired_by_key: dict[str, dict[int, int]] = {
            edge_type: {}
            for edge_type in (HOT_EDGE_TYPES | COLD_CP_TYPES | COLD_CC_TYPES)
        }
        self._delta_batch: list[dict[str, Any]] = []
        self._delta_batch_index: dict[int, int] = {}
        self._closure_batch: list[dict[str, Any]] = []
        self._event_batches: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._live_manifest_path = self.history_dir / "live_manifest.json"
        self._live_manifest: dict[str, Any] = {}
        self._start_live_session()
        self.edges = EdgeHistorySequence(self)
        self.affinity_index = GraphAffinityIndexView(self)
        self.edge_weights = EdgeWeightsView(self)
        self.friend_adjacency = GraphAdjacencyView(self, "friendship")
        self.rival_adjacency = GraphAdjacencyView(self, "rivalry")

    @classmethod
    def load_or_compile(
        cls,
        base_dir: str | Path,
        *,
        row_iter: Iterable[dict[str, Any]] | None = None,
        source_label: str | None = None,
        normalize_row: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        name_resolver: Callable[[int, str], str] | None = None,
    ) -> "GraphRuntime":
        base_path = Path(base_dir)
        manifest_path = base_path / "graph" / "runtime_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return cls(base_path, manifest, name_resolver=name_resolver)
        if row_iter is None:
            raise FileNotFoundError(f"No runtime manifest present in {base_path / 'graph'} and no legacy graph iterator was provided.")
        cls.compile_runtime_graph(
            base_path,
            row_iter=row_iter,
            source_label=source_label or "legacy",
            normalize_row=normalize_row,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(base_path, manifest, name_resolver=name_resolver)

    @classmethod
    def from_rows(
        cls,
        base_dir: str | Path,
        rows: Sequence[dict[str, Any]],
        *,
        name_resolver: Callable[[int, str], str] | None = None,
    ) -> "GraphRuntime":
        base_path = Path(base_dir)
        manifest_path = base_path / "graph" / "runtime_manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        cls.compile_runtime_graph(base_path, row_iter=rows, source_label="inline")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(base_path, manifest, name_resolver=name_resolver)

    @classmethod
    def compile_runtime_graph(
        cls,
        base_dir: str | Path,
        *,
        row_iter: Iterable[dict[str, Any]],
        source_label: str,
        normalize_row: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        base_path = Path(base_dir)
        graph_dir = base_path / "graph"
        runtime_dir = graph_dir / "runtime"
        history_dir = graph_dir / "history"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        history_dir.mkdir(parents=True, exist_ok=True)

        hot_cols: dict[str, dict[str, array]] = {}
        for edge_type in HOT_EDGE_TYPES:
            hot_cols[edge_type] = {
                "src": array("I"),
                "dst": array("I"),
                "weight": array("f"),
                "valid_from": array("i"),
                "valid_to": array("i"),
                "row_id": array("q"),
            }

        cp_cols = {
            "keys": array("Q"),
            "edge_type": array("B"),
            "src": array("I"),
            "dst": array("I"),
            "weight": array("f"),
            "valid_from": array("i"),
            "valid_to": array("i"),
            "row_id": array("q"),
            "sign": array("b"),
        }
        cc_cols = {
            "keys": array("Q"),
            "edge_type": array("B"),
            "src": array("I"),
            "dst": array("I"),
            "weight": array("f"),
            "valid_from": array("i"),
            "valid_to": array("i"),
            "row_id": array("q"),
            "sign": array("b"),
        }

        history_path = history_dir / "edges_initial.arrow"
        writer = _history_writer(history_path, HISTORY_SCHEMA)
        history_batch: list[dict[str, Any]] = []
        row_id = 0
        history_rows = 0

        def _flush_history() -> None:
            nonlocal history_batch
            if not history_batch:
                return
            writer.write(pa.Table.from_pylist(history_batch, schema=HISTORY_SCHEMA))
            history_batch = []

        for raw_row in row_iter:
            row = normalize_row(raw_row) if normalize_row is not None else dict(raw_row)
            edge_type = str(row.get("edge_type", "") or "")
            if edge_type not in HOT_EDGE_TYPES and edge_type not in COLD_CP_TYPES and edge_type not in COLD_CC_TYPES:
                continue
            src_id = _safe_int(row.get("src_id"))
            dst_id = _safe_int(row.get("dst_id"))
            if src_id <= 0 or dst_id <= 0:
                continue
            src_id, dst_id = _canonical_ids(src_id, dst_id, edge_type)
            sign = _edge_sign(edge_type, row.get("sign"))
            src_type, dst_type = _edge_entity_types(edge_type)
            valid_from = _norm_year(row.get("valid_from"))
            valid_to = _norm_year(row.get("valid_to"))
            retired = _norm_bool(row.get("_scd2_retired"))
            weight = _safe_float(row.get("weight"), 0.0)
            raw_weight = row.get("raw_weight")
            raw_weight = float(raw_weight) if raw_weight not in (None, "", "nan") else weight
            row_id += 1
            history_rows += 1
            history_batch.append(
                {
                    "row_id": int(row_id),
                    "src_id": int(src_id),
                    "dst_id": int(dst_id),
                    "src_name": str(row.get("src_name", "") or ""),
                    "dst_name": str(row.get("dst_name", "") or ""),
                    "src_type": src_type,
                    "dst_type": dst_type,
                    "edge_type": edge_type,
                    "sign": sign,
                    "weight": float(weight),
                    "raw_weight": float(raw_weight),
                    "reason": str(row.get("reason", "") or ""),
                    "source_batch": str(row.get("source_batch", "") or ""),
                    "source_kind": str(row.get("source_kind", "") or ""),
                    "valid_from": _display_year(valid_from),
                    "valid_to": _display_year(valid_to),
                    "community_id": (_safe_int(row.get("community_id")) if row.get("community_id") not in (None, "", "nan") else None),
                    "_scd2_retired": bool(retired),
                    "_overridden_by_rivalry": _norm_bool(row.get("_overridden_by_rivalry")),
                }
            )
            if len(history_batch) >= 50_000:
                _flush_history()

            if retired:
                continue
            if edge_type in HOT_EDGE_TYPES:
                cols = hot_cols[edge_type]
                cols["src"].append(int(src_id))
                cols["dst"].append(int(dst_id))
                cols["weight"].append(float(weight))
                cols["valid_from"].append(int(valid_from))
                cols["valid_to"].append(int(valid_to))
                cols["row_id"].append(int(row_id))
                continue

            target = cp_cols if edge_type in COLD_CP_TYPES else cc_cols
            target["keys"].append(int(pack_edge_key(src_id, dst_id, edge_type)))
            target["edge_type"].append(int(TYPE_NAME_TO_CODE[edge_type]))
            target["src"].append(int(src_id))
            target["dst"].append(int(dst_id))
            target["weight"].append(float(weight))
            target["valid_from"].append(int(valid_from))
            target["valid_to"].append(int(valid_to))
            target["row_id"].append(int(row_id))
            target["sign"].append(-1 if sign == "-" else 1)

        _flush_history()
        writer.close()

        hot_manifest: dict[str, dict[str, Any]] = {}
        hot_root = runtime_dir / "hot"
        hot_root.mkdir(parents=True, exist_ok=True)
        for edge_type, cols in hot_cols.items():
            src = np.frombuffer(cols["src"], dtype=np.uint32).copy()
            dst = np.frombuffer(cols["dst"], dtype=np.uint32).copy()
            weight = np.frombuffer(cols["weight"], dtype=np.float32).copy()
            valid_from = np.frombuffer(cols["valid_from"], dtype=np.int32).copy()
            valid_to = np.frombuffer(cols["valid_to"], dtype=np.int32).copy()
            row_ids = np.frombuffer(cols["row_id"], dtype=np.int64).copy()
            keys = np.array([pack_edge_key(int(a), int(b), edge_type) for a, b in zip(src, dst)], dtype=np.uint64)
            order = np.argsort(keys, kind="mergesort")
            src = src[order]
            dst = dst[order]
            weight = weight[order]
            valid_from = valid_from[order]
            valid_to = valid_to[order]
            row_ids = row_ids[order]
            keys = keys[order]
            type_dir = hot_root / edge_type
            type_dir.mkdir(parents=True, exist_ok=True)
            np.save(type_dir / "keys.npy", keys, allow_pickle=False)
            np.save(type_dir / "src.npy", src, allow_pickle=False)
            np.save(type_dir / "dst.npy", dst, allow_pickle=False)
            np.save(type_dir / "weight.npy", weight, allow_pickle=False)
            np.save(type_dir / "valid_from.npy", valid_from, allow_pickle=False)
            np.save(type_dir / "valid_to.npy", valid_to, allow_pickle=False)
            np.save(type_dir / "row_id.npy", row_ids, allow_pickle=False)
            meta = {"count": int(len(keys)), "directed": edge_type not in UNDIRECTED_TYPES, "sign": _edge_sign(edge_type)}
            if edge_type in {"friendship", "rivalry"}:
                adj_nodes, adj_indptr, adj_neighbors, adj_weight, adj_vf, adj_vt = _build_adjacency(src, dst, weight, valid_from, valid_to)
                np.save(type_dir / "adj_nodes.npy", adj_nodes, allow_pickle=False)
                np.save(type_dir / "adj_indptr.npy", adj_indptr, allow_pickle=False)
                np.save(type_dir / "adj_neighbors.npy", adj_neighbors, allow_pickle=False)
                np.save(type_dir / "adj_weight.npy", adj_weight, allow_pickle=False)
                np.save(type_dir / "adj_valid_from.npy", adj_vf, allow_pickle=False)
                np.save(type_dir / "adj_valid_to.npy", adj_vt, allow_pickle=False)
            if edge_type in HOT_DIRECTED_TYPES:
                src_nodes, src_indptr, src_order = _build_src_offsets(src)
                np.save(type_dir / "src_nodes.npy", src_nodes, allow_pickle=False)
                np.save(type_dir / "src_indptr.npy", src_indptr, allow_pickle=False)
                np.save(type_dir / "src_order.npy", src_order, allow_pickle=False)
            hot_manifest[edge_type] = meta

        cold_root = runtime_dir / "cold"
        cold_root.mkdir(parents=True, exist_ok=True)
        for prefix, cols in (("cp", cp_cols), ("cc", cc_cols)):
            keys = np.frombuffer(cols["keys"], dtype=np.uint64).copy()
            edge_type = np.frombuffer(cols["edge_type"], dtype=np.uint8).copy()
            src = np.frombuffer(cols["src"], dtype=np.uint32).copy()
            dst = np.frombuffer(cols["dst"], dtype=np.uint32).copy()
            weight = np.frombuffer(cols["weight"], dtype=np.float32).copy()
            valid_from = np.frombuffer(cols["valid_from"], dtype=np.int32).copy()
            valid_to = np.frombuffer(cols["valid_to"], dtype=np.int32).copy()
            row_ids = np.frombuffer(cols["row_id"], dtype=np.int64).copy()
            sign = np.frombuffer(cols["sign"], dtype=np.int8).copy()
            if len(keys) > 0:
                order = np.argsort(keys, kind="mergesort")
                keys = keys[order]
                edge_type = edge_type[order]
                src = src[order]
                dst = dst[order]
                weight = weight[order]
                valid_from = valid_from[order]
                valid_to = valid_to[order]
                row_ids = row_ids[order]
                sign = sign[order]
            np.save(cold_root / f"{prefix}_keys.npy", keys, allow_pickle=False)
            np.save(cold_root / f"{prefix}_edge_type.npy", edge_type, allow_pickle=False)
            np.save(cold_root / f"{prefix}_src.npy", src, allow_pickle=False)
            np.save(cold_root / f"{prefix}_dst.npy", dst, allow_pickle=False)
            np.save(cold_root / f"{prefix}_weight.npy", weight, allow_pickle=False)
            np.save(cold_root / f"{prefix}_valid_from.npy", valid_from, allow_pickle=False)
            np.save(cold_root / f"{prefix}_valid_to.npy", valid_to, allow_pickle=False)
            np.save(cold_root / f"{prefix}_row_id.npy", row_ids, allow_pickle=False)
            np.save(cold_root / f"{prefix}_sign.npy", sign, allow_pickle=False)

        manifest = {
            "version": RUNTIME_VERSION,
            "source": source_label,
            "history_rows": int(history_rows),
            "next_row_id": int(history_rows + 1),
            "history_path": "graph/history/edges_initial.arrow",
            "hot_types": hot_manifest,
            "cold_cp_count": int(len(cp_cols["keys"])),
            "cold_cc_count": int(len(cc_cols["keys"])),
        }
        (graph_dir / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def compile_runtime_graph_batches(
        cls,
        base_dir: str | Path,
        *,
        batch_iter: Iterable[Mapping[str, Any]],
        source_label: str,
    ) -> None:
        base_path = Path(base_dir)
        graph_dir = base_path / "graph"
        runtime_dir = graph_dir / "runtime"
        history_dir = graph_dir / "history"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        history_dir.mkdir(parents=True, exist_ok=True)

        hot_cols: dict[str, dict[str, array]] = {}
        for edge_type in HOT_EDGE_TYPES:
            hot_cols[edge_type] = {
                "src": array("I"),
                "dst": array("I"),
                "weight": array("f"),
                "valid_from": array("i"),
                "valid_to": array("i"),
                "row_id": array("q"),
            }

        cp_cols = {
            "keys": array("Q"),
            "edge_type": array("B"),
            "src": array("I"),
            "dst": array("I"),
            "weight": array("f"),
            "valid_from": array("i"),
            "valid_to": array("i"),
            "row_id": array("q"),
            "sign": array("b"),
        }
        cc_cols = {
            "keys": array("Q"),
            "edge_type": array("B"),
            "src": array("I"),
            "dst": array("I"),
            "weight": array("f"),
            "valid_from": array("i"),
            "valid_to": array("i"),
            "row_id": array("q"),
            "sign": array("b"),
        }

        history_path = history_dir / "edges_initial.arrow"
        writer = _history_writer(history_path, HISTORY_SCHEMA)
        history_rows = 0
        next_row_id = 1

        def _append_numeric(col: array, values: np.ndarray, dtype: np.dtype) -> None:
            arr = np.ascontiguousarray(values, dtype=dtype)
            if arr.size:
                col.frombytes(arr.tobytes())

        def _write_history_batch(
            *,
            row_ids: np.ndarray,
            src: np.ndarray,
            dst: np.ndarray,
            edge_type: str,
            sign: str,
            weight: np.ndarray,
            src_type: str,
            dst_type: str,
            valid_from: np.ndarray,
            valid_to: np.ndarray,
            source_kind: str,
            source_batch: str,
            reason: str,
            community_id: np.ndarray | None,
        ) -> None:
            n_rows = int(len(row_ids))
            if n_rows <= 0:
                return
            if community_id is None:
                community_arr = pa.nulls(n_rows, type=pa.int32())
            else:
                community_arr = pa.array(np.ascontiguousarray(community_id, dtype=np.int32), type=pa.int32())
            table = pa.Table.from_arrays(
                [
                    pa.array(np.ascontiguousarray(row_ids, dtype=np.int64), type=pa.int64()),
                    pa.array(np.ascontiguousarray(src, dtype=np.int64), type=pa.int64()),
                    pa.array(np.ascontiguousarray(dst, dtype=np.int64), type=pa.int64()),
                    pa.array([""] * n_rows, type=pa.string()),
                    pa.array([""] * n_rows, type=pa.string()),
                    pa.array([src_type] * n_rows, type=pa.string()),
                    pa.array([dst_type] * n_rows, type=pa.string()),
                    pa.array([edge_type] * n_rows, type=pa.string()),
                    pa.array([sign] * n_rows, type=pa.string()),
                    pa.array(np.ascontiguousarray(weight, dtype=np.float32), type=pa.float32()),
                    pa.array(np.ascontiguousarray(weight, dtype=np.float32), type=pa.float32()),
                    pa.array([reason] * n_rows, type=pa.string()),
                    pa.array([source_batch] * n_rows, type=pa.string()),
                    pa.array([source_kind] * n_rows, type=pa.string()),
                    pa.array(np.ascontiguousarray(valid_from, dtype=np.int32), type=pa.int32()),
                    pa.array(np.ascontiguousarray(valid_to, dtype=np.int32), type=pa.int32()),
                    community_arr,
                    pa.array([False] * n_rows, type=pa.bool_()),
                    pa.array([False] * n_rows, type=pa.bool_()),
                ],
                schema=HISTORY_SCHEMA,
            )
            writer.write(table)

        for raw_batch in batch_iter:
            edge_type = str(raw_batch.get("edge_type", "") or "")
            if edge_type not in HOT_EDGE_TYPES and edge_type not in COLD_CP_TYPES and edge_type not in COLD_CC_TYPES:
                continue
            src = np.asarray(raw_batch.get("src", ()), dtype=np.uint32)
            dst = np.asarray(raw_batch.get("dst", ()), dtype=np.uint32)
            if src.size == 0 or dst.size == 0:
                continue
            if src.shape != dst.shape:
                raise ValueError(f"Batch src/dst shape mismatch for edge_type={edge_type}: {src.shape} vs {dst.shape}")

            mask = (src > 0) & (dst > 0)
            if not np.any(mask):
                continue
            src = src[mask]
            dst = dst[mask]
            if edge_type in UNDIRECTED_TYPES:
                src2 = np.minimum(src, dst)
                dst2 = np.maximum(src, dst)
                src, dst = src2.astype(np.uint32, copy=False), dst2.astype(np.uint32, copy=False)

            weight = np.asarray(raw_batch.get("weight", np.zeros(len(src), dtype=np.float32)), dtype=np.float32)
            valid_from = np.asarray(raw_batch.get("valid_from", np.zeros(len(src), dtype=np.int32)), dtype=np.int32)
            valid_to = np.asarray(raw_batch.get("valid_to", np.full(len(src), INT32_MAX, dtype=np.int32)), dtype=np.int32)
            if weight.shape != src.shape or valid_from.shape != src.shape or valid_to.shape != src.shape:
                raise ValueError(f"Batch payload shape mismatch for edge_type={edge_type}")

            sign = _edge_sign(edge_type, raw_batch.get("sign"))
            src_type = str(raw_batch.get("src_type", _edge_entity_types(edge_type)[0]) or _edge_entity_types(edge_type)[0])
            dst_type = str(raw_batch.get("dst_type", _edge_entity_types(edge_type)[1]) or _edge_entity_types(edge_type)[1])
            source_kind = str(raw_batch.get("source_kind", "") or "")
            source_batch = str(raw_batch.get("source_batch", "") or "")
            reason = str(raw_batch.get("reason", "") or "")
            community_id = raw_batch.get("community_id")
            if community_id is not None:
                community_id = np.asarray(community_id, dtype=np.int32)
                if community_id.shape != src.shape:
                    raise ValueError(f"Batch community_id shape mismatch for edge_type={edge_type}")

            count = int(len(src))
            row_ids = np.arange(next_row_id, next_row_id + count, dtype=np.int64)
            next_row_id += count
            history_rows += count
            _write_history_batch(
                row_ids=row_ids,
                src=src,
                dst=dst,
                edge_type=edge_type,
                sign=sign,
                weight=weight,
                src_type=src_type,
                dst_type=dst_type,
                valid_from=valid_from,
                valid_to=valid_to,
                source_kind=source_kind,
                source_batch=source_batch,
                reason=reason,
                community_id=community_id,
            )

            if edge_type in HOT_EDGE_TYPES:
                cols = hot_cols[edge_type]
                _append_numeric(cols["src"], src, np.uint32)
                _append_numeric(cols["dst"], dst, np.uint32)
                _append_numeric(cols["weight"], weight, np.float32)
                _append_numeric(cols["valid_from"], valid_from, np.int32)
                _append_numeric(cols["valid_to"], valid_to, np.int32)
                _append_numeric(cols["row_id"], row_ids, np.int64)
                continue

            target = cp_cols if edge_type in COLD_CP_TYPES else cc_cols
            keys = np.array([pack_edge_key(int(a), int(b), edge_type) for a, b in zip(src, dst)], dtype=np.uint64)
            codes = np.full(count, TYPE_NAME_TO_CODE[edge_type], dtype=np.uint8)
            sign_arr = np.full(count, -1 if sign == "-" else 1, dtype=np.int8)
            _append_numeric(target["keys"], keys, np.uint64)
            _append_numeric(target["edge_type"], codes, np.uint8)
            _append_numeric(target["src"], src, np.uint32)
            _append_numeric(target["dst"], dst, np.uint32)
            _append_numeric(target["weight"], weight, np.float32)
            _append_numeric(target["valid_from"], valid_from, np.int32)
            _append_numeric(target["valid_to"], valid_to, np.int32)
            _append_numeric(target["row_id"], row_ids, np.int64)
            _append_numeric(target["sign"], sign_arr, np.int8)

        writer.close()

        hot_manifest: dict[str, dict[str, Any]] = {}
        hot_root = runtime_dir / "hot"
        hot_root.mkdir(parents=True, exist_ok=True)
        for edge_type, cols in hot_cols.items():
            src = np.frombuffer(cols["src"], dtype=np.uint32).copy()
            dst = np.frombuffer(cols["dst"], dtype=np.uint32).copy()
            weight = np.frombuffer(cols["weight"], dtype=np.float32).copy()
            valid_from = np.frombuffer(cols["valid_from"], dtype=np.int32).copy()
            valid_to = np.frombuffer(cols["valid_to"], dtype=np.int32).copy()
            row_ids = np.frombuffer(cols["row_id"], dtype=np.int64).copy()
            keys = np.array([pack_edge_key(int(a), int(b), edge_type) for a, b in zip(src, dst)], dtype=np.uint64)
            order = np.argsort(keys, kind="mergesort")
            src = src[order]
            dst = dst[order]
            weight = weight[order]
            valid_from = valid_from[order]
            valid_to = valid_to[order]
            row_ids = row_ids[order]
            keys = keys[order]
            type_dir = hot_root / edge_type
            type_dir.mkdir(parents=True, exist_ok=True)
            np.save(type_dir / "keys.npy", keys, allow_pickle=False)
            np.save(type_dir / "src.npy", src, allow_pickle=False)
            np.save(type_dir / "dst.npy", dst, allow_pickle=False)
            np.save(type_dir / "weight.npy", weight, allow_pickle=False)
            np.save(type_dir / "valid_from.npy", valid_from, allow_pickle=False)
            np.save(type_dir / "valid_to.npy", valid_to, allow_pickle=False)
            np.save(type_dir / "row_id.npy", row_ids, allow_pickle=False)
            meta = {"count": int(len(keys)), "directed": edge_type not in UNDIRECTED_TYPES, "sign": _edge_sign(edge_type)}
            if edge_type in {"friendship", "rivalry"}:
                adj_nodes, adj_indptr, adj_neighbors, adj_weight, adj_vf, adj_vt = _build_adjacency(src, dst, weight, valid_from, valid_to)
                np.save(type_dir / "adj_nodes.npy", adj_nodes, allow_pickle=False)
                np.save(type_dir / "adj_indptr.npy", adj_indptr, allow_pickle=False)
                np.save(type_dir / "adj_neighbors.npy", adj_neighbors, allow_pickle=False)
                np.save(type_dir / "adj_weight.npy", adj_weight, allow_pickle=False)
                np.save(type_dir / "adj_valid_from.npy", adj_vf, allow_pickle=False)
                np.save(type_dir / "adj_valid_to.npy", adj_vt, allow_pickle=False)
            if edge_type in HOT_DIRECTED_TYPES:
                src_nodes, src_indptr, src_order = _build_src_offsets(src)
                np.save(type_dir / "src_nodes.npy", src_nodes, allow_pickle=False)
                np.save(type_dir / "src_indptr.npy", src_indptr, allow_pickle=False)
                np.save(type_dir / "src_order.npy", src_order, allow_pickle=False)
            hot_manifest[edge_type] = meta

        cold_root = runtime_dir / "cold"
        cold_root.mkdir(parents=True, exist_ok=True)
        for prefix, cols in (("cp", cp_cols), ("cc", cc_cols)):
            keys = np.frombuffer(cols["keys"], dtype=np.uint64).copy()
            edge_type = np.frombuffer(cols["edge_type"], dtype=np.uint8).copy()
            src = np.frombuffer(cols["src"], dtype=np.uint32).copy()
            dst = np.frombuffer(cols["dst"], dtype=np.uint32).copy()
            weight = np.frombuffer(cols["weight"], dtype=np.float32).copy()
            valid_from = np.frombuffer(cols["valid_from"], dtype=np.int32).copy()
            valid_to = np.frombuffer(cols["valid_to"], dtype=np.int32).copy()
            row_ids = np.frombuffer(cols["row_id"], dtype=np.int64).copy()
            sign = np.frombuffer(cols["sign"], dtype=np.int8).copy()
            if len(keys) > 0:
                order = np.argsort(keys, kind="mergesort")
                keys = keys[order]
                edge_type = edge_type[order]
                src = src[order]
                dst = dst[order]
                weight = weight[order]
                valid_from = valid_from[order]
                valid_to = valid_to[order]
                row_ids = row_ids[order]
                sign = sign[order]
            np.save(cold_root / f"{prefix}_keys.npy", keys, allow_pickle=False)
            np.save(cold_root / f"{prefix}_edge_type.npy", edge_type, allow_pickle=False)
            np.save(cold_root / f"{prefix}_src.npy", src, allow_pickle=False)
            np.save(cold_root / f"{prefix}_dst.npy", dst, allow_pickle=False)
            np.save(cold_root / f"{prefix}_weight.npy", weight, allow_pickle=False)
            np.save(cold_root / f"{prefix}_valid_from.npy", valid_from, allow_pickle=False)
            np.save(cold_root / f"{prefix}_valid_to.npy", valid_to, allow_pickle=False)
            np.save(cold_root / f"{prefix}_row_id.npy", row_ids, allow_pickle=False)
            np.save(cold_root / f"{prefix}_sign.npy", sign, allow_pickle=False)

        manifest = {
            "version": RUNTIME_VERSION,
            "source": source_label,
            "history_rows": int(history_rows),
            "next_row_id": int(history_rows + 1),
            "history_path": "graph/history/edges_initial.arrow",
            "hot_types": hot_manifest,
            "cold_cp_count": int(len(cp_cols["keys"])),
            "cold_cc_count": int(len(cc_cols["keys"])),
        }
        (graph_dir / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _load_runtime_arrays(self) -> None:
        hot_root = self.runtime_dir / "hot"
        for edge_type in HOT_EDGE_TYPES:
            type_dir = hot_root / edge_type
            if not type_dir.exists():
                self.hot_stores[edge_type] = _empty_hot_store(edge_type)
                continue
            store = HotEdgeStore(
                edge_type=edge_type,
                directed=edge_type not in UNDIRECTED_TYPES,
                sign=_edge_sign(edge_type),
                keys=np.load(type_dir / "keys.npy", mmap_mode="r"),
                src=np.load(type_dir / "src.npy", mmap_mode="r"),
                dst=np.load(type_dir / "dst.npy", mmap_mode="r"),
                weight=np.load(type_dir / "weight.npy", mmap_mode="r"),
                valid_from=np.load(type_dir / "valid_from.npy", mmap_mode="r"),
                valid_to=np.load(type_dir / "valid_to.npy", mmap_mode="r"),
                row_id=np.load(type_dir / "row_id.npy", mmap_mode="r"),
            )
            if edge_type in {"friendship", "rivalry"} and (type_dir / "adj_nodes.npy").exists():
                store.adj_nodes = np.load(type_dir / "adj_nodes.npy", mmap_mode="r")
                store.adj_indptr = np.load(type_dir / "adj_indptr.npy", mmap_mode="r")
                store.adj_neighbors = np.load(type_dir / "adj_neighbors.npy", mmap_mode="r")
                store.adj_weight = np.load(type_dir / "adj_weight.npy", mmap_mode="r")
                store.adj_valid_from = np.load(type_dir / "adj_valid_from.npy", mmap_mode="r")
                store.adj_valid_to = np.load(type_dir / "adj_valid_to.npy", mmap_mode="r")
            if edge_type in HOT_DIRECTED_TYPES and (type_dir / "src_nodes.npy").exists():
                store.src_nodes = np.load(type_dir / "src_nodes.npy", mmap_mode="r")
                store.src_indptr = np.load(type_dir / "src_indptr.npy", mmap_mode="r")
                store.src_order = np.load(type_dir / "src_order.npy", mmap_mode="r")
            self.hot_stores[edge_type] = store

        cold_root = self.runtime_dir / "cold"
        if cold_root.exists():
            self.cold_cp = ColdEdgeStore(
                keys=np.load(cold_root / "cp_keys.npy", mmap_mode="r"),
                edge_type=np.load(cold_root / "cp_edge_type.npy", mmap_mode="r"),
                src=np.load(cold_root / "cp_src.npy", mmap_mode="r"),
                dst=np.load(cold_root / "cp_dst.npy", mmap_mode="r"),
                weight=np.load(cold_root / "cp_weight.npy", mmap_mode="r"),
                valid_from=np.load(cold_root / "cp_valid_from.npy", mmap_mode="r"),
                valid_to=np.load(cold_root / "cp_valid_to.npy", mmap_mode="r"),
                row_id=np.load(cold_root / "cp_row_id.npy", mmap_mode="r"),
                sign=np.load(cold_root / "cp_sign.npy", mmap_mode="r"),
            )
            self.cold_cc = ColdEdgeStore(
                keys=np.load(cold_root / "cc_keys.npy", mmap_mode="r"),
                edge_type=np.load(cold_root / "cc_edge_type.npy", mmap_mode="r"),
                src=np.load(cold_root / "cc_src.npy", mmap_mode="r"),
                dst=np.load(cold_root / "cc_dst.npy", mmap_mode="r"),
                weight=np.load(cold_root / "cc_weight.npy", mmap_mode="r"),
                valid_from=np.load(cold_root / "cc_valid_from.npy", mmap_mode="r"),
                valid_to=np.load(cold_root / "cc_valid_to.npy", mmap_mode="r"),
                row_id=np.load(cold_root / "cc_row_id.npy", mmap_mode="r"),
                sign=np.load(cold_root / "cc_sign.npy", mmap_mode="r"),
            )

    def _make_record(
        self,
        edge_type: str,
        src_id: int,
        dst_id: int,
        weight: float,
        valid_from: int,
        valid_to: int,
        *,
        row_id: int | None = None,
        sign: str | None = None,
        reason: str = "",
        source_kind: str = "",
        source_batch: str = "",
        src_name: str = "",
        dst_name: str = "",
    ) -> EdgeRecord:
        src_id, dst_id = _canonical_ids(src_id, dst_id, edge_type)
        src_type, dst_type = _edge_entity_types(edge_type)
        return EdgeRecord(
            row_id=int(self._next_row_id if row_id is None else row_id),
            src_id=int(src_id),
            dst_id=int(dst_id),
            edge_type=edge_type,
            sign=_edge_sign(edge_type, sign),
            weight=float(weight),
            valid_from=int(valid_from),
            valid_to=int(valid_to),
            src_type=src_type,
            dst_type=dst_type,
            src_name=src_name,
            dst_name=dst_name,
            reason=reason,
            source_kind=source_kind,
            source_batch=source_batch,
            raw_weight=float(weight),
        )

    def _resolve_name(self, entity_id: int, entity_type: str) -> str:
        if self.name_resolver is None:
            return ""
        return str(self.name_resolver(int(entity_id), entity_type) or "")

    def _new_live_manifest(self) -> dict[str, Any]:
        return {
            "version": 1,
            "session_id": uuid4().hex[:12],
            "delta_parts": [],
            "closure_parts": [],
            "event_parts": {},
            "next_delta_part": 1,
            "next_closure_part": 1,
            "next_event_part": {},
            "delta_rows": 0,
            "closure_rows": 0,
            "event_rows": 0,
        }

    def _write_live_manifest(self) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._live_manifest_path.write_text(json.dumps(self._live_manifest, indent=2), encoding="utf-8")

    def _start_live_session(self) -> None:
        self._live_manifest = self._new_live_manifest()
        self._write_live_manifest()

    def _history_record_from_row(self, row: Mapping[str, Any]) -> EdgeRecord | None:
        edge_type = str(row.get("edge_type", "") or "")
        if edge_type not in HOT_EDGE_TYPES and edge_type not in COLD_CP_TYPES and edge_type not in COLD_CC_TYPES:
            return None
        src_id = _safe_int(row.get("src_id"))
        dst_id = _safe_int(row.get("dst_id"))
        row_id = _safe_int(row.get("row_id"))
        if src_id <= 0 or dst_id <= 0 or row_id <= 0:
            return None
        src_id, dst_id = _canonical_ids(src_id, dst_id, edge_type)
        src_type, dst_type = _edge_entity_types(edge_type)
        raw_weight = row.get("raw_weight")
        weight = _safe_float(row.get("weight"), 0.0)
        community_raw = row.get("community_id")
        community_id = None if community_raw in (None, "", "nan") else _safe_int(community_raw)
        return EdgeRecord(
            row_id=int(row_id),
            src_id=int(src_id),
            dst_id=int(dst_id),
            edge_type=edge_type,
            sign=_edge_sign(edge_type, row.get("sign")),
            weight=float(weight),
            valid_from=int(_norm_year(row.get("valid_from"))),
            valid_to=int(_norm_year(row.get("valid_to"))),
            src_type=src_type,
            dst_type=dst_type,
            src_name=str(row.get("src_name", "") or ""),
            dst_name=str(row.get("dst_name", "") or ""),
            reason=str(row.get("reason", "") or ""),
            source_kind=str(row.get("source_kind", "") or ""),
            source_batch=str(row.get("source_batch", "") or ""),
            raw_weight=(float(raw_weight) if raw_weight not in (None, "", "nan") else float(weight)),
            community_id=community_id,
            overridden_by_rivalry=_norm_bool(row.get("_overridden_by_rivalry")),
        )

    def _reset_live_overlay_state(self) -> None:
        self._overlay_stores = {
            edge_type: OverlayEdgeStore(
                edge_type=edge_type,
                directed=edge_type not in UNDIRECTED_TYPES,
                index_mode=_overlay_index_mode(edge_type),
            )
            for edge_type in (HOT_EDGE_TYPES | COLD_CP_TYPES | COLD_CC_TYPES)
        }
        self._base_retired_by_key = {
            edge_type: {}
            for edge_type in (HOT_EDGE_TYPES | COLD_CP_TYPES | COLD_CC_TYPES)
        }
        self._delta_batch = []
        self._delta_batch_index = {}
        self._closure_batch = []
        self._event_batches = defaultdict(list)

    def _apply_base_closures(self, closures: dict[int, int]) -> None:
        remaining = {int(row_id): int(close_year) for row_id, close_year in closures.items()}
        if not remaining:
            return

        for edge_type, store in self.hot_stores.items():
            if not remaining or store is None or len(store.row_id) == 0:
                continue
            target = np.fromiter(remaining.keys(), dtype=np.int64, count=len(remaining))
            matches = np.flatnonzero(np.isin(store.row_id, target))
            for idx in matches.tolist():
                row_id = int(store.row_id[int(idx)])
                close_year = remaining.pop(row_id, None)
                if close_year is None:
                    continue
                self._base_retired_by_key[edge_type][int(store.keys[int(idx)])] = int(close_year)

        for store in (self.cold_cp, self.cold_cc):
            if not remaining or store is None or len(store.row_id) == 0:
                continue
            target = np.fromiter(remaining.keys(), dtype=np.int64, count=len(remaining))
            matches = np.flatnonzero(np.isin(store.row_id, target))
            for idx in matches.tolist():
                row_id = int(store.row_id[int(idx)])
                close_year = remaining.pop(row_id, None)
                if close_year is None:
                    continue
                edge_type = TYPE_CODE_TO_NAME[int(store.edge_type[int(idx)])]
                self._base_retired_by_key[edge_type][int(store.keys[int(idx)])] = int(close_year)

    def restore_live_history(self, manifest_path: str | Path | Mapping[str, Any]) -> None:
        if isinstance(manifest_path, Mapping):
            payload = dict(manifest_path)
        else:
            path = Path(manifest_path)
            if not path.exists():
                return
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return

        self._live_manifest = dict(payload)
        self._reset_live_overlay_state()

        delta_by_row_id: dict[int, tuple[str, int]] = {}
        records_by_type: dict[str, list[EdgeRecord]] = defaultdict(list)
        max_row_id = int(self._next_row_id) - 1
        for row in self._iter_history_rows_from_paths(self._current_history_paths("delta_parts")):
            if _norm_bool(row.get("_scd2_retired")):
                continue
            record = self._history_record_from_row(row)
            if record is None:
                continue
            records_by_type[record.edge_type].append(record)
            delta_by_row_id[int(record.row_id)] = (record.edge_type, int(record.key))
            max_row_id = max(max_row_id, int(record.row_id))

        for edge_type, records in records_by_type.items():
            if records:
                self._overlay_stores[edge_type].add_many(records)

        base_closures: dict[int, int] = {}
        for row in self._iter_history_rows_from_paths(self._current_history_paths("closure_parts")):
            row_id = _safe_int(row.get("row_id"))
            if row_id <= 0:
                continue
            close_year = int(_norm_year(row.get("valid_to")))
            delta_ref = delta_by_row_id.get(int(row_id))
            if delta_ref is not None:
                edge_type, key = delta_ref
                self._overlay_stores[edge_type].tombstone(int(key), close_year=close_year)
            else:
                base_closures[int(row_id)] = int(close_year)

        self._apply_base_closures(base_closures)
        self._next_row_id = max(int(self._next_row_id), int(max_row_id) + 1)
        self.history_row_count = max(int(self.history_row_count), int(self._next_row_id) - 1)
        self._write_live_manifest()

    def _history_rel_path(self, *parts: str) -> str:
        return str((Path("graph") / "history" / Path(*parts)).as_posix())

    def _history_abs_path(self, rel_path: str) -> Path:
        return self.base_dir / Path(rel_path)

    def _current_history_paths(self, key: str) -> list[Path]:
        paths = self._live_manifest.get(key, [])
        if not isinstance(paths, list):
            return []
        return [self._history_abs_path(str(path)) for path in paths]

    def _iter_overlay_nodes(self, edge_type: str) -> Iterator[int]:
        store = self._overlay_stores.get(edge_type)
        if store is None:
            return iter(())
        return store.iter_node_ids()

    def _iter_overlay_src_nodes(self, edge_type: str) -> Iterator[int]:
        store = self._overlay_stores.get(edge_type)
        if store is None:
            return iter(())
        return store.iter_src_ids()

    def _overlay_store(self, edge_type: str) -> OverlayEdgeStore:
        return self._overlay_stores[edge_type]

    def _key_retired(self, edge_type: str, key: int, year: int | None = None) -> bool:
        close_year = self._base_retired_by_key.get(edge_type, {}).get(int(key))
        if close_year is None:
            return False
        if year is None:
            return True
        return int(year) >= int(close_year)

    def _append_delta_row(self, row: dict[str, Any]) -> None:
        row_id = int(row["row_id"])
        self._delta_batch_index[row_id] = len(self._delta_batch)
        self._delta_batch.append(dict(row))
        if len(self._delta_batch) >= DELTA_FLUSH_THRESHOLD:
            self._flush_delta_batch()

    def _close_pending_delta_row(self, row_id: int, year: int) -> bool:
        idx = self._delta_batch_index.get(int(row_id))
        if idx is None:
            return False
        row = self._delta_batch[int(idx)]
        row["valid_to"] = int(year)
        row["_scd2_retired"] = True
        return True

    def _append_closure_row(self, row_id: int, year: int) -> None:
        self._closure_batch.append(
            {
                "row_id": int(row_id),
                "valid_to": int(year),
                "_scd2_retired": True,
            }
        )
        if len(self._closure_batch) >= CLOSURE_FLUSH_THRESHOLD:
            self._flush_closure_batch()

    def _record_event(self, year: int, event_kind: str, row: EdgeRecord, *, prior_row_id: int | None = None) -> None:
        bucket = self._event_batches[int(year)]
        bucket.append(
            {
                "event_year": int(year),
                "event_kind": event_kind,
                "row_id": int(row.row_id),
                "prior_row_id": None if prior_row_id is None else int(prior_row_id),
                "src_id": int(row.src_id),
                "dst_id": int(row.dst_id),
                "edge_type": row.edge_type,
                "weight": float(row.weight),
                "valid_from": _display_year(row.valid_from),
                "valid_to": _display_year(row.valid_to),
                "reason": row.reason,
                "source_kind": row.source_kind,
            }
        )
        if len(bucket) >= EVENT_FLUSH_THRESHOLD:
            self._flush_event_year(int(year))

    def _write_rows_part(self, path: Path, schema: pa.Schema, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        writer = _history_writer(path, schema)
        writer.write(pa.Table.from_pylist(rows, schema=schema))
        writer.close()

    def _flush_delta_batch(self) -> None:
        if not self._delta_batch:
            return
        rel_path = self._history_rel_path(
            "delta_rows",
            f"part-{self._live_manifest['session_id']}-{int(self._live_manifest['next_delta_part']):06d}.arrow",
        )
        self._write_rows_part(self._history_abs_path(rel_path), HISTORY_SCHEMA, self._delta_batch)
        self._live_manifest["delta_parts"].append(rel_path)
        self._live_manifest["next_delta_part"] = int(self._live_manifest["next_delta_part"]) + 1
        self._live_manifest["delta_rows"] = int(self._live_manifest.get("delta_rows", 0)) + len(self._delta_batch)
        self._delta_batch = []
        self._delta_batch_index = {}
        self._write_live_manifest()

    def _flush_closure_batch(self) -> None:
        if not self._closure_batch:
            return
        rel_path = self._history_rel_path(
            "closures",
            f"part-{self._live_manifest['session_id']}-{int(self._live_manifest['next_closure_part']):06d}.arrow",
        )
        self._write_rows_part(self._history_abs_path(rel_path), CLOSURE_SCHEMA, self._closure_batch)
        self._live_manifest["closure_parts"].append(rel_path)
        self._live_manifest["next_closure_part"] = int(self._live_manifest["next_closure_part"]) + 1
        self._live_manifest["closure_rows"] = int(self._live_manifest.get("closure_rows", 0)) + len(self._closure_batch)
        self._closure_batch = []
        self._write_live_manifest()

    def _flush_event_year(self, year: int) -> None:
        rows = self._event_batches.get(int(year), [])
        if not rows:
            return
        year_key = str(int(year))
        next_part = int(self._live_manifest.setdefault("next_event_part", {}).get(year_key, 1))
        rel_path = self._history_rel_path(
            "events",
            f"year={int(year)}",
            f"part-{self._live_manifest['session_id']}-{next_part:06d}.arrow",
        )
        self._write_rows_part(self._history_abs_path(rel_path), EVENT_SCHEMA, rows)
        event_parts = self._live_manifest.setdefault("event_parts", {})
        event_parts.setdefault(year_key, []).append(rel_path)
        self._live_manifest.setdefault("next_event_part", {})[year_key] = next_part + 1
        self._live_manifest["event_rows"] = int(self._live_manifest.get("event_rows", 0)) + len(rows)
        self._event_batches[int(year)] = []
        self._write_live_manifest()

    def compact_overlays(self, force: bool = False) -> None:
        for store in self._overlay_stores.values():
            if force or store.needs_compaction():
                store.compact()

    def flush_year(self, year: int) -> None:
        self._flush_delta_batch()
        self._flush_closure_batch()
        self._flush_event_year(int(year))
        self.compact_overlays()

    def flush_all(self) -> None:
        self._flush_delta_batch()
        self._flush_closure_batch()
        for year in sorted(self._event_batches):
            self._flush_event_year(int(year))
        self.compact_overlays(force=True)

    def _lookup_base_hot(self, edge_type: str, src_id: int, dst_id: int, year: int | None = None) -> tuple[HotEdgeStore, int] | None:
        store = self.hot_stores.get(edge_type)
        if store is None:
            return None
        key = pack_edge_key(src_id, dst_id, edge_type)
        if self._key_retired(edge_type, key, year):
            return None
        idx = store.lookup_index(key)
        if idx is None:
            return None
        return store, idx

    def _lookup_base_cold(self, edge_type: str, src_id: int, dst_id: int, year: int | None = None) -> tuple[ColdEdgeStore, int] | None:
        key = pack_edge_key(src_id, dst_id, edge_type)
        if self._key_retired(edge_type, key, year):
            return None
        store = self.cold_cp if edge_type in COLD_CP_TYPES else self.cold_cc
        idx = store.lookup_index(key, edge_type)
        if idx is None:
            return None
        return store, idx

    def _overlay_record(self, edge_type: str, key: int, *, override_valid_to: int | None = None) -> EdgeRecord | None:
        store = self._overlay_stores.get(edge_type)
        if store is None:
            return None
        slot = store.lookup_slot(int(key))
        if slot is None:
            return None
        return store.edge_record(int(slot), override_valid_to=override_valid_to)

    def _close_existing(self, edge_type: str, src_id: int, dst_id: int, year: int, reason: str = "") -> EdgeRecord | None:
        key = pack_edge_key(src_id, dst_id, edge_type)
        store = self._overlay_stores.get(edge_type)
        if store is not None:
            slot = store.lookup_slot(int(key))
            if slot is not None:
                record = store.edge_record(int(slot), override_valid_to=int(year))
                self._close_pending_delta_row(int(record.row_id), int(year))
                self._append_closure_row(int(record.row_id), int(year))
                store.tombstone(int(key), close_year=int(year))
                if reason:
                    record.reason = reason
                self._record_event(int(year), "expire", record, prior_row_id=record.row_id)
                return record

        if edge_type in HOT_EDGE_TYPES:
            found = self._lookup_base_hot(edge_type, src_id, dst_id)
            if found is None:
                return None
            store, idx = found
            row_id = int(store.row_id[idx])
            self._base_retired_by_key[edge_type][int(key)] = int(year)
            self._append_closure_row(row_id, int(year))
            record = self._make_record(
                edge_type,
                int(store.src[idx]),
                int(store.dst[idx]),
                float(store.weight[idx]),
                int(store.valid_from[idx]),
                int(year),
                row_id=row_id,
                sign=store.sign,
                reason=reason,
            )
            self._record_event(int(year), "expire", record, prior_row_id=row_id)
            return record

        found = self._lookup_base_cold(edge_type, src_id, dst_id)
        if found is None:
            return None
        store, idx = found
        row_id = int(store.row_id[idx])
        self._base_retired_by_key[edge_type][int(key)] = int(year)
        self._append_closure_row(row_id, int(year))
        record = self._make_record(
            edge_type,
            int(store.src[idx]),
            int(store.dst[idx]),
            float(store.weight[idx]),
            int(store.valid_from[idx]),
            int(year),
            row_id=row_id,
            sign="-" if int(store.sign[idx]) < 0 else "+",
            reason=reason,
        )
        self._record_event(int(year), "expire", record, prior_row_id=row_id)
        return record

    def get_active_payload(self, edge_type: str, src_id: int, dst_id: int, year: int | None = None) -> dict[str, Any] | None:
        key = pack_edge_key(src_id, dst_id, edge_type)
        overlay_store = self._overlay_stores.get(edge_type)
        if overlay_store is not None:
            overlay_payload = overlay_store.active_payload(int(key), year)
            if overlay_payload is not None:
                return overlay_payload

        if edge_type in HOT_EDGE_TYPES:
            found = self._lookup_base_hot(edge_type, src_id, dst_id, year)
            if found is None:
                return None
            store, idx = found
            vf = int(store.valid_from[idx])
            vt = int(store.valid_to[idx])
            if not _active_in_year(vf, vt, year):
                return None
            return {"weight": float(store.weight[idx]), "valid_from": _display_year(vf), "valid_to": _display_year(vt)}

        found = self._lookup_base_cold(edge_type, src_id, dst_id, year)
        if found is None:
            return None
        store, idx = found
        vf = int(store.valid_from[idx])
        vt = int(store.valid_to[idx])
        if not _active_in_year(vf, vt, year):
            return None
        return {"weight": float(store.weight[idx]), "valid_from": _display_year(vf), "valid_to": _display_year(vt)}

    def has_active_edge(self, edge_type: str, src_id: int, dst_id: int, year: int | None = None) -> bool:
        return self.get_active_payload(edge_type, src_id, dst_id, year) is not None

    def get_active_edge_weight(self, edge_type: str, src_id: int, dst_id: int, year: int | None = None) -> float:
        payload = self.get_active_payload(edge_type, src_id, dst_id, year)
        return float(payload.get("weight", 0.0)) if payload is not None else 0.0

    def _iter_overlay_neighbors(self, edge_type: str, person_id: int) -> Iterator[tuple[int, float, int, int]]:
        store = self._overlay_stores.get(edge_type)
        if store is None:
            return iter(())
        return store.iter_neighbors(int(person_id), None)

    def _iter_overlay_from_src(self, edge_type: str, src_id: int) -> Iterator[EdgeRecord]:
        store = self._overlay_stores.get(edge_type)
        if store is None:
            return iter(())
        return store.iter_from_src(int(src_id), None)

    def iter_friend_neighbors(self, person_id: int, year: int | None) -> Iterator[tuple[int, float, int, int]]:
        yielded: set[int] = set()
        store = self.hot_stores.get("friendship")
        if store is not None:
            for neighbor, weight, valid_from, valid_to in store.iter_neighbors(int(person_id)):
                key = pack_edge_key(person_id, neighbor, "friendship")
                if self._overlay_stores["friendship"].lookup_slot(int(key)) is not None:
                    continue
                if self._key_retired("friendship", key, year):
                    continue
                if _active_in_year(valid_from, valid_to, year):
                    yielded.add(int(neighbor))
                    yield (int(neighbor), float(weight), int(valid_from), int(valid_to))
        for neighbor, weight, valid_from, valid_to in self._overlay_stores["friendship"].iter_neighbors(int(person_id), year):
            if int(neighbor) in yielded:
                continue
            if _active_in_year(valid_from, valid_to, year):
                yield (int(neighbor), float(weight), int(valid_from), int(valid_to))

    def iter_rival_neighbors(self, person_id: int, year: int | None) -> Iterator[tuple[int, float, int, int]]:
        yielded: set[int] = set()
        store = self.hot_stores.get("rivalry")
        if store is not None:
            for neighbor, weight, valid_from, valid_to in store.iter_neighbors(int(person_id)):
                key = pack_edge_key(person_id, neighbor, "rivalry")
                if self._overlay_stores["rivalry"].lookup_slot(int(key)) is not None:
                    continue
                if self._key_retired("rivalry", key, year):
                    continue
                if _active_in_year(valid_from, valid_to, year):
                    yielded.add(int(neighbor))
                    yield (int(neighbor), float(weight), int(valid_from), int(valid_to))
        for neighbor, weight, valid_from, valid_to in self._overlay_stores["rivalry"].iter_neighbors(int(person_id), year):
            if int(neighbor) in yielded:
                continue
            if _active_in_year(valid_from, valid_to, year):
                yield (int(neighbor), float(weight), int(valid_from), int(valid_to))

    def get_director_prefs(self, director_id: int, year: int | None) -> list[tuple[int, float, int, int]]:
        out: list[tuple[int, float, int, int]] = []
        seen: set[int] = set()
        store = self.hot_stores.get("mentorship")
        if store is not None:
            for actor_id, weight, valid_from, valid_to in store.iter_from_src(int(director_id)):
                key = pack_edge_key(director_id, actor_id, "mentorship")
                if self._overlay_stores["mentorship"].lookup_slot(int(key)) is not None:
                    continue
                if self._key_retired("mentorship", key, year):
                    continue
                if _active_in_year(valid_from, valid_to, year):
                    seen.add(int(actor_id))
                    out.append((int(actor_id), float(weight), int(valid_from), int(valid_to)))
        for row in self._overlay_stores["mentorship"].iter_from_src(int(director_id), year):
            if int(row.dst_id) in seen:
                continue
            out.append((int(row.dst_id), float(row.weight), int(row.valid_from), int(row.valid_to)))
        return out

    def get_director_avoids(self, director_id: int, year: int | None) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        seen: set[int] = set()
        store = self.hot_stores.get("avoid")
        if store is not None:
            for actor_id, _weight, valid_from, valid_to in store.iter_from_src(int(director_id)):
                key = pack_edge_key(director_id, actor_id, "avoid")
                if self._overlay_stores["avoid"].lookup_slot(int(key)) is not None:
                    continue
                if self._key_retired("avoid", key, year):
                    continue
                if _active_in_year(valid_from, valid_to, year):
                    seen.add(int(actor_id))
                    out.append((int(actor_id), int(valid_from), int(valid_to)))
        for row in self._overlay_stores["avoid"].iter_from_src(int(director_id), year):
            if int(row.dst_id) in seen:
                continue
            out.append((int(row.dst_id), int(row.valid_from), int(row.valid_to)))
        return out

    def sample_bounded_neighbors(
        self,
        edge_type: str,
        person_id: int,
        year: int | None,
        *,
        limit: int,
        seed: int,
    ) -> list[tuple[int, float, int, int]]:
        if limit <= 0 or edge_type not in {"friendship", "rivalry"}:
            return []
        pid = int(person_id)
        chosen: list[tuple[int, float, int, int]] = []
        seen: set[int] = set()

        overlay_store = self._overlay_stores.get(edge_type)
        if overlay_store is not None:
            for neighbor, weight, valid_from, valid_to in overlay_store.iter_neighbors(pid, year):
                if int(neighbor) in seen or not _active_in_year(valid_from, valid_to, year):
                    continue
                seen.add(int(neighbor))
                chosen.append((int(neighbor), float(weight), int(valid_from), int(valid_to)))
                if len(chosen) >= limit:
                    return chosen

        remaining = int(limit - len(chosen))
        if remaining <= 0:
            return chosen

        store = self.hot_stores.get(edge_type)
        if store is None or store.adj_nodes is None or store.adj_indptr is None or store.adj_neighbors is None:
            return chosen
        pos = int(np.searchsorted(store.adj_nodes, np.int32(pid)))
        if pos >= len(store.adj_nodes) or int(store.adj_nodes[pos]) != pid:
            return chosen
        start = int(store.adj_indptr[pos])
        end = int(store.adj_indptr[pos + 1])
        if end <= start:
            return chosen

        neighbors = store.adj_neighbors[start:end]
        weights = store.adj_weight[start:end] if store.adj_weight is not None else np.zeros(end - start, dtype=np.float32)
        valid_from = store.adj_valid_from[start:end] if store.adj_valid_from is not None else np.zeros(end - start, dtype=np.int32)
        valid_to = store.adj_valid_to[start:end] if store.adj_valid_to is not None else np.full(end - start, INT32_MAX, dtype=np.int32)

        overlay_lookup = self._overlay_stores[edge_type]
        active_mask = np.fromiter(
            (
                int(neighbor) not in seen
                and overlay_lookup.lookup_slot(int(pack_edge_key(pid, int(neighbor), edge_type))) is None
                and not self._key_retired(edge_type, int(pack_edge_key(pid, int(neighbor), edge_type)), year)
                and _active_in_year(int(valid_from[idx]), int(valid_to[idx]), year)
                for idx, neighbor in enumerate(neighbors)
            ),
            dtype=bool,
            count=len(neighbors),
        )
        positions = _select_bounded_positions(np.asarray(weights, dtype=np.float32), active_mask, remaining, seed)
        for idx in positions:
            neighbor = int(neighbors[int(idx)])
            if neighbor in seen:
                continue
            chosen.append(
                (
                    neighbor,
                    float(weights[int(idx)]),
                    int(valid_from[int(idx)]),
                    int(valid_to[int(idx)]),
                )
            )
            seen.add(neighbor)
            if len(chosen) >= limit:
                break
        return chosen

    def add_edge(
        self,
        edge_type: str,
        src_id: int,
        dst_id: int,
        weight: float,
        valid_from: int,
        valid_to: int | None = None,
        sign: str | None = None,
        reason: str = "",
        source_kind: str = "",
        source_batch: str = "",
    ) -> bool:
        if self.has_active_edge(edge_type, src_id, dst_id, valid_from):
            return False
        valid_to_norm = INT32_MAX if valid_to is None else max(int(valid_from), int(valid_to))
        record = self._make_record(
            edge_type,
            src_id,
            dst_id,
            weight,
            valid_from,
            valid_to_norm,
            sign=sign,
            reason=reason,
            source_kind=source_kind,
            source_batch=source_batch,
        )
        self._next_row_id += 1
        self.history_row_count += 1
        self._append_delta_row(record.to_history_row())
        self._overlay_stores[edge_type].add(record)
        self._record_event(int(valid_from), "add", record)
        return True

    def update_edge(
        self,
        edge_type: str,
        src_id: int,
        dst_id: int,
        from_year: int,
        *,
        weight: float | None = None,
        delta_weight: float | None = None,
        reason: str = "",
    ) -> bool:
        payload = self.get_active_payload(edge_type, src_id, dst_id, None)
        if payload is None:
            return False
        prior = self._close_existing(edge_type, src_id, dst_id, from_year, reason=reason)
        new_weight = float(payload.get("weight", 0.0))
        if weight is not None:
            new_weight = float(weight)
        if delta_weight is not None:
            new_weight = float(np.clip(new_weight + float(delta_weight), 0.0, 1.0))
        record = self._make_record(
            edge_type,
            src_id,
            dst_id,
            new_weight,
            from_year,
            INT32_MAX,
            sign=_edge_sign(edge_type),
            reason=reason,
            source_kind="temporal_update",
        )
        self._next_row_id += 1
        self.history_row_count += 1
        self._append_delta_row(record.to_history_row())
        self._overlay_stores[edge_type].add(record)
        self._record_event(int(from_year), "update", record, prior_row_id=None if prior is None else prior.row_id)
        return True

    def expire_edge(self, edge_type: str, src_id: int, dst_id: int, year: int, reason: str = "") -> bool:
        prior = self._close_existing(edge_type, src_id, dst_id, year, reason=reason)
        return prior is not None

    def apply_edge_batch(
        self,
        ops: Sequence[Mapping[str, Any]],
        *,
        default_from_year: int | None = None,
    ) -> tuple[int, int, int, list[str]]:
        if not ops:
            return 0, 0, 0, []
        first_mode = str(ops[0].get("mode", ""))
        first_type = str(ops[0].get("edge_type", ""))
        if first_mode and first_type and all(
            str(op.get("mode", "")) == first_mode and str(op.get("edge_type", "")) == first_type
            for op in ops
        ):
            return self._apply_uniform_edge_batch(first_mode, first_type, ops, default_from_year=default_from_year)
        applied = 0
        skipped = 0
        errors = 0
        messages: list[str] = []
        for op in ops:
            try:
                mode = str(op.get("mode", ""))
                if mode == "add":
                    ok = self.add_edge(
                        str(op.get("edge_type", "")),
                        int(op.get("src_id", 0)),
                        int(op.get("dst_id", 0)),
                        float(op.get("weight", 0.0)),
                        int(op.get("valid_from", default_from_year or 0)),
                        valid_to=(op.get("valid_to") if op.get("valid_to") not in (None, "", "nan") else None),
                        reason=str(op.get("reason", "") or ""),
                        source_kind=str(op.get("source_kind", "") or ""),
                    )
                elif mode == "update":
                    ok = self.update_edge(
                        str(op.get("edge_type", "")),
                        int(op.get("src_id", 0)),
                        int(op.get("dst_id", 0)),
                        int(default_from_year if default_from_year is not None else op.get("year", 0)),
                        delta_weight=float(op.get("delta_weight", 0.0)),
                        reason=str(op.get("reason", "") or ""),
                    )
                else:
                    ok = self.expire_edge(
                        str(op.get("edge_type", "")),
                        int(op.get("src_id", 0)),
                        int(op.get("dst_id", 0)),
                        int(op.get("year", default_from_year or 0)),
                        reason=str(op.get("reason", "") or ""),
                    )
                applied += int(bool(ok))
                skipped += int(not ok)
            except Exception as exc:
                errors += 1
                messages.append(f"edge_op failed: {exc}")
        return applied, skipped, errors, messages

    def _apply_uniform_edge_batch(
        self,
        mode: str,
        edge_type: str,
        ops: Sequence[Mapping[str, Any]],
        *,
        default_from_year: int | None = None,
    ) -> tuple[int, int, int, list[str]]:
        applied = 0
        skipped = 0
        errors = 0
        messages: list[str] = []
        overlay_store = self._overlay_stores.get(edge_type)
        if overlay_store is None:
            return 0, len(ops), 0, [f"edge_op failed: unknown edge_type={edge_type}"]

        if mode == "add":
            queued: list[EdgeRecord] = []
            seen_keys: set[int] = set()
            for op in ops:
                try:
                    src_id, dst_id = _canonical_ids(int(op.get("src_id", 0)), int(op.get("dst_id", 0)), edge_type)
                    if src_id <= 0 or dst_id <= 0:
                        skipped += 1
                        continue
                    key = int(pack_edge_key(src_id, dst_id, edge_type))
                    if key in seen_keys or self.get_active_payload(edge_type, src_id, dst_id, int(op.get("valid_from", default_from_year or 0))) is not None:
                        skipped += 1
                        continue
                    seen_keys.add(key)
                    valid_from = int(op.get("valid_from", default_from_year or 0))
                    valid_to_raw = op.get("valid_to")
                    valid_to = None if valid_to_raw in (None, "", "nan") else int(valid_to_raw)
                    queued.append(
                        self._make_record(
                            edge_type,
                            src_id,
                            dst_id,
                            float(op.get("weight", 0.0)),
                            valid_from,
                            INT32_MAX if valid_to is None else max(valid_from, valid_to),
                            row_id=int(self._next_row_id + len(queued)),
                            sign=op.get("sign"),
                            reason=str(op.get("reason", "") or ""),
                            source_kind=str(op.get("source_kind", "") or ""),
                            source_batch=str(op.get("source_batch", "") or ""),
                        )
                    )
                except Exception as exc:
                    errors += 1
                    messages.append(f"edge_op failed: {exc}")
            if queued:
                self._next_row_id += len(queued)
                self.history_row_count += len(queued)
                for record in queued:
                    self._append_delta_row(record.to_history_row())
                overlay_store.add_many(queued)
                for record in queued:
                    self._record_event(int(record.valid_from), "add", record)
                applied += len(queued)
            return applied, skipped, errors, messages

        if mode == "update":
            queued: list[tuple[EdgeRecord, int | None]] = []
            for op in ops:
                try:
                    src_id, dst_id = _canonical_ids(int(op.get("src_id", 0)), int(op.get("dst_id", 0)), edge_type)
                    if src_id <= 0 or dst_id <= 0:
                        skipped += 1
                        continue
                    payload = self.get_active_payload(edge_type, src_id, dst_id, None)
                    if payload is None:
                        skipped += 1
                        continue
                    from_year = int(default_from_year if default_from_year is not None else op.get("year", 0))
                    prior = self._close_existing(edge_type, src_id, dst_id, from_year, reason=str(op.get("reason", "") or ""))
                    new_weight = float(payload.get("weight", 0.0))
                    if op.get("weight") not in (None, "", "nan"):
                        new_weight = float(op.get("weight", new_weight))
                    if op.get("delta_weight") not in (None, "", "nan"):
                        new_weight = float(np.clip(new_weight + float(op.get("delta_weight", 0.0)), 0.0, 1.0))
                    queued.append(
                        (
                            self._make_record(
                                edge_type,
                                src_id,
                                dst_id,
                                new_weight,
                                from_year,
                                INT32_MAX,
                                row_id=int(self._next_row_id + len(queued)),
                                sign=_edge_sign(edge_type),
                                reason=str(op.get("reason", "") or ""),
                                source_kind="temporal_update",
                            ),
                            None if prior is None else int(prior.row_id),
                        )
                    )
                except Exception as exc:
                    errors += 1
                    messages.append(f"edge_op failed: {exc}")
            if queued:
                records = [record for record, _prior_row_id in queued]
                self._next_row_id += len(records)
                self.history_row_count += len(records)
                for record in records:
                    self._append_delta_row(record.to_history_row())
                overlay_store.add_many(records)
                for record, prior_row_id in queued:
                    self._record_event(int(record.valid_from), "update", record, prior_row_id=prior_row_id)
                applied += len(records)
            return applied, skipped, errors, messages

        for op in ops:
            try:
                src_id, dst_id = _canonical_ids(int(op.get("src_id", 0)), int(op.get("dst_id", 0)), edge_type)
                ok = self.expire_edge(
                    edge_type,
                    src_id,
                    dst_id,
                    int(op.get("year", default_from_year or 0)),
                    reason=str(op.get("reason", "") or ""),
                )
                applied += int(bool(ok))
                skipped += int(not ok)
            except Exception as exc:
                errors += 1
                messages.append(f"edge_op failed: {exc}")
        return applied, skipped, errors, messages

    def _hot_row_at(self, edge_type: str, idx: int) -> dict[str, Any]:
        store = self.hot_stores[edge_type]
        valid_from = int(store.valid_from[idx])
        valid_to = int(store.valid_to[idx])
        return {
            "src_id": int(store.src[idx]),
            "dst_id": int(store.dst[idx]),
            "edge_type": edge_type,
            "weight": float(store.weight[idx]),
            "sign": store.sign,
            "valid_from": _display_year(valid_from),
            "valid_to": _display_year(valid_to),
        }

    def _cold_row_at(self, store: ColdEdgeStore, idx: int, edge_type: str) -> dict[str, Any]:
        valid_from = int(store.valid_from[idx])
        valid_to = int(store.valid_to[idx])
        return {
            "src_id": int(store.src[idx]),
            "dst_id": int(store.dst[idx]),
            "edge_type": edge_type,
            "weight": float(store.weight[idx]),
            "sign": "-" if int(store.sign[idx]) < 0 else "+",
            "valid_from": _display_year(valid_from),
            "valid_to": _display_year(valid_to),
        }

    def iter_active_rows_for_type(self, edge_type: str, year: int | None = None) -> Iterator[dict[str, Any]]:
        if edge_type in HOT_EDGE_TYPES:
            store = self.hot_stores.get(edge_type)
            if store is not None:
                for idx in range(len(store.keys)):
                    key = int(store.keys[idx])
                    if self._key_retired(edge_type, key, year):
                        continue
                    if self._overlay_stores[edge_type].lookup_slot(int(key)) is not None:
                        continue
                    vf = int(store.valid_from[idx])
                    vt = int(store.valid_to[idx])
                    if not _active_in_year(vf, vt, year):
                        continue
                    yield {
                        "src_id": int(store.src[idx]),
                        "dst_id": int(store.dst[idx]),
                        "edge_type": edge_type,
                        "sign": store.sign,
                        "weight": float(store.weight[idx]),
                        "valid_from": _display_year(vf),
                        "valid_to": _display_year(vt),
                    }
        else:
            store = self.cold_cp if edge_type in COLD_CP_TYPES else self.cold_cc
            code = TYPE_NAME_TO_CODE[edge_type]
            for idx in range(len(store.keys)):
                if int(store.edge_type[idx]) != code:
                    continue
                key = int(store.keys[idx])
                if self._key_retired(edge_type, key, year):
                    continue
                if self._overlay_stores[edge_type].lookup_slot(int(key)) is not None:
                    continue
                vf = int(store.valid_from[idx])
                vt = int(store.valid_to[idx])
                if not _active_in_year(vf, vt, year):
                    continue
                yield {
                    "src_id": int(store.src[idx]),
                    "dst_id": int(store.dst[idx]),
                    "edge_type": edge_type,
                    "sign": "-" if int(store.sign[idx]) < 0 else "+",
                    "weight": float(store.weight[idx]),
                    "valid_from": _display_year(vf),
                    "valid_to": _display_year(vt),
                }
        for row in self._overlay_stores[edge_type].iter_active_export_rows(self.name_resolver):
            if _active_in_year(_norm_year(row.get("valid_from")), _norm_year(row.get("valid_to")), year):
                yield row

    def _iter_history_rows_from_paths(self, paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
        for path in paths:
            if not path.exists():
                continue
            reader = ipc.open_file(str(path))
            for batch_idx in range(reader.num_record_batches):
                yield from reader.get_batch(batch_idx).to_pylist()

    def _closure_map(self) -> dict[int, dict[str, Any]]:
        closures: dict[int, dict[str, Any]] = {}
        for row in self._iter_history_rows_from_paths(self._current_history_paths("closure_parts")):
            row_id = int(row.get("row_id", 0))
            if row_id <= 0:
                continue
            closures[row_id] = {
                "valid_to": _display_year(_norm_year(row.get("valid_to"))),
                "_scd2_retired": bool(row.get("_scd2_retired", True)),
            }
        for row in self._closure_batch:
            row_id = int(row.get("row_id", 0))
            if row_id <= 0:
                continue
            closures[row_id] = {
                "valid_to": _display_year(_norm_year(row.get("valid_to"))),
                "_scd2_retired": bool(row.get("_scd2_retired", True)),
            }
        return closures

    def iter_temporal_history_rows(self) -> Iterator[dict[str, Any]]:
        closures = self._closure_map()
        history_paths = [self.history_dir / "edges_initial.arrow", *self._current_history_paths("delta_parts")]
        for row in self._iter_history_rows_from_paths(history_paths):
            row_id = int(row.get("row_id", 0))
            override = closures.get(row_id)
            if override:
                item = dict(row)
                item.update(override)
                yield item
            else:
                yield dict(row)
        for row in self._delta_batch:
            row_id = int(row.get("row_id", 0))
            override = closures.get(row_id)
            if override:
                item = dict(row)
                item.update(override)
                yield item
            else:
                yield dict(row)

    def sample_active_edges(self, edge_types: Sequence[str], year: int, sample_size: int, seed: int) -> list[dict[str, Any]]:
        if sample_size <= 0:
            return []
        normalized = [str(edge_type) for edge_type in edge_types if str(edge_type) in (HOT_EDGE_TYPES | COLD_CP_TYPES | COLD_CC_TYPES)]
        if not normalized:
            return []
        rng = np.random.RandomState(_normalize_seed32(seed))
        families = list(dict.fromkeys(normalized))
        per_family = max(1, int(np.ceil(float(sample_size) / float(len(families)))))
        rows: list[dict[str, Any]] = []

        for family_idx, edge_type in enumerate(families):
            if len(rows) >= sample_size:
                break
            target = min(sample_size - len(rows), per_family if family_idx < len(families) - 1 else sample_size)
            overlay_store = self._overlay_stores.get(edge_type)
            if overlay_store is not None and overlay_store.active_count > 0 and target > 0:
                overlay_slots = list(overlay_store.slot_by_key.values())
                overlay_rows: list[dict[str, Any]] = []
                for slot in overlay_slots:
                    valid_from = int(overlay_store.valid_from[int(slot)])
                    valid_to = int(overlay_store.valid_to[int(slot)])
                    if not _active_in_year(valid_from, valid_to, year):
                        continue
                    overlay_rows.append(
                        {
                            "src_id": int(overlay_store.src[int(slot)]),
                            "dst_id": int(overlay_store.dst[int(slot)]),
                            "edge_type": edge_type,
                            "weight": float(overlay_store.weight[int(slot)]),
                            "sign": _edge_sign(edge_type),
                            "valid_from": _display_year(valid_from),
                            "valid_to": _display_year(valid_to),
                        }
                    )
                if overlay_rows:
                    if len(overlay_rows) > target:
                        chosen = rng.choice(len(overlay_rows), size=target, replace=False)
                        overlay_rows = [overlay_rows[int(idx)] for idx in np.atleast_1d(chosen).tolist()]
                    rows.extend(overlay_rows)
                    target = min(sample_size - len(rows), target - len(overlay_rows))
                    if len(rows) >= sample_size or target <= 0:
                        continue
            if edge_type in HOT_EDGE_TYPES:
                store = self.hot_stores.get(edge_type)
                if store is None or len(store.keys) == 0:
                    continue
                attempts = 0
                max_attempts = max(64, target * 20)
                while len(rows) < sample_size and target > 0 and attempts < max_attempts:
                    batch_size = min(max(8, target * 4), len(store.keys))
                    indices = rng.choice(len(store.keys), size=batch_size, replace=batch_size > len(store.keys))
                    for idx in np.atleast_1d(indices).tolist():
                        idx = int(idx)
                        key = int(store.keys[idx])
                        if self._key_retired(edge_type, key, year):
                            continue
                        if self._overlay_stores[edge_type].lookup_slot(key) is not None:
                            continue
                        valid_from = int(store.valid_from[idx])
                        valid_to = int(store.valid_to[idx])
                        if not _active_in_year(valid_from, valid_to, year):
                            continue
                        rows.append(self._hot_row_at(edge_type, idx))
                        target -= 1
                        if len(rows) >= sample_size or target <= 0:
                            break
                    attempts += len(np.atleast_1d(indices))
                continue

            store = self.cold_cp if edge_type in COLD_CP_TYPES else self.cold_cc
            if len(store.keys) == 0:
                continue
            code = int(TYPE_NAME_TO_CODE[edge_type])
            attempts = 0
            max_attempts = max(64, target * 40)
            while len(rows) < sample_size and target > 0 and attempts < max_attempts:
                batch_size = min(max(16, target * 6), len(store.keys))
                indices = rng.choice(len(store.keys), size=batch_size, replace=batch_size > len(store.keys))
                for idx in np.atleast_1d(indices).tolist():
                    idx = int(idx)
                    if int(store.edge_type[idx]) != code:
                        continue
                    key = int(store.keys[idx])
                    if self._key_retired(edge_type, key, year):
                        continue
                    if self._overlay_stores[edge_type].lookup_slot(key) is not None:
                        continue
                    valid_from = int(store.valid_from[idx])
                    valid_to = int(store.valid_to[idx])
                    if not _active_in_year(valid_from, valid_to, year):
                        continue
                    rows.append(self._cold_row_at(store, idx, edge_type))
                    target -= 1
                    if len(rows) >= sample_size or target <= 0:
                        break
                attempts += len(np.atleast_1d(indices))
        if len(rows) <= sample_size:
            return rows
        chosen = rng.choice(len(rows), size=sample_size, replace=False)
        return [rows[int(idx)] for idx in np.atleast_1d(chosen).tolist()]

    def sample_hot_edges(self, year: int, sample_size: int, seed: int) -> list[dict[str, Any]]:
        return self.sample_active_edges(sorted(HOT_EDGE_TYPES), year, sample_size, seed)

    def _sample_cold_edges(self, year: int, sample_size: int, seed: int) -> list[dict[str, Any]]:
        return self.sample_active_edges(sorted(COLD_CP_TYPES | COLD_CC_TYPES), year, sample_size, seed)

    def sample_frontier(
        self,
        anchors: Sequence[int],
        edge_types: Sequence[str],
        year: int,
        per_anchor_limit: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        if per_anchor_limit <= 0:
            return []
        normalized = [str(edge_type) for edge_type in edge_types if str(edge_type) in HOT_EDGE_TYPES]
        if not normalized:
            return []
        rng = np.random.RandomState(_normalize_seed32(seed))
        rows: list[dict[str, Any]] = []
        for anchor in dict.fromkeys(int(anchor) for anchor in anchors if int(anchor) > 0):
            local: list[dict[str, Any]] = []
            for edge_type in normalized:
                if edge_type in HOT_UNDIRECTED_TYPES:
                    store = self.hot_stores.get(edge_type)
                    if store is not None:
                        for neighbor, weight, valid_from, valid_to in store.iter_neighbors(anchor):
                            if not _active_in_year(valid_from, valid_to, year):
                                continue
                            local.append(
                                {
                                    "src_id": int(anchor),
                                    "dst_id": int(neighbor),
                                    "edge_type": edge_type,
                                    "weight": float(weight),
                                    "sign": _edge_sign(edge_type),
                                    "valid_from": _display_year(valid_from),
                                    "valid_to": _display_year(valid_to),
                                }
                            )
                    for neighbor, weight, valid_from, valid_to in self._overlay_stores[edge_type].iter_neighbors(anchor, year):
                        local.append(
                            {
                                "src_id": int(anchor),
                                "dst_id": int(neighbor),
                                "edge_type": edge_type,
                                "weight": float(weight),
                                "sign": _edge_sign(edge_type),
                                "valid_from": _display_year(valid_from),
                                "valid_to": _display_year(valid_to),
                            }
                        )
                elif edge_type == "mentorship":
                    for neighbor, weight, valid_from, valid_to in self.get_director_prefs(anchor, year):
                        local.append(
                            {
                                "src_id": int(anchor),
                                "dst_id": int(neighbor),
                                "edge_type": edge_type,
                                "weight": float(weight),
                                "sign": "+",
                                "valid_from": _display_year(valid_from),
                                "valid_to": _display_year(valid_to),
                            }
                        )
                elif edge_type == "avoid":
                    for neighbor, valid_from, valid_to in self.get_director_avoids(anchor, year):
                        local.append(
                            {
                                "src_id": int(anchor),
                                "dst_id": int(neighbor),
                                "edge_type": edge_type,
                                "weight": 0.9,
                                "sign": "-",
                                "valid_from": _display_year(valid_from),
                                "valid_to": _display_year(valid_to),
                            }
                        )
            if len(local) > per_anchor_limit:
                chosen = rng.choice(len(local), size=per_anchor_limit, replace=False)
                local = [local[int(idx)] for idx in np.atleast_1d(chosen).tolist()]
            rows.extend(local)
        return rows

    def sample_company_pairs(
        self,
        companies: Sequence[int],
        edge_types: Sequence[str],
        year: int,
        sample_size: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        if sample_size <= 0:
            return []
        company_list = [int(cid) for cid in dict.fromkeys(companies) if int(cid) > 0]
        if len(company_list) < 2:
            return []
        normalized = [str(edge_type) for edge_type in edge_types if str(edge_type) in COLD_CC_TYPES]
        if not normalized:
            return []
        rng = np.random.RandomState(_normalize_seed32(seed))
        rows: list[dict[str, Any]] = []
        attempts = 0
        max_attempts = max(64, sample_size * 20)
        while len(rows) < sample_size and attempts < max_attempts:
            left = int(company_list[int(rng.randint(0, len(company_list)))])
            right = int(company_list[int(rng.randint(0, len(company_list)))])
            if left == right:
                attempts += 1
                continue
            edge_type = str(normalized[int(rng.randint(0, len(normalized)))])
            payload = self.get_active_payload(edge_type, left, right, year)
            if payload is None:
                attempts += 1
                continue
            src_id, dst_id = _canonical_ids(left, right, edge_type)
            rows.append(
                {
                    "src_id": int(src_id),
                    "dst_id": int(dst_id),
                    "edge_type": edge_type,
                    "weight": float(payload.get("weight", 0.0) or 0.0),
                    "sign": _edge_sign(edge_type),
                    "valid_from": payload.get("valid_from"),
                    "valid_to": payload.get("valid_to"),
                }
            )
            attempts += 1
        return rows

    def sample_edges_for_entities(self, people: Sequence[int], companies: Sequence[int], year: int, limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[tuple[int, int, str]] = set()
        people_set = set(int(x) for x in people)
        companies_set = set(int(x) for x in companies)

        for pid in sorted(people_set):
            for neighbor, weight, valid_from, valid_to in self.iter_friend_neighbors(pid, year):
                if neighbor not in people_set:
                    continue
                a, b = _canonical_ids(pid, neighbor, "friendship")
                key = (int(a), int(b), "friendship")
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"src_id": int(a), "dst_id": int(b), "edge_type": "friendship", "weight": float(weight), "sign": "+", "valid_from": _display_year(valid_from), "valid_to": _display_year(valid_to)})
                if len(rows) >= limit:
                    return rows
            for neighbor, weight, valid_from, valid_to in self.iter_rival_neighbors(pid, year):
                if neighbor not in people_set:
                    continue
                a, b = _canonical_ids(pid, neighbor, "rivalry")
                key = (int(a), int(b), "rivalry")
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"src_id": int(a), "dst_id": int(b), "edge_type": "rivalry", "weight": float(weight), "sign": "-", "valid_from": _display_year(valid_from), "valid_to": _display_year(valid_to)})
                if len(rows) >= limit:
                    return rows

        for director_id in sorted(people_set):
            for actor_id, weight, valid_from, valid_to in self.get_director_prefs(director_id, year):
                if actor_id not in people_set:
                    continue
                key = (int(director_id), int(actor_id), "mentorship")
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"src_id": int(director_id), "dst_id": int(actor_id), "edge_type": "mentorship", "weight": float(weight), "sign": "+", "valid_from": _display_year(valid_from), "valid_to": _display_year(valid_to)})
                if len(rows) >= limit:
                    return rows
            for actor_id, valid_from, valid_to in self.get_director_avoids(director_id, year):
                if actor_id not in people_set:
                    continue
                key = (int(director_id), int(actor_id), "avoid")
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"src_id": int(director_id), "dst_id": int(actor_id), "edge_type": "avoid", "weight": 0.9, "sign": "-", "valid_from": _display_year(valid_from), "valid_to": _display_year(valid_to)})
                if len(rows) >= limit:
                    return rows

        if not companies_set:
            return rows

        for company_id in sorted(companies_set):
            for person_id in sorted(people_set):
                for edge_type in ("brand_fit", "employment", "blacklist", "exclusive_deal"):
                    payload = self.get_active_payload(edge_type, company_id, person_id, year)
                    if payload is None:
                        continue
                    key = (int(company_id), int(person_id), edge_type)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({"src_id": int(company_id), "dst_id": int(person_id), "edge_type": edge_type, "weight": float(payload.get("weight", 0.0)), "sign": _edge_sign(edge_type), "valid_from": payload.get("valid_from"), "valid_to": payload.get("valid_to")})
                    if len(rows) >= limit:
                        return rows

        company_list = sorted(companies_set)
        for idx, company_id in enumerate(company_list):
            for other_id in company_list[idx + 1 :]:
                for edge_type in ("co_production", "market_rival", "subsidiary"):
                    payload = self.get_active_payload(edge_type, company_id, other_id, year)
                    if payload is None:
                        continue
                    a, b = _canonical_ids(company_id, other_id, edge_type)
                    key = (int(a), int(b), edge_type)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({"src_id": int(a), "dst_id": int(b), "edge_type": edge_type, "weight": float(payload.get("weight", 0.0)), "sign": _edge_sign(edge_type), "valid_from": payload.get("valid_from"), "valid_to": payload.get("valid_to")})
                    if len(rows) >= limit:
                        return rows
        return rows

    def export_temporal_history(self, out_path: str | Path) -> int:
        self.flush_all()
        target = Path(out_path)
        export_schema = HISTORY_SCHEMA.remove(0)
        writer = _history_writer(target, export_schema)
        rows: list[dict[str, Any]] = []
        total = 0
        for row in self.iter_temporal_history_rows():
            row = dict(row)
            row.pop("row_id", None)
            rows.append(row)
            if len(rows) >= 50_000:
                writer.write(pa.Table.from_pylist(rows, schema=export_schema))
                total += len(rows)
                rows = []
        if rows:
            writer.write(pa.Table.from_pylist(rows, schema=export_schema))
            total += len(rows)
        writer.close()
        return total

    def export_final_active(self, out_path: str | Path, year: int) -> int:
        target = Path(out_path)
        export_schema = HISTORY_SCHEMA.remove(0)
        writer = _history_writer(target, export_schema)
        rows: list[dict[str, Any]] = []
        total = 0
        for edge_type in HOT_EDGE_TYPES | COLD_CP_TYPES | COLD_CC_TYPES:
            for row in self.iter_active_rows_for_type(edge_type, year):
                src_type, dst_type = _edge_entity_types(edge_type)
                rows.append(
                    {
                        "src_id": int(row["src_id"]),
                        "dst_id": int(row["dst_id"]),
                        "src_name": self._resolve_name(int(row["src_id"]), src_type),
                        "dst_name": self._resolve_name(int(row["dst_id"]), dst_type),
                        "src_type": src_type,
                        "dst_type": dst_type,
                        "edge_type": edge_type,
                        "sign": row.get("sign", _edge_sign(edge_type)),
                        "weight": float(row.get("weight", 0.0)),
                        "raw_weight": float(row.get("raw_weight", row.get("weight", 0.0))),
                        "reason": row.get("reason", ""),
                        "source_batch": row.get("source_batch", ""),
                        "source_kind": row.get("source_kind", ""),
                        "valid_from": row.get("valid_from"),
                        "valid_to": row.get("valid_to"),
                        "community_id": row.get("community_id"),
                        "_scd2_retired": False,
                        "_overridden_by_rivalry": bool(row.get("_overridden_by_rivalry", False)),
                    }
                )
                if len(rows) >= 50_000:
                    writer.write(pa.Table.from_pylist(rows, schema=export_schema))
                    total += len(rows)
                    rows = []
        if rows:
            writer.write(pa.Table.from_pylist(rows, schema=export_schema))
            total += len(rows)
        writer.close()
        return total

    def materialize_legacy_csv(self, out_path: str | Path, year: int) -> int:
        import csv

        rows_written = 0
        fields = [
            "src_id",
            "dst_id",
            "src_name",
            "dst_name",
            "src_type",
            "dst_type",
            "edge_type",
            "sign",
            "weight",
            "raw_weight",
            "reason",
            "source_batch",
            "source_kind",
            "valid_from",
            "valid_to",
            "community_id",
            "_scd2_retired",
            "_overridden_by_rivalry",
        ]
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for edge_type in HOT_EDGE_TYPES | COLD_CP_TYPES | COLD_CC_TYPES:
                for row in self.iter_active_rows_for_type(edge_type, year):
                    src_type, dst_type = _edge_entity_types(edge_type)
                    writer.writerow(
                        {
                            "src_id": int(row["src_id"]),
                            "dst_id": int(row["dst_id"]),
                            "src_name": self._resolve_name(int(row["src_id"]), src_type),
                            "dst_name": self._resolve_name(int(row["dst_id"]), dst_type),
                            "src_type": src_type,
                            "dst_type": dst_type,
                            "edge_type": edge_type,
                            "sign": row.get("sign", _edge_sign(edge_type)),
                            "weight": float(row.get("weight", 0.0)),
                            "raw_weight": float(row.get("raw_weight", row.get("weight", 0.0))),
                            "reason": row.get("reason", ""),
                            "source_batch": row.get("source_batch", ""),
                            "source_kind": row.get("source_kind", ""),
                            "valid_from": row.get("valid_from"),
                            "valid_to": row.get("valid_to"),
                            "community_id": row.get("community_id"),
                            "_scd2_retired": False,
                            "_overridden_by_rivalry": bool(row.get("_overridden_by_rivalry", False)),
                        }
                    )
                    rows_written += 1
        return rows_written

    def build_affinity_index(self) -> Mapping[str, Any]:
        return self.affinity_index

    def index_add_edge(self, index: MutableMapping[str, Any] | None, edge: Mapping[str, Any]) -> None:
        self.add_edge(
            str(edge.get("edge_type", "") or ""),
            _safe_int(edge.get("src_id")),
            _safe_int(edge.get("dst_id")),
            _safe_float(edge.get("weight"), 0.0),
            _safe_int(edge.get("valid_from"), 0),
            sign=str(edge.get("sign", "") or ""),
            reason=str(edge.get("reason", "") or ""),
            source_kind=str(edge.get("source_kind", "") or ""),
            source_batch=str(edge.get("source_batch", "") or ""),
        )

    def index_expire_edge(self, index: MutableMapping[str, Any] | None, src_id: int, dst_id: int, edge_type: str) -> None:
        self.expire_edge(edge_type, src_id, dst_id, INT32_MAX - 1)

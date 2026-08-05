from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from engine.aeiou import Unit, event_vector
from engine.vector_math import ZERO, add, scale


JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}
DEFAULT_RECORD_KEYS = ("frames", "records", "events", "items", "data")
DEFAULT_IGNORED_KEYS = {
    "frame_id",
    "frame_index",
    "timestamp",
    "timestamp_ms",
    "time",
    "id",
    "index",
}


@dataclass(frozen=True)
class JsonDocument:
    raw_text: str
    records: list[Any]
    container_key: str | None


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def read_json_document(path: Path) -> JsonDocument:
    raw = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()

    if suffix in {".jsonl", ".ndjson"}:
        records: list[Any] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL 第 {line_number} 行无法解析：{exc}"
                ) from exc
        return JsonDocument(raw_text=raw, records=records, container_key=None)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 无法解析：{exc}") from exc

    if isinstance(payload, list):
        return JsonDocument(raw_text=raw, records=payload, container_key=None)

    if isinstance(payload, dict):
        for key in DEFAULT_RECORD_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return JsonDocument(
                    raw_text=raw,
                    records=candidate,
                    container_key=key,
                )
        return JsonDocument(raw_text=raw, records=[payload], container_key=None)

    return JsonDocument(raw_text=raw, records=[payload], container_key=None)


def _bounded_number(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    # 不假定字段量纲。仅将任意实数确定性压到 (-1, 1)。
    return value / (1.0 + abs(value))


def _iter_leaf_values(
    value: Any,
    *,
    path: str,
    ignored_keys: set[str],
) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            key_text = str(key)
            if key_text in ignored_keys:
                continue
            next_path = f"{path}.{key_text}" if path else key_text
            yield from _iter_leaf_values(
                value[key],
                path=next_path,
                ignored_keys=ignored_keys,
            )
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            next_path = f"{path}[{index}]"
            yield from _iter_leaf_values(
                item,
                path=next_path,
                ignored_keys=ignored_keys,
            )
        return

    yield path or "$", value


def json_record_vector(
    record: Any,
    *,
    ignored_keys: set[str],
) -> tuple[tuple[float, float, float, float, float], int]:
    total = ZERO
    event_count = 0

    for path, value in _iter_leaf_values(
        record,
        path="",
        ignored_keys=ignored_keys,
    ):
        if value is None:
            total = add(total, event_vector(f"json-null:{path}"))
            event_count += 1
            continue

        if isinstance(value, bool):
            direction = event_vector(f"json-field:{path}")
            total = add(total, scale(direction, 1.0 if value else -1.0))
            event_count += 1
            continue

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            direction = event_vector(f"json-field:{path}")
            total = add(total, scale(direction, _bounded_number(float(value))))
            event_count += 1
            continue

        total = add(total, event_vector(f"json-value:{path}={value}"))
        event_count += 1

    if event_count == 0:
        total = event_vector("json-empty-record")
        event_count = 1
    return total, event_count


def make_json_units(
    records: list[Any],
    *,
    ignored_keys: Iterable[str] = DEFAULT_IGNORED_KEYS,
) -> list[Unit]:
    ignored = {str(item) for item in ignored_keys}
    units: list[Unit] = []
    for index, record in enumerate(records):
        text = _canonical(record)
        vector, count = json_record_vector(record, ignored_keys=ignored)
        units.append(
            Unit(
                index=index,
                text=text,
                vector=vector,
                event_count=count,
            )
        )
    return units

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import re

from engine.vector_math import (
    Vector,
    ZERO,
    add,
    norm,
    scale,
    sub,
)


CHANNELS = "AEIOU"


@dataclass(frozen=True)
class Unit:
    index: int
    text: str
    vector: Vector
    event_count: int


@dataclass(frozen=True)
class HierarchyItem:
    index: int
    rank: int
    deviation: float | None


def split_original_units(text: str, fallback_chars: int) -> list[str]:
    """
    先只在原文已有强句末和换行处分开。
    超长无标点单位再使用原文已有弱标点，最后才按原字符切开。
    """
    pieces: list[str] = []
    start = 0
    for match in re.finditer(r"(?:\r?\n)+|[。！？!?]+", text):
        end = match.end()
        part = text[start:end]
        if part.strip():
            pieces.append(part)
        elif pieces:
            pieces[-1] += part
        start = end

    tail = text[start:]
    if tail.strip():
        pieces.append(tail)
    elif tail and pieces:
        pieces[-1] += tail

    expanded: list[str] = []
    for piece in pieces:
        if len(piece) <= fallback_chars:
            expanded.append(piece)
            continue

        weak_parts: list[str] = []
        weak_start = 0
        for match in re.finditer(r"[；;：:,，]+|\s{2,}", piece):
            weak_end = match.end()
            weak = piece[weak_start:weak_end]
            if weak:
                weak_parts.append(weak)
            weak_start = weak_end
        if weak_start < len(piece):
            weak_parts.append(piece[weak_start:])

        buffer = ""
        for weak in weak_parts:
            if buffer and len(buffer) + len(weak) > fallback_chars:
                expanded.append(buffer)
                buffer = ""
            if len(weak) <= fallback_chars:
                buffer += weak
                continue
            if buffer:
                expanded.append(buffer)
                buffer = ""
            for index in range(0, len(weak), fallback_chars):
                expanded.append(weak[index:index + fallback_chars])
        if buffer:
            expanded.append(buffer)

    return [piece for piece in expanded if piece.strip()]


def event_vector(event: str) -> Vector:
    digest = hashlib.sha256(
        event.encode("utf-8", errors="surrogatepass")
    ).digest()
    values = []
    for index in range(5):
        integer = int.from_bytes(
            digest[index * 4:(index + 1) * 4],
            "little",
        )
        values.append(integer / 2**32 * 2.0 - 1.0)

    length = norm(tuple(values))  # type: ignore[arg-type]
    if length == 0:
        return ZERO
    return tuple(value / length for value in values)  # type: ignore[return-value]


def text_vector(text: str, ngram_min: int, ngram_max: int) -> tuple[Vector, int]:
    total = ZERO
    event_count = 0
    for width in range(ngram_min, ngram_max + 1):
        limit = len(text) - width + 1
        for index in range(max(0, limit)):
            total = add(total, event_vector(text[index:index + width]))
            event_count += 1
    return total, event_count


def make_units(
    text: str,
    *,
    ngram_min: int,
    ngram_max: int,
    fallback_chars: int,
) -> list[Unit]:
    pieces = split_original_units(text, fallback_chars)
    units: list[Unit] = []
    for index, piece in enumerate(pieces):
        vector, count = text_vector(piece, ngram_min, ngram_max)
        units.append(Unit(index=index, text=piece, vector=vector, event_count=count))
    return units


def global_trajectory_hierarchy(units: list[Unit]) -> list[HierarchyItem]:
    count = len(units)
    if count == 0:
        return []
    if count == 1:
        return [HierarchyItem(index=0, rank=1, deviation=None)]

    path: list[Vector] = []
    state = ZERO
    for unit in units:
        state = add(state, unit.vector)
        path.append(state)

    result = [
        HierarchyItem(index=0, rank=1, deviation=None),
        HierarchyItem(index=count - 1, rank=2, deviation=None),
    ]
    seen = {0, count - 1}
    heap: list[tuple[float, int, int, int]] = []

    def push(left: int, right: int) -> None:
        if right <= left + 1:
            return
        best_index = -1
        best_distance = -1.0
        for index in range(left + 1, right):
            ratio = (index - left) / (right - left)
            expected = add(
                path[left],
                scale(sub(path[right], path[left]), ratio),
            )
            distance = norm(sub(path[index], expected))
            if distance > best_distance:
                best_distance = distance
                best_index = index
        heapq.heappush(
            heap,
            (-best_distance, left, right, best_index),
        )

    push(0, count - 1)
    while heap:
        negative, left, right, chosen = heapq.heappop(heap)
        if chosen in seen:
            continue
        seen.add(chosen)
        result.append(
            HierarchyItem(
                index=chosen,
                rank=len(result) + 1,
                deviation=-negative,
            )
        )
        push(left, chosen)
        push(chosen, right)
    return result

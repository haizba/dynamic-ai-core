from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from engine.aeiou import (
    Unit,
    global_trajectory_hierarchy,
    make_units,
)
from engine.json_adapter import make_json_units
from engine.vector_math import (
    Vector,
    ZERO,
    add,
    divide,
    norm,
    scale,
    sub,
    sum_vectors,
)


@dataclass
class Block:
    units: list[Unit]


@dataclass
class CycleRecord:
    phase: str
    step: int
    blocks_before: int
    blocks_after: int
    output_characters: int
    merged_pair_start: int | None
    distortion: float | None


@dataclass
class CompressionResult:
    text: str
    units: list[Unit]
    selected_indices: list[int]
    cycle_records: list[CycleRecord]
    cycles_count: int
    fell_below_minimum: bool


def representative(block: Block) -> Unit:
    units = block.units
    if len(units) == 1:
        return units[0]

    total_vector = sum_vectors(unit.vector for unit in units)
    lengths = [max(1, len(unit.text)) for unit in units]
    total_length = sum(lengths)

    if len(units) > 2:
        cumulative = ZERO
        cumulative_length = 0
        best_boundary = 0
        best_distance = -1.0
        for boundary in range(len(units) - 1):
            cumulative = add(cumulative, units[boundary].vector)
            cumulative_length += lengths[boundary]
            progress = cumulative_length / total_length
            expected = scale(total_vector, progress)
            distance = norm(sub(cumulative, expected))
            if distance > best_distance:
                best_distance = distance
                best_boundary = boundary
        return units[best_boundary + 1]

    target_per_character = divide(total_vector, total_length)
    best = min(
        units,
        key=lambda unit: (
            norm(
                sub(
                    divide(unit.vector, max(1, len(unit.text))),
                    target_per_character,
                )
            ),
            unit.index,
        ),
    )
    return best


def render_representatives(blocks: list[Block]) -> tuple[str, list[int]]:
    representatives = [representative(block) for block in blocks]
    unique: dict[int, Unit] = {}
    for unit in representatives:
        unique[unit.index] = unit
    ordered = [unique[index] for index in sorted(unique)]
    text = "\n\n".join(unit.text.strip() for unit in ordered if unit.text.strip())
    return text, [unit.index for unit in ordered]


def pairwise_merge(blocks: list[Block]) -> list[Block]:
    merged: list[Block] = []
    for index in range(0, len(blocks), 2):
        if index + 1 < len(blocks):
            merged.append(
                Block(blocks[index].units + blocks[index + 1].units)
            )
        else:
            merged.append(blocks[index])
    return merged


def merge_distortion(left: Block, right: Block) -> float:
    left_rep = representative(left)
    right_rep = representative(right)
    merged_rep = representative(Block(left.units + right.units))

    separate = divide(
        add(left_rep.vector, right_rep.vector),
        max(1, len(left_rep.text) + len(right_rep.text)),
    )
    merged_value = divide(
        merged_rep.vector,
        max(1, len(merged_rep.text)),
    )
    return norm(sub(separate, merged_value))


def replace_pair(blocks: list[Block], index: int) -> list[Block]:
    merged = Block(blocks[index].units + blocks[index + 1].units)
    return blocks[:index] + [merged] + blocks[index + 2:]


def add_until_minimum(
    *,
    units: list[Unit],
    selected_indices: list[int],
    minimum: int,
    maximum: int,
) -> tuple[str, list[int]]:
    selected = set(selected_indices)
    hierarchy = global_trajectory_hierarchy(units)

    def render() -> str:
        return "\n\n".join(
            units[index].text.strip()
            for index in sorted(selected)
            if units[index].text.strip()
        )

    text = render()
    if len(text) >= minimum:
        return text, sorted(selected)

    for item in hierarchy:
        if item.index in selected:
            continue
        candidate = set(selected)
        candidate.add(item.index)
        candidate_text = "\n\n".join(
            units[index].text.strip()
            for index in sorted(candidate)
            if units[index].text.strip()
        )
        if len(candidate_text) <= maximum:
            selected = candidate
            text = candidate_text
        if len(text) >= minimum:
            break
    return text, sorted(selected)


def _compress_units(
    *,
    units: list[Unit],
    source_characters: int,
    config: dict,
) -> CompressionResult:
    minimum = int(config["min_output_chars"])
    maximum = int(config["max_output_chars"])
    if minimum <= 0 or maximum < minimum:
        raise ValueError("输出字符范围无效")
    if not units:
        raise ValueError("没有读到可压缩的单位")

    if source_characters <= maximum:
        text = "\n\n".join(unit.text.strip() for unit in units if unit.text.strip())
        return CompressionResult(
            text=text,
            units=units,
            selected_indices=[unit.index for unit in units],
            cycle_records=[],
            cycles_count=0,
            fell_below_minimum=len(text) < minimum,
        )

    blocks = [Block([unit]) for unit in units]
    records: list[CycleRecord] = []
    step = 0

    current_text, current_selected = render_representatives(blocks)

    # 完整同步循环：所有相邻块同一轮两两合并。
    while len(current_text) > maximum and len(blocks) > 1:
        next_blocks = pairwise_merge(blocks)
        next_text, _ = render_representatives(next_blocks)

        # 下一整轮已经跨到最低范围以下，转为逐对细化。
        if len(next_text) < minimum:
            break

        step += 1
        records.append(
            CycleRecord(
                phase="同步循环",
                step=step,
                blocks_before=len(blocks),
                blocks_after=len(next_blocks),
                output_characters=len(next_text),
                merged_pair_start=None,
                distortion=None,
            )
        )
        blocks = next_blocks
        current_text = next_text
        if len(current_text) <= maximum:
            break

    # 逐对细化：每次只合并对代表轨迹改变最小的一对。
    fine_step = 0
    while len(current_text) > maximum and len(blocks) > 1:
        options: list[tuple[bool, float, float, int, list[Block], str]] = []
        for index in range(len(blocks) - 1):
            candidate_blocks = replace_pair(blocks, index)
            candidate_text, _ = render_representatives(candidate_blocks)
            distortion = merge_distortion(blocks[index], blocks[index + 1])

            below_minimum = len(candidate_text) < minimum
            distance_to_middle = abs(
                len(candidate_text)
                - int(config["preferred_output_chars"])
            )
            options.append(
                (
                    below_minimum,
                    distortion,
                    distance_to_middle,
                    index,
                    candidate_blocks,
                    candidate_text,
                )
            )

        chosen = min(
            options,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
            ),
        )
        _, distortion, _, index, blocks, current_text = chosen
        fine_step += 1
        records.append(
            CycleRecord(
                phase="逐对细化",
                step=fine_step,
                blocks_before=len(blocks) + 1,
                blocks_after=len(blocks),
                output_characters=len(current_text),
                merged_pair_start=index,
                distortion=distortion,
            )
        )

    current_text, current_selected = render_representatives(blocks)
    current_text, current_selected = add_until_minimum(
        units=units,
        selected_indices=current_selected,
        minimum=minimum,
        maximum=maximum,
    )

    return CompressionResult(
        text=current_text,
        units=units,
        selected_indices=current_selected,
        cycle_records=records,
        cycles_count=len(records),
        fell_below_minimum=len(current_text) < minimum,
    )


def compress_document(text: str, config: dict) -> CompressionResult:
    units = make_units(
        text,
        ngram_min=int(config["ngram_min"]),
        ngram_max=int(config["ngram_max"]),
        fallback_chars=int(config["long_unit_fallback_chars"]),
    )
    return _compress_units(
        units=units,
        source_characters=len(text),
        config=config,
    )


def compress_json_records(records: list[object], config: dict) -> CompressionResult:
    ignored = config.get(
        "json_ignored_keys",
        ["frame_id", "frame_index", "timestamp", "timestamp_ms", "time", "id", "index"],
    )
    units = make_json_units(records, ignored_keys=ignored)
    source_characters = sum(len(unit.text) + 1 for unit in units)
    return _compress_units(
        units=units,
        source_characters=source_characters,
        config=config,
    )

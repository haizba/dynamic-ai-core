from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from engine.compressor import CompressionResult
from engine.docx_io import write_docx


CHANNELS = "AEIOU"


def safe_stem(path: Path) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", path.stem)


def write_all_outputs(
    *,
    source_path: Path,
    source_relative: Path,
    source_text: str,
    result: CompressionResult,
    output_dir: Path,
    config: dict,
    elapsed_seconds: float,
    source_format: str = "txt",
    source_records: int | None = None,
) -> dict:
    stem = safe_stem(source_path)
    txt_path = output_dir / f"{stem}_AEIOU压缩.txt"
    docx_path = output_dir / f"{stem}_AEIOU压缩.docx"
    units_path = output_dir / f"{stem}_单位轨迹.csv"
    cycles_path = output_dir / f"{stem}_压缩循环.csv"
    summary_path = output_dir / f"{stem}_运行摘要.json"
    method_path = output_dir / f"{stem}_算法说明.txt"

    txt_path.write_text(result.text, encoding="utf-8")
    if source_format == "json":
        (output_dir / f"{stem}_AEIOU选择记录.jsonl").write_text(
            result.text,
            encoding="utf-8",
        )

    unit_label = "JSON记录" if source_format == "json" else "原文单位"
    subtitle = (
        f"源文档：{source_relative}｜"
        f"{len(source_text)} → {len(result.text)} 字符｜"
        + ("原始记录抽取，不改字段值" if source_format == "json" else "原文抽取，不改写")
    )
    write_docx(
        docx_path,
        title=str(config["output_title"]),
        subtitle=subtitle,
        body_text=result.text,
    )

    selected = set(result.selected_indices)
    with units_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "unit_index",
                "selected",
                "characters",
                "event_count",
                *CHANNELS,
                "original_record_or_text",
            ]
        )
        for unit in result.units:
            writer.writerow(
                [
                    unit.index + 1,
                    1 if unit.index in selected else 0,
                    len(unit.text),
                    unit.event_count,
                    *[f"{value:.12g}" for value in unit.vector],
                    unit.text,
                ]
            )

    with cycles_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "phase",
                "step",
                "blocks_before",
                "blocks_after",
                "output_characters",
                "merged_pair_start",
                "distortion",
            ]
        )
        for record in result.cycle_records:
            writer.writerow(
                [
                    record.phase,
                    record.step,
                    record.blocks_before,
                    record.blocks_after,
                    record.output_characters,
                    "" if record.merged_pair_start is None else record.merged_pair_start,
                    "" if record.distortion is None else f"{record.distortion:.12g}",
                ]
            )

    summary = {
        "source_file": str(source_relative),
        "source_format": source_format,
        "source_characters": len(source_text),
        "source_records": source_records,
        "source_units": len(result.units),
        "output_characters": len(result.text),
        "selected_units": [index + 1 for index in result.selected_indices],
        "cycle_records": result.cycles_count,
        "fell_below_minimum": result.fell_below_minimum,
        "elapsed_seconds": elapsed_seconds,
        "language_model_used": False,
        "training_used": False,
        "network_used": False,
        "third_party_packages_used": False,
        "semantic_threshold_used": False,
        "json_adapter_used": source_format == "json",
        "json_values_rewritten": False,
        "json_record_serialization_canonicalized": source_format == "json",
        "output_is_original_text_only": source_format != "json",
        "output_is_original_records_only": source_format == "json",
        "config": config,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if source_format == "json":
        method = f"""AEIOU 动态轨迹压缩：JSON 本次运行说明
============================================================

源数据
- 文件：{source_relative}
- 原始字符：{len(source_text)}
- JSON记录：{source_records}
- 进入轨迹的记录：{len(result.units)}

结果
- 输出字符：{len(result.text)}
- 最终保留记录：{len(result.selected_indices)}
- 循环记录：{result.cycles_count}
- 运行时间：{elapsed_seconds:.6f} 秒

JSON 适配规则
1. 顶层数组直接作为记录序列。
2. 顶层对象若含 frames、records、events、items 或 data 数组，自动读取该数组。
3. 每条记录仍是一个完整原始 JSON 记录；输出不修改字段和值。
4. frame_id、timestamp_ms 等标识字段默认不参与五维向量，但仍完整保留在输出记录中。
5. 每个数值字段名通过 SHA-256 得到固定五维方向，字段值仅以确定性的有界数值参与载荷。
6. 字符串、布尔值和空值同样通过固定事件进入五维载荷。
7. A、E、I、O、U 不赋予固定语义，不训练、不联网、不设内容阈值。
8. 后续同步循环、逐对细化和轨迹补入规则沿用 AEIOU v2.0 核心架构。

本次没有使用
- 语言模型
- 神经模型训练
- 网络连接
- 视觉识别模型
- 目标检测器
- 人脸识别
- 关键词表
- 主题标签
- 语义相似度阈值
- top-k / top-p
- 人工通道权重

工程边界
- 输出范围：{config["min_output_chars"]}～{config["max_output_chars"]} 字符
- 希望靠近：{config["preferred_output_chars"]} 字符
- 默认忽略的标识字段：{config.get("json_ignored_keys", [])}

重要边界
这次实验处理的是“已经提取成 JSON 的视频帧特征序列”。
它不直接解码视频像素，也不等于物体、人物或动作识别。
"""
    else:
        method = f"""AEIOU 动态轨迹压缩：本次运行说明
============================================================

源文档
- 文件：{source_relative}
- 原文字符：{len(source_text)}
- {unit_label}：{len(result.units)}

结果
- 输出字符：{len(result.text)}
- 最终原文单位：{len(result.selected_indices)}
- 循环记录：{result.cycles_count}
- 运行时间：{elapsed_seconds:.6f} 秒

本次没有使用
- 语言模型
- 神经模型训练
- 网络连接
- 关键词表
- 主题标签
- 相似度达到多少才保留
- top-k / top-p
- 人工通道权重
- 句子改写
"""
    method_path.write_text(method, encoding="utf-8")
    return summary

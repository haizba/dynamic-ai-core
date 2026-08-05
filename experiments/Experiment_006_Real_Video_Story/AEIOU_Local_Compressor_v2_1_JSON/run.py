from __future__ import annotations

import json
import time
from pathlib import Path

from engine.compressor import compress_document, compress_json_records
from engine.json_adapter import JSON_SUFFIXES, read_json_document
from engine.reader import read_document, scan_documents
from engine.report import write_all_outputs


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
CONFIG_PATH = ROOT / "config.json"


def load_config() -> dict:
    defaults = {
        "min_output_chars": 1000,
        "max_output_chars": 2000,
        "preferred_output_chars": 1800,
        "ngram_min": 2,
        "ngram_max": 5,
        "long_unit_fallback_chars": 800,
        "include_subfolders": True,
        "output_title": "AEIOU 动态轨迹压缩",
        "json_ignored_keys": [
            "frame_id",
            "frame_index",
            "timestamp",
            "timestamp_ms",
            "time",
            "id",
            "index",
        ],
    }
    if not CONFIG_PATH.exists():
        return defaults
    loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    defaults.update(loaded)
    return defaults


def main() -> None:
    cfg = load_config()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = scan_documents(
        INPUT_DIR,
        include_subfolders=bool(cfg["include_subfolders"]),
    )
    if not paths:
        print("没有找到 .txt、.docx、.json、.jsonl 或 .ndjson。")
        print(f"请把文档放进：{INPUT_DIR}")
        return

    print("=" * 64)
    print("AEIOU 本地动态压缩 v2.1 JSON")
    print("无语言模型｜无训练｜无联网｜记录不改写")
    print("=" * 64)

    completed: list[dict] = []
    errors: list[dict] = []
    batch_started = time.perf_counter()

    for number, path in enumerate(paths, start=1):
        relative = path.relative_to(INPUT_DIR)
        print(f"\n[{number}/{len(paths)}] {relative}")
        started = time.perf_counter()
        try:
            suffix = path.suffix.lower()
            if suffix in JSON_SUFFIXES:
                json_doc = read_json_document(path)
                source_text = json_doc.raw_text
                source_records = len(json_doc.records)
                result = compress_json_records(json_doc.records, cfg)
                source_format = "json"
            else:
                source_text = read_document(path)
                source_records = None
                result = compress_document(source_text, cfg)
                source_format = suffix.lstrip(".")

            elapsed = time.perf_counter() - started
            summary = write_all_outputs(
                source_path=path,
                source_relative=relative,
                source_text=source_text,
                result=result,
                output_dir=OUTPUT_DIR,
                config=cfg,
                elapsed_seconds=elapsed,
                source_format=source_format,
                source_records=source_records,
            )
            completed.append(summary)
            unit_word = "记录" if source_format == "json" else "原文单位"
            print(
                f"完成：{len(source_text)} 字符，{len(result.units)} 个{unit_word} → "
                f"{len(result.text)} 字符，{result.cycles_count} 个循环记录，"
                f"{elapsed:.2f} 秒"
            )
        except Exception as exc:
            error = {
                "file": str(relative),
                "type": type(exc).__name__,
                "message": str(exc),
            }
            errors.append(error)
            print(f"失败：{error['type']}: {error['message']}")

    batch = {
        "completed": completed,
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - batch_started,
    }
    (OUTPUT_DIR / "本次批量运行总账.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if errors:
        (OUTPUT_DIR / "未完成文件.txt").write_text(
            "\n".join(
                f"{item['file']}: {item['type']}: {item['message']}"
                for item in errors
            ),
            encoding="utf-8",
        )

    print("\n" + "=" * 64)
    print(f"完成：{len(completed)} 个；失败：{len(errors)} 个")
    print(f"结果位置：{OUTPUT_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    main()

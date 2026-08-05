from __future__ import annotations

from pathlib import Path

from engine.docx_io import read_docx
from engine.json_adapter import JSON_SUFFIXES


SUPPORTED = {".txt", ".docx", *JSON_SUFFIXES}


def decode_txt(path: Path) -> str:
    data = path.read_bytes()
    for encoding in (
        "utf-8",
        "utf-8-sig",
        "gb18030",
        "utf-16",
        "big5",
    ):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return decode_txt(path)
    if suffix == ".docx":
        return read_docx(path)
    raise ValueError(f"该入口不支持的文件格式：{suffix}")


def scan_documents(folder: Path, *, include_subfolders: bool) -> list[Path]:
    iterator = folder.rglob("*") if include_subfolders else folder.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in SUPPORTED
        and not path.name.startswith("~$")
    )

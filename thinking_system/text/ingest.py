"""Ингест книг: PDF / Markdown / txt → нормализованный текст.

PDF извлекается через PyMuPDF (самодостаточный, без внешних зависимостей).
Markdown/txt читаются как есть (символьная модель прекрасно работает с разметкой).
"""

from __future__ import annotations

import re
from pathlib import Path

_PDF_SUFFIXES = {".pdf"}
_TEXT_SUFFIXES = {
    ".md", ".markdown", ".txt", ".text",
    # код — система учится и на нём (всё это просто текст)
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".go", ".rs",
    ".rb", ".sh", ".html", ".css", ".sql", ".json", ".yaml", ".yml", ".toml",
}


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)        # хвостовые пробелы
    text = re.sub(r"\n{3,}", "\n\n", text)         # схлопнуть пустые строки
    return text.strip() + "\n"


def _load_pdf(path: Path) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    try:
        return "\n\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def load_book(path: str | Path) -> str:
    """Загрузить одну книгу (PDF/MD/txt) в нормализованный текст."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _PDF_SUFFIXES:
        raw = _load_pdf(path)
    elif suffix in _TEXT_SUFFIXES:
        raw = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"unsupported book format: {suffix} ({path})")
    return _normalize(raw)


def load_corpus(directory: str | Path) -> tuple[str, list[str]]:
    """Загрузить все книги из каталога (PDF/MD/txt) в один текст.

    Returns:
        (текст, список_имён_файлов). Файлы, начинающиеся с '_' или '.', пропускаются.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    supported = _PDF_SUFFIXES | _TEXT_SUFFIXES
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in supported and not p.name.startswith(("_", "."))
    )
    parts: list[str] = []
    sources: list[str] = []
    for p in files:
        try:
            parts.append(load_book(p))
            sources.append(p.name)
        except Exception as exc:  # noqa: BLE001 — пропускаем нечитаемые файлы, сообщая
            print(f"  ! пропуск {p.name}: {exc}")
    return ("\n\n".join(parts), sources)

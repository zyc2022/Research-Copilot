from pathlib import Path

from pypdf import PdfReader


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages: list[str] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {idx}]\n{text}")
        return "\n\n".join(pages)
    raise ValueError("Only .txt and .pdf files are supported")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    cleaned = "\n".join(line.strip() for line in text.splitlines())
    cleaned = "\n".join(line for line in cleaned.splitlines() if line)
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from research_agent.database import db
from research_agent.documents import chunk_text, extract_text
from research_agent.embeddings import cosine_similarity, dumps_embedding, embed_text, loads_embedding
from research_agent.paths import FILES_DIR


def row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row)


def get_model_config() -> dict[str, str]:
    with db() as conn:
        row = conn.execute("SELECT api_key, base_url, model FROM model_config WHERE id = 1").fetchone()
        return row_to_dict(row)


def save_model_config(api_key: str, base_url: str, model: str) -> dict[str, str]:
    with db() as conn:
        conn.execute(
            """
            UPDATE model_config
            SET api_key = ?, base_url = ?, model = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (api_key, base_url, model),
        )
    return get_model_config()


def list_conversations() -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT c.*,
                   (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY id DESC LIMIT 1) AS last_message
            FROM conversations c
            ORDER BY c.updated_at DESC, c.id DESC
            """
        ).fetchall()
        return [row_to_dict(row) for row in rows]


def create_conversation(title: str = "新对话") -> dict[str, Any]:
    with db() as conn:
        cur = conn.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (cur.lastrowid,)).fetchone()
        return row_to_dict(row)


def delete_conversation(conversation_id: int) -> None:
    with db() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def get_messages(conversation_id: int) -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,)
        ).fetchall()
        result = []
        for row in rows:
            item = row_to_dict(row)
            item["citations"] = json.loads(item.get("citations") or "[]")
            result.append(item)
        return result


def add_message(conversation_id: int, role: str, content: str, citations: list[dict[str, Any]] | None = None) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, citations) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, json.dumps(citations or [], ensure_ascii=False)),
        )
        title = content[:28] if role == "user" else None
        if title:
            conn.execute(
                """
                UPDATE conversations
                SET title = CASE WHEN title = '新对话' THEN ? ELSE title END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (title, conversation_id),
            )
        else:
            conn.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (conversation_id,),
            )


def list_knowledge_bases() -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            """
            SELECT kb.*,
                   COUNT(DISTINCT d.id) AS document_count,
                   COUNT(ch.id) AS chunk_count
            FROM knowledge_bases kb
            LEFT JOIN documents d ON d.kb_id = kb.id
            LEFT JOIN chunks ch ON ch.kb_id = kb.id
            GROUP BY kb.id
            ORDER BY kb.created_at DESC, kb.id DESC
            """
        ).fetchall()
        return [row_to_dict(row) for row in rows]


def create_kb(
    name: str,
    description: str,
    embedding_base_url: str,
    embedding_api_key: str,
    embedding_model: str,
) -> dict[str, Any]:
    dim = 384 if embedding_model == "local-hash" or not embedding_base_url else 0
    with db() as conn:
        cur = conn.execute(
            """
            INSERT INTO knowledge_bases
            (name, description, embedding_base_url, embedding_api_key, embedding_model, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, description, embedding_base_url, embedding_api_key, embedding_model, dim),
        )
        row = conn.execute("SELECT * FROM knowledge_bases WHERE id = ?", (cur.lastrowid,)).fetchone()
        return row_to_dict(row)


def delete_kb(kb_id: int) -> None:
    with db() as conn:
        rows = conn.execute("SELECT file_path FROM documents WHERE kb_id = ?", (kb_id,)).fetchall()
        conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
    for row in rows:
        path = Path(row["file_path"])
        if path.exists():
            path.unlink(missing_ok=True)


def set_kb_enabled(kb_id: int, enabled: bool) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE knowledge_bases SET enabled = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (1 if enabled else 0, kb_id),
        )


def get_kb(kb_id: int) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,)).fetchone()
        if row is None:
            raise ValueError("Knowledge base not found")
        return row_to_dict(row)


def list_documents(kb_id: int) -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE kb_id = ? ORDER BY created_at DESC, id DESC", (kb_id,)
        ).fetchall()
        return [row_to_dict(row) for row in rows]


def delete_document(document_id: int) -> None:
    with db() as conn:
        row = conn.execute("SELECT file_path FROM documents WHERE id = ?", (document_id,)).fetchone()
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    if row:
        Path(row["file_path"]).unlink(missing_ok=True)


async def upload_document(kb_id: int, upload: UploadFile) -> dict[str, Any]:
    kb = get_kb(kb_id)
    filename = Path(upload.filename or "document.txt").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".pdf"}:
        raise ValueError("Only .txt and .pdf files are supported")

    kb_dir = FILES_DIR / f"kb_{kb_id}"
    kb_dir.mkdir(parents=True, exist_ok=True)
    target = kb_dir / filename
    counter = 1
    while target.exists():
        target = kb_dir / f"{Path(filename).stem}_{counter}{suffix}"
        counter += 1

    with target.open("wb") as out:
        shutil.copyfileobj(upload.file, out)

    text = extract_text(target)
    chunks = chunk_text(text)
    if not chunks:
        target.unlink(missing_ok=True)
        raise ValueError("No extractable text found")

    with db() as conn:
        cur = conn.execute(
            """
            INSERT INTO documents (kb_id, filename, file_type, file_path, chunk_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (kb_id, target.name, suffix.lstrip("."), str(target), len(chunks)),
        )
        document_id = cur.lastrowid

        actual_dim = int(kb["embedding_dim"] or 0)
        for idx, chunk in enumerate(chunks):
            embedding = embed_text(
                chunk,
                kb["embedding_base_url"],
                kb["embedding_api_key"],
                kb["embedding_model"],
            )
            if actual_dim == 0:
                actual_dim = len(embedding)
                conn.execute(
                    "UPDATE knowledge_bases SET embedding_dim = ? WHERE id = ?",
                    (actual_dim, kb_id),
                )
            conn.execute(
                """
                INSERT INTO chunks (kb_id, document_id, chunk_index, content, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    kb_id,
                    document_id,
                    idx,
                    chunk,
                    dumps_embedding(embedding),
                    json.dumps({"filename": target.name, "chunk_index": idx}, ensure_ascii=False),
                ),
            )

        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return row_to_dict(row)


def retrieve(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    with db() as conn:
        kbs = conn.execute("SELECT * FROM knowledge_bases WHERE enabled = 1").fetchall()
        all_hits: list[dict[str, Any]] = []
        for kb in kbs:
            rows = conn.execute(
                """
                SELECT ch.*, d.filename, kb.name AS kb_name
                FROM chunks ch
                JOIN documents d ON d.id = ch.document_id
                JOIN knowledge_bases kb ON kb.id = ch.kb_id
                WHERE ch.kb_id = ?
                """,
                (kb["id"],),
            ).fetchall()
            if not rows:
                continue
            try:
                query_vec = embed_text(
                    query,
                    kb["embedding_base_url"],
                    kb["embedding_api_key"],
                    kb["embedding_model"],
                )
            except Exception:
                continue
            for row in rows:
                item = row_to_dict(row)
                score = cosine_similarity(query_vec, loads_embedding(item["embedding"]))
                all_hits.append(
                    {
                        "kb_id": item["kb_id"],
                        "kb_name": item["kb_name"],
                        "document_id": item["document_id"],
                        "filename": item["filename"],
                        "chunk_index": item["chunk_index"],
                        "content": item["content"],
                        "score": round(float(score), 4),
                    }
                )
    all_hits.sort(key=lambda x: x["score"], reverse=True)
    return all_hits[:top_k]

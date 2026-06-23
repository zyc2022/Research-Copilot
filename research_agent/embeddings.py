import hashlib
import json
import math
import urllib.error
import urllib.request


LOCAL_EMBEDDING_DIM = 384


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def local_hash_embedding(text: str, dim: int = LOCAL_EMBEDDING_DIM) -> list[float]:
    vec = [0.0] * dim
    tokens = [t.strip().lower() for t in text.replace("\n", " ").split() if t.strip()]
    if not tokens:
        tokens = [text[:120] or "empty"]

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    return normalize(vec)


def external_embedding(text: str, base_url: str, api_key: str, model: str) -> list[float]:
    url = base_url.rstrip("/") + "/embeddings"
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Embedding API error: {exc.code} {detail}") from exc

    return normalize([float(v) for v in data["data"][0]["embedding"]])


def embed_text(text: str, base_url: str, api_key: str, model: str) -> list[float]:
    if model == "local-hash" or not base_url:
        return local_hash_embedding(text)
    return external_embedding(text, base_url, api_key, model)


def dumps_embedding(vec: list[float]) -> str:
    return json.dumps(vec, separators=(",", ":"))


def loads_embedding(raw: str) -> list[float]:
    return [float(v) for v in json.loads(raw)]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    return sum(a[i] * b[i] for i in range(length))

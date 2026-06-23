import json
import urllib.error
import urllib.request


def call_chat_model(api_key: str, base_url: str, model: str, messages: list[dict[str, str]]) -> str:
    if not base_url or not model:
        return (
            "尚未配置聊天模型。请点击右下角“模型管理”，填写 base_url、api_key 和模型名称后再提问。"
        )

    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }
    ).encode("utf-8")
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Chat API error: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Chat API connection failed: {exc}") from exc

    return data["choices"][0]["message"]["content"]

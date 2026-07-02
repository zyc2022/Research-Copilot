from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from research_agent.llm import call_chat_model
from research_agent.services import (
    add_message,
    create_conversation,
    get_conversation_context,
    get_messages,
    get_model_config,
    retrieve,
    save_context_compression,
)


CONTEXT_LIMIT_CHARS = 20 * 1024
COMPRESSION_THRESHOLD_CHARS = int(CONTEXT_LIMIT_CHARS * 0.6)
RECENT_MESSAGE_COUNT = 8


class AgentState(TypedDict, total=False):
    user_message: str
    conversation_id: int | None
    model_config: dict[str, str]
    conversation_summary: str
    compressed_until_message_id: int | None
    history: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    context_compressed: bool


def estimate_chars(text: str) -> int:
    return len(text or "")


def format_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)


def load_state_node(state: AgentState) -> AgentState:
    conversation_id = state.get("conversation_id")
    if conversation_id is None:
        conversation_id = create_conversation()["id"]
    context = get_conversation_context(conversation_id)
    return {
        **state,
        "conversation_id": conversation_id,
        "model_config": get_model_config(),
        "conversation_summary": context.get("summary") or "",
        "compressed_until_message_id": context.get("compressed_until_message_id"),
        "history": get_messages(conversation_id),
        "context_compressed": False,
    }


def build_compression_messages(
    previous_summary: str,
    original_text: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You compress research conversation history for a Research Copilot. "
                "Preserve only information useful for future research reasoning. "
                "Do not invent facts. If prior summary and new messages conflict, keep the conflict explicit."
            ),
        },
        {
            "role": "user",
            "content": (
                "Previous compressed summary:\n"
                f"{previous_summary or '(empty)'}\n\n"
                "New original conversation text to compress:\n"
                f"{original_text}\n\n"
                "Return a concise structured summary with these sections:\n"
                "1. Research goals\n"
                "2. Confirmed facts and conclusions\n"
                "3. User experiments, data, and plans\n"
                "4. Pending hypotheses\n"
                "5. Open questions\n"
                "6. Constraints and preferences to remember"
            ),
        },
    ]


def compress_context_node(state: AgentState) -> AgentState:
    history = state.get("history", [])
    if len(history) <= RECENT_MESSAGE_COUNT:
        return state

    summary = state.get("conversation_summary", "")
    compressed_until = state.get("compressed_until_message_id")
    recent_messages = history[-RECENT_MESSAGE_COUNT:]
    uncompressed_messages = [
        msg for msg in history[:-RECENT_MESSAGE_COUNT] if compressed_until is None or msg["id"] > compressed_until
    ]
    if not uncompressed_messages:
        return {**state, "history": recent_messages}

    context_chars = (
        estimate_chars(summary)
        + estimate_chars(format_messages(uncompressed_messages))
        + estimate_chars(format_messages(recent_messages))
        + estimate_chars(state.get("user_message", ""))
    )
    if context_chars < COMPRESSION_THRESHOLD_CHARS:
        return state

    original_text = format_messages(uncompressed_messages)
    cfg = state["model_config"]
    if cfg.get("base_url") and cfg.get("model"):
        try:
            new_summary = call_chat_model(
                cfg.get("api_key", ""),
                cfg.get("base_url", ""),
                cfg.get("model", ""),
                build_compression_messages(summary, original_text),
            )
        except Exception as exc:
            new_summary = (
                f"{summary}\n\n"
                "Compression fallback: model call failed, preserving original text in compact log form.\n"
                f"Failure: {exc}\n"
                f"{original_text[:COMPRESSION_THRESHOLD_CHARS]}"
            ).strip()
    else:
        new_summary = (
            f"{summary}\n\n"
            "Compression fallback: no chat model configured, preserving original text in compact log form.\n"
            f"{original_text[:COMPRESSION_THRESHOLD_CHARS]}"
        ).strip()

    from_message_id = int(uncompressed_messages[0]["id"])
    to_message_id = int(uncompressed_messages[-1]["id"])
    save_context_compression(
        int(state["conversation_id"]),
        new_summary,
        from_message_id,
        to_message_id,
        original_text,
    )
    return {
        **state,
        "conversation_summary": new_summary,
        "compressed_until_message_id": to_message_id,
        "history": recent_messages,
        "context_compressed": True,
    }


def retrieve_node(state: AgentState) -> AgentState:
    chunks = retrieve(state["user_message"], top_k=6)
    return {**state, "retrieved_chunks": chunks}


def generate_node(state: AgentState) -> AgentState:
    chunks = state.get("retrieved_chunks", [])
    context = "\n\n".join(
        f"[{idx + 1}] Knowledge base: {chunk['kb_name']}; document: {chunk['filename']}; chunk: {chunk['chunk_index']}\n{chunk['content']}"
        for idx, chunk in enumerate(chunks)
    )
    history_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in state.get("history", [])[-RECENT_MESSAGE_COUNT:]
        if msg["role"] in {"user", "assistant"}
    ]
    summary = state.get("conversation_summary", "")
    system_prompt = (
        "You are Research Copilot. Answer primarily from the enabled knowledge-base context. "
        "Use the compressed conversation summary as memory from older dialogue, and recent messages as immediate context. "
        "If compressed memory conflicts with retrieved evidence, trust the retrieved evidence and mention the uncertainty. "
        "If evidence is insufficient, say that the enabled knowledge bases do not provide enough support. "
        "Use clear structure and cite sources with bracket numbers such as [1]."
    )
    user_prompt = (
        f"Compressed conversation summary:\n{summary or '(none)'}\n\n"
        f"Current user question:\n{state['user_message']}\n\n"
        f"Retrieved knowledge-base context:\n{context or 'No relevant context was retrieved.'}"
    )
    cfg = state["model_config"]
    try:
        answer = call_chat_model(
            cfg.get("api_key", ""),
            cfg.get("base_url", ""),
            cfg.get("model", ""),
            [{"role": "system", "content": system_prompt}, *history_messages, {"role": "user", "content": user_prompt}],
        )
    except Exception as exc:
        answer = f"Model call failed: {exc}"

    citations = [
        {
            "kb_name": chunk["kb_name"],
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "score": chunk["score"],
        }
        for chunk in chunks
    ]
    return {**state, "answer": answer, "citations": citations}


def save_node(state: AgentState) -> AgentState:
    conversation_id = int(state["conversation_id"])
    add_message(conversation_id, "user", state["user_message"])
    add_message(conversation_id, "assistant", state["answer"], state.get("citations", []))
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("load_state", load_state_node)
    graph.add_node("compress_context", compress_context_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("save", save_node)
    graph.set_entry_point("load_state")
    graph.add_edge("load_state", "compress_context")
    graph.add_edge("compress_context", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "save")
    graph.add_edge("save", END)
    return graph.compile()


rag_graph = build_graph()

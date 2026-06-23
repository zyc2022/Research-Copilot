from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from research_agent.llm import call_chat_model
from research_agent.services import add_message, create_conversation, get_messages, get_model_config, retrieve


class AgentState(TypedDict, total=False):
    user_message: str
    conversation_id: int | None
    model_config: dict[str, str]
    history: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]


def load_state_node(state: AgentState) -> AgentState:
    conversation_id = state.get("conversation_id")
    if conversation_id is None:
        conversation_id = create_conversation()["id"]
    return {
        **state,
        "conversation_id": conversation_id,
        "model_config": get_model_config(),
        "history": get_messages(conversation_id),
    }


def retrieve_node(state: AgentState) -> AgentState:
    chunks = retrieve(state["user_message"], top_k=6)
    return {**state, "retrieved_chunks": chunks}


def generate_node(state: AgentState) -> AgentState:
    chunks = state.get("retrieved_chunks", [])
    context = "\n\n".join(
        f"[{idx + 1}] 知识库：{chunk['kb_name']}；文档：{chunk['filename']}；片段：{chunk['chunk_index']}\n{chunk['content']}"
        for idx, chunk in enumerate(chunks)
    )
    history_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in state.get("history", [])[-8:]
        if msg["role"] in {"user", "assistant"}
    ]
    system_prompt = (
        "你是 Research Copilot。请优先依据提供的知识库上下文回答。"
        "如果上下文不足，要明确说明没有在已启用知识库中找到充分依据。"
        "回答要结构清晰、简洁，并在适合的位置标注来源编号，例如 [1]。"
    )
    user_prompt = (
        f"用户问题：{state['user_message']}\n\n"
        f"已检索上下文：\n{context or '没有检索到相关上下文。'}"
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
        answer = f"调用模型失败：{exc}"

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
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("save", save_node)
    graph.set_entry_point("load_state")
    graph.add_edge("load_state", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "save")
    graph.add_edge("save", END)
    return graph.compile()


rag_graph = build_graph()

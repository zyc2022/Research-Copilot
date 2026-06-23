from fastapi import APIRouter, File, HTTPException, UploadFile

from research_agent.graph import rag_graph
from research_agent.schemas import ChatIn, ConversationIn, KnowledgeBaseIn, ModelConfigIn
from research_agent.services import (
    create_conversation,
    create_kb,
    delete_conversation,
    delete_document,
    delete_kb,
    get_messages,
    get_model_config,
    list_conversations,
    list_documents,
    list_knowledge_bases,
    save_model_config,
    set_kb_enabled,
    upload_document,
)


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/model")
def model_config():
    return get_model_config()


@router.post("/model")
def update_model(payload: ModelConfigIn):
    return save_model_config(payload.api_key, payload.base_url, payload.model)


@router.get("/conversations")
def conversations():
    return list_conversations()


@router.post("/conversations")
def new_conversation(payload: ConversationIn):
    return create_conversation(payload.title)


@router.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: int):
    delete_conversation(conversation_id)
    return {"ok": True}


@router.get("/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: int):
    return get_messages(conversation_id)


@router.post("/chat")
def chat(payload: ChatIn):
    result = rag_graph.invoke(
        {"user_message": payload.message, "conversation_id": payload.conversation_id}
    )
    return {
        "conversation_id": result["conversation_id"],
        "answer": result["answer"],
        "citations": result.get("citations", []),
    }


@router.get("/knowledge-bases")
def knowledge_bases():
    return list_knowledge_bases()


@router.post("/knowledge-bases")
def new_kb(payload: KnowledgeBaseIn):
    try:
        return create_kb(
            payload.name,
            payload.description,
            payload.embedding_base_url,
            payload.embedding_api_key,
            payload.embedding_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/knowledge-bases/{kb_id}")
def remove_kb(kb_id: int):
    delete_kb(kb_id)
    return {"ok": True}


@router.post("/knowledge-bases/{kb_id}/enabled")
def toggle_kb(kb_id: int, enabled: bool):
    set_kb_enabled(kb_id, enabled)
    return {"ok": True}


@router.get("/knowledge-bases/{kb_id}/documents")
def documents(kb_id: int):
    return list_documents(kb_id)


@router.post("/knowledge-bases/{kb_id}/documents")
async def add_document(kb_id: int, file: UploadFile = File(...)):
    try:
        return await upload_document(kb_id, file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/documents/{document_id}")
def remove_document(document_id: int):
    delete_document(document_id)
    return {"ok": True}

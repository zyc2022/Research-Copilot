from pydantic import BaseModel, Field


class ModelConfigIn(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = ""


class KnowledgeBaseIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "local-hash"


class ChatIn(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: int | None = None


class ConversationIn(BaseModel):
    title: str = "新对话"

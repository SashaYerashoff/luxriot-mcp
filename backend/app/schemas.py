from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChunkResult(BaseModel):
    doc_id: str
    page_id: str
    heading_path: list[str]
    text: str
    score: float


class Citation(BaseModel):
    title: str
    doc_id: str
    page_id: str
    anchor: str | None = None
    source_path: str


class ImageResult(BaseModel):
    url: str
    doc_id: str
    page_id: str
    near_heading: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=8, ge=1, le=25)


class SearchResponse(BaseModel):
    chunks: list[ChunkResult]
    citations: list[Citation]
    images: list[ImageResult]


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1)
    k: int = Field(default=8, ge=1, le=25)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    images: list[ImageResult]
    session_id: str


class SessionCreateRequest(BaseModel):
    title: str | None = None


class SessionInfo(BaseModel):
    session_id: str
    title: str | None = None
    created_at: str
    last_message_at: str | None = None


class SessionsResponse(BaseModel):
    sessions: list[SessionInfo]


class Message(BaseModel):
    message_id: str
    session_id: str
    role: Literal["system", "user", "assistant"]
    content: str
    created_at: str


class MessagesResponse(BaseModel):
    messages: list[Message]


class ErrorResponse(BaseModel):
    error: str
    detail: Any | None = None


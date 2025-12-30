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
    debug: bool = False


class SearchResponse(BaseModel):
    chunks: list[ChunkResult]
    citations: list[Citation]
    images: list[ImageResult]
    debug: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1)
    k: int = Field(default=8, ge=1, le=25)


class WebSource(BaseModel):
    kind: Literal["fetch", "search"]
    url: str
    title: str | None = None
    snippet: str | None = None
    status: int | None = None
    final_url: str | None = None
    truncated: bool | None = None
    error: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    images: list[ImageResult]
    web_sources: list[WebSource] = Field(default_factory=list)
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


class AdminSettingsUpdateRequest(BaseModel):
    settings: dict[str, Any]


class AdminSettingsResponse(BaseModel):
    defaults: dict[str, Any]
    settings: dict[str, Any]
    effective: dict[str, Any]


class ReindexRequest(BaseModel):
    docs_dir: str | None = None
    compute_embeddings: bool = True
    embedding_max_chars: int = Field(default=448, ge=256, le=8000)
    embedding_batch_size: int = Field(default=8, ge=1, le=64)


class ReindexJob(BaseModel):
    job_id: str
    status: Literal["running", "succeeded", "failed"]
    phase: str
    docs_dir: str
    version: str
    compute_embeddings: bool
    embedding_max_chars: int
    embedding_batch_size: int
    pid: int | None = None
    build_dir: str | None = None
    started_at: str
    updated_at: str
    doc_total: int | None = None
    doc_done: int | None = None
    exit_code: int | None = None
    error: str | None = None
    logs_tail: list[str] = Field(default_factory=list)


class ReindexStatusResponse(BaseModel):
    defaults: dict[str, Any]
    job: ReindexJob | None = None


class DocCatalogEntry(BaseModel):
    doc_id: str
    doc_title: str
    page_count: int


class DocsCatalogResponse(BaseModel):
    docs: list[DocCatalogEntry]


class DocPageInfo(BaseModel):
    page_id: str
    page_title: str
    heading_path: list[str]
    anchor: str | None = None
    source_path: str


class DocCatalogDetailResponse(BaseModel):
    doc_id: str
    doc_title: str
    pages: list[DocPageInfo]


class PageImage(BaseModel):
    original: str
    url: str
    alt: str | None = None


class DocPageResponse(BaseModel):
    version: str
    doc_id: str
    doc_title: str
    page_id: str
    page_title: str
    heading_path: list[str]
    anchor: str | None = None
    source_path: str
    markdown: str
    images: list[PageImage]


class AuthMeResponse(BaseModel):
    authenticated: bool
    role: Literal["admin", "redactor", "support", "client", "anonymous"]
    username: str | None = None
    email: str | None = None
    greeting: str | None = None


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class LoginResponse(BaseModel):
    user: AuthMeResponse


class UserInfo(BaseModel):
    user_id: str
    username: str
    email: str | None = None
    role: Literal["admin", "redactor", "support", "client"]
    greeting: str | None = None
    disabled_at: str | None = None
    created_at: str
    updated_at: str


class UsersResponse(BaseModel):
    users: list[UserInfo]


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=1)
    email: str | None = None
    password: str = Field(min_length=6)
    role: Literal["admin", "redactor", "support", "client"] = "client"
    greeting: str | None = None


class UserUpdateRequest(BaseModel):
    email: str | None = None
    role: Literal["admin", "redactor", "support", "client"] | None = None
    greeting: str | None = None
    disabled: bool | None = None
    password: str | None = Field(default=None, min_length=6)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=6)


class PasswordResetRequest(BaseModel):
    new_password: str = Field(min_length=6)


class OkResponse(BaseModel):
    ok: bool = True

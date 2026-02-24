"""
schemas.py defines Pydantic models for that are used to validate incoming request bodys and outgoing responses.
"""

from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel


class Chat(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class Title(BaseModel):
    title: str


class MessageResponseFile(BaseModel):
    file_name: str
    url: str


class MessageResponseContent(BaseModel):
    message: str
    files: List[MessageResponseFile]


class MessageResponse(BaseModel):
    id: str
    chat_id: str
    role: Literal["user", "assistant"]
    content: List[MessageResponseContent]
    created_at: datetime


class ListFilesResponse(BaseModel):
    id: str
    name: str
    created_at: datetime


class ListChunksResponse(BaseModel):
    id: str
    file_id: str
    file_name: str
    index: int
    content: str
    page: str
    created_at: datetime


class ChatRequest(BaseModel):
    message: str

"""Database functionalities."""

from src.db.client import close_db_client, get_db_client
from src.db.crud import (
    create_chat,
    create_message,
    delete_all_chats,
    delete_chat,
    delete_file,
    get_all_chats,
    get_all_chunks,
    get_all_files,
    get_chat,
    get_chunks_by_file,
    insert_chunks,
    insert_file,
    update_chat,
)

__all__ = [
    "close_db_client",
    "get_db_client",
    "create_chat",
    "create_message",
    "delete_all_chats",
    "delete_chat",
    "delete_file",
    "get_all_chats",
    "get_all_chunks",
    "get_all_files",
    "get_chat",
    "get_chunks_by_file",
    "insert_chunks",
    "insert_file",
    "update_chat",
]

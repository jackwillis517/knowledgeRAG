from datetime import datetime
from typing import Any

from bson import ObjectId

from src.db.client import get_db_client

DB_NAME = "rag_db"


def _db():
    return get_db_client()[DB_NAME]


# Chunks


async def get_all_chunks() -> list[dict[str, Any]]:
    return await _db()["chunks"].find().to_list(100)


async def get_chunks_by_file(file_id: str) -> list[dict[str, Any]]:
    return await _db()["chunks"].find({"file_id": ObjectId(file_id)}).to_list(100)


async def insert_chunks(chunk_dicts: list[dict[str, Any]]) -> None:
    """Batch insert chunk documents."""
    if chunk_dicts:
        await _db()["chunks"].insert_many(chunk_dicts, ordered=False)


# Files


async def get_all_files() -> list[dict[str, Any]]:
    return (
        await _db()["files"]
        .find({}, {"_id": 1, "name": 1, "num_chunks": 1, "created_at": 1})
        .to_list(100)
    )


async def insert_file(
    filename: str, content: str, num_chunks: int, metadata: Any
) -> ObjectId:
    """Insert a file document and return its _id."""
    file_dict = {
        "name": filename,
        "content": content,
        "num_chunks": num_chunks,
        "metadata": metadata,
        "created_at": datetime.now(),
    }
    # TODO: store actual file in GridFS or s3 or r2 with file_id
    result = await _db()["files"].insert_one(file_dict)
    return result.inserted_id


async def delete_file(file_id: str) -> None:
    oid = ObjectId(file_id)
    await _db()["files"].delete_one({"_id": oid})
    await _db()["chunks"].delete_many({"file_id": oid})


# Chats


async def get_all_chats(user_id: str) -> list[dict[str, Any]]:
    return await _db()["chats"].find({"user_id": user_id}).to_list(100)


async def get_chat(chat_id: str) -> list[dict[str, Any]]:
    return await _db()["messages"].find({"chat_id": ObjectId(chat_id)}).to_list(100)


async def create_chat(title: str, user_id: str = "jacks-test-id") -> ObjectId:
    chat_dict = {
        "title": title,
        "user_id": user_id,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }
    result = await _db()["chats"].insert_one(chat_dict)
    return result.inserted_id


async def update_chat(chat_id: str, title: str) -> None:
    await _db()["chats"].update_one(
        {"_id": ObjectId(chat_id)},
        {"$set": {"title": title, "updated_at": datetime.now()}},
    )


async def delete_chat(chat_id: str) -> None:
    await _db()["chats"].delete_one({"_id": ObjectId(chat_id)})
    await _db()["messages"].delete_many({"chat_id": ObjectId(chat_id)})


async def delete_all_chats() -> None:
    await _db()["chats"].delete_many({})
    await _db()["messages"].delete_many({})


# Messages


async def create_message(chat_id: ObjectId, role: str, content: str) -> ObjectId:
    message_dict = {
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(),
    }
    result = await _db()["messages"].insert_one(message_dict)
    await _db()["chats"].update_one(
        {"_id": ObjectId(chat_id)}, {"$set": {"updated_at": datetime.now()}}
    )
    return result.inserted_id

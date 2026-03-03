import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path as FilePath
from typing import Annotated, List

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Path, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from langchain_openai import ChatOpenAI

from src.core import chunk_file, ingest, summarize_chat_title
from src.core.agent import query_agent, query_agent_include_info
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
    update_chat,
)
from src.schemas import ListChunksResponse, ListFilesResponse, Title

load_dotenv()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = get_db_client()
    app.state.model = ChatOpenAI(model=os.environ["AGENT_MODEL"], temperature=0.1)
    await client.admin.command("ping")
    logger.info("MongoDB Connected...")

    # Pre-warm Docling pipelines so the first real request is fast
    from src.core.chunker import file_converter, file_converter_ocr

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        f.write("warmup")
        warmup_path = FilePath(f.name)
    try:
        file_converter.convert(warmup_path)
        file_converter_ocr.convert(warmup_path)
        logger.info("Docling pipelines warmed up")
    finally:
        warmup_path.unlink(missing_ok=True)

    yield
    await close_db_client()
    logger.info("MongoDB Disconnected...")


app = FastAPI(lifespan=lifespan, title="KnowledgeRAG API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health", tags=["Health"], status_code=200)
async def health():
    return {"status": "ok"}


@app.get("/chunks", tags=["Chunks"], status_code=200)
async def list_all_chunks_route():
    chunks = await get_all_chunks()

    return [
        ListChunksResponse(
            id=str(c["_id"]),
            file_id=str(c["file_id"]),
            file_name=c["metadata"]["filename"],
            index=c["metadata"]["index"],
            content=c["content"],
            page=c["metadata"]["page"],
            created_at=c["created_at"],
        )
        for c in chunks
    ]


@app.get("/chunks/{file_id}", tags=["Chunks"], status_code=200)
async def list_chunks_for_file_route(file_id: str):
    chunks = await get_chunks_by_file(file_id)

    return [
        ListChunksResponse(
            id=str(c["_id"]),
            file_id=str(c["file_id"]),
            file_name=c["metadata"]["filename"],
            index=c["metadata"]["index"],
            content=c["content"],
            page=c["metadata"]["page"],
            created_at=c["created_at"],
        )
        for c in chunks
    ]


@app.get("/file", tags=["File"], status_code=200)
async def list_files() -> list[ListFilesResponse]:
    files = await get_all_files()
    return [
        ListFilesResponse(id=str(f["_id"]), name=f["name"], num_chunks=f["num_chunks"], created_at=f["created_at"])
        for f in files
    ]


@app.post("/file", tags=["File"], status_code=201)
async def upload_file(
    file: Annotated[
        UploadFile | None, File(description="File to upload via bytes")
    ] = None,
    url: Annotated[str | None, Form(description="File to upload via URL")] = None,
    path: Annotated[str | None, Form(description="File to upload via path")] = None,
    do_ocr: Annotated[
        bool,
        Form(description="Enable OCR for scanned/image-based PDFs. Not needed for digital PDFs with selectable text."),
    ] = False,
):
    # Validate: must provide one method for file ingestion
    if not file and not url and not path:
        raise HTTPException(400, "Must provide either 'file' or 'url'")
    if file and url or file and path or url and path:
        raise HTTPException(400, "Provide either 'file' or 'url', not both")

    # Chunk the file and get it's content in markdown for images it will be an llm description
    logger.info("Chunking File...")
    chunks = []
    file_content = ""
    if file:
        chunks, file_content = chunk_file(
            file_bytes=file.file.read(), filename=file.filename, do_ocr=do_ocr
        )
    elif url:
        chunks, file_content = chunk_file(url=url, do_ocr=do_ocr)
    elif path:
        chunks, file_content = chunk_file(file_path=path, do_ocr=do_ocr)
    logger.info("Chunking Succesful")

    # Uploads file and chunks to MongoDB
    logger.info("Ingesting chunks and file...")
    await ingest(
        chunks=chunks,
        filename=chunks[0]["metadata"]["filename"],
        content=file_content,
        metadata=None,
    )
    logger.info("File Ingestion Complete")


@app.delete("/file/{file_id}", tags=["File"], status_code=204)
async def delete_file_with_chunks(file_id: str):
    await delete_file(file_id)
    return Response(status_code=204)


# TODO: update return types with file urls
@app.get("/chat", tags=["Chat"], status_code=200)
async def list_all_chats():
    chats = await get_all_chats("jacks-test-id")
    return {"chats": chats}


# TODO: update return types with file urls
@app.get("/chat/{chat_id}", tags=["Chat"], status_code=200)
async def list_messages_for_chat(
    chat_id: Annotated[str, Path(description="ID of the chat to include message in")],
):
    messages = await get_chat(chat_id)
    # Serialize ObjectId fields to strings for JSON response
    for msg in messages:
        msg["_id"] = str(msg["_id"])
        msg["chat_id"] = str(msg["chat_id"])
    return {"chats": messages}


# TODO: need to deal with files
@app.post("/chat", tags=["Chat"], status_code=200)
async def send_message_new_chat(
    message: Annotated[str, Form(description="Message to ask the agent")],
    files: Annotated[
        List[UploadFile | None], File(description="File to upload via bytes")
    ] = [],
    include_info: Annotated[
        bool,
        Query(description="Include retrieved chunks and search method in response"),
    ] = False,
):
    title = summarize_chat_title(message)
    chat_id = await create_chat(title, user_id="jacks-test-id")
    await create_message(chat_id, "user", message)

    if include_info:
        result = await query_agent_include_info(app.state.model, message)
        await create_message(chat_id, "assistant", result["content"])
        return {"chat_id": str(chat_id), "result": result}
    else:
        result = await query_agent(app.state.model, message)
        await create_message(chat_id, "assistant", result["content"])
        return {"chat_id": str(chat_id), "result": result}


# TODO: need to deal with files
@app.post("/chat/{chat_id}", tags=["Chat"], status_code=200)
async def send_message_existing_chat(
    chat_id: Annotated[str, Path(description="ID of the chat to include message in")],
    message: Annotated[str, Form(description="Message to ask the agent")],
    files: Annotated[
        List[UploadFile | None], File(description="File to upload via bytes")
    ] = [],
    include_info: Annotated[
        bool,
        Query(description="Include retrieved chunks and search method in response"),
    ] = False,
):
    await create_message(ObjectId(chat_id), "user", message)

    if include_info:
        result = await query_agent_include_info(app.state.model, message)
        await create_message(ObjectId(chat_id), "assistant", result["content"])
        return {"result": result}
    else:
        result = await query_agent(app.state.model, message)
        await create_message(ObjectId(chat_id), "assistant", result["content"])
        return {"result": result}


@app.put("/chat/{chat_id}", tags=["Chat"], status_code=200)
async def update_chat_title(
    chat_id: Annotated[str, Path(description="ID of the chat to update")],
    title: Title,
):
    await update_chat(chat_id, title.title)
    return {"message": "Chat title updated successfully"}


@app.delete("/chat/{chat_id}", tags=["Chat"], status_code=204)
async def delete_chat_with_messages(
    chat_id: Annotated[str, Path(description="ID of the chat to delete")],
):
    await delete_chat(chat_id)
    return Response(status_code=204)


@app.delete("/chat", tags=["Chat"], status_code=204)
async def delete_every_chat():
    await delete_all_chats()
    return Response(status_code=204)

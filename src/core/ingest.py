import logging
from datetime import datetime

from src.core.llm import get_embeddings
from src.db.crud import insert_chunks, insert_file

logger = logging.getLogger(__name__)


async def ingest(chunks: list[dict], filename: str, content: str, metadata) -> None:
    file_id = await insert_file(filename, content, len(chunks), metadata)
    logger.info(f"File {filename} uploaded successfully")

    # Batch embed all chunks in one API call
    logger.info("Embedding chunks...")
    texts = [chunk["content"] for chunk in chunks]
    embeddings = get_embeddings(texts)

    chunk_dicts = [
        {
            "file_id": file_id,
            "content": chunk["content"],
            "embedding": embedding,
            "metadata": chunk["metadata"],
            "created_at": datetime.now(),
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    await insert_chunks(chunk_dicts)

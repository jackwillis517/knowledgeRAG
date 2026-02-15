import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv
from pymongo import AsyncMongoClient

from chunker import chunk_file
from llm import get_embeddings

load_dotenv()


async def ingest(mongo_client, chunks, filename, content, metadata):
    database = mongo_client["rag_db"]
    files_collection = database["files"]
    chunks_collection = database["chunks"]

    file_dict = {
        "name": filename,
        "content": content,
        "metadata": metadata,
        "created_at": datetime.now(),
    }

    # TODO: store actual file in GridFS or s3 with file_id

    file_result = await files_collection.insert_one(file_dict)
    file_id = file_result.inserted_id

    # Batch embed all chunks in one API call
    texts = [chunk["content"] for chunk in chunks]
    embeddings = get_embeddings(texts)

    chunk_dicts = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk_dict = {
            "file_id": file_id,
            "content": chunk["content"],
            "embedding": embedding,
            "metadata": chunk["metadata"],
            "created_at": datetime.now(),
        }
        chunk_dicts.append(chunk_dict)

    # Batch insert with ordered=False for partial success
    if chunk_dicts:
        await chunks_collection.insert_many(chunk_dicts, ordered=False)


async def main():
    try:
        client = AsyncMongoClient(os.environ["MONGO_CONNECTION_STRING"])
        await client.admin.command("ping")
        print("Connected successfully")

        test_filepath = "Introduction to Agents.pdf"
        # Get the chunks and markdown content of a file
        print("File chunking...")
        chunks, file_content = chunk_file(file_path=test_filepath)
        # Uploads file and chunks to MongoDB
        print("Chunks uploading...")
        await ingest(
            mongo_client=client,
            chunks=chunks,
            filename=chunks[0]["metadata"]["filename"],
            content=file_content,
            metadata=None,
        )

        await client.close()
        print("Ingestion successful")
    except Exception as e:
        raise Exception("The following error occurred: ", e)


if __name__ == "__main__":
    asyncio.run(main())

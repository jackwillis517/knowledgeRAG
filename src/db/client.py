import os

from dotenv import load_dotenv
from pymongo import AsyncMongoClient

load_dotenv()

_client: AsyncMongoClient | None = None


def get_db_client() -> AsyncMongoClient:
    """Return the shared AsyncMongoClient singleton.

    PyMongo's AsyncMongoClient manages its own connection pool internally,
    so a single instance is the recommended way to share connections
    across the application.
    """
    global _client
    if _client is None:
        _client = AsyncMongoClient(os.environ["MONGO_CONNECTION_STRING"])
    return _client


async def close_db_client() -> None:
    """Gracefully close the shared client and release its connection pool."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None

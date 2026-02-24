import asyncio
from typing import Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.core.llm import get_embedding
from src.db.client import get_db_client


class SearchResult(BaseModel):
    """Model for search results."""

    chunk_id: str = Field(..., description="MongoDB ObjectId of chunk as string")
    chunk_index: int = Field(..., description="Index of the chunk within the file")
    file_id: str = Field(..., description="Parent file ObjectId as string")
    file_name: str = Field(..., description="Name of the source file")
    file_type: str = Field(..., description="MIME type of the source file")
    file_page: str = Field(..., description="Page number or identifier")
    content: str = Field(..., description="Chunk text content")
    similarity: float = Field(..., description="Relevance score (0-1)")


files_collection_name = "files"
chunks_collection_name = "chunks"
chunks_embedding_index_name = "chunks_embedding_index"
chunks_search_index_name = "chunks_search_index"
match_count = 10


async def _semantic_search(query: str) -> List[SearchResult]:
    """
    Perform pure semantic search using MongoDB vector similarity.

    Args:
        query: Search query text

    Returns:
        List of search results ordered by similarity

    Raises:
        OperationFailure: If MongoDB operation fails (e.g., missing index)
    """
    # Generate embedding for query
    query_embedding = get_embedding(query)

    # Build MongoDB aggregation pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": chunks_embedding_index_name,
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,  # Search space (10x limit is good default)
                "limit": match_count,
            }
        },
        {
            "$lookup": {
                "from": files_collection_name,
                "localField": "file_id",
                "foreignField": "_id",
                "as": "file_info",
            }
        },
        {"$unwind": "$file_info"},
        {
            "$project": {
                "chunk_id": "$_id",
                "chunk_index": "$metadata.index",
                "file_id": 1,
                "file_name": "$file_info.name",
                "file_type": "$metadata.mime_type",
                "file_page": "$metadata.page",
                "content": 1,
                "similarity": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    # Execute aggregation
    database = get_db_client()["rag_db"]
    collection = database[chunks_collection_name]
    cursor = await collection.aggregate(pipeline)
    results = [doc async for doc in cursor][:match_count]

    search_results = [
        SearchResult(
            chunk_id=str(doc["chunk_id"]),
            chunk_index=doc["chunk_index"],
            file_id=str(doc["file_id"]),
            file_name=doc["file_name"],
            file_type=doc["file_type"],
            file_page=doc["file_page"],
            content=doc["content"],
            similarity=doc["similarity"],
        )
        for doc in results
    ]

    return search_results


async def _text_search(
    query: str,
) -> List[SearchResult]:
    """
    Perform full-text search using MongoDB Atlas Search.

    Uses $search operator for keyword matching, fuzzy matching, and phrase matching.

    Args:
        query: Search query text

    Returns:
        List of search results ordered by text relevance

    Raises:
        OperationFailure: If MongoDB operation fails (e.g., missing index)
    """
    # Build MongoDB Atlas Search aggregation pipeline
    pipeline = [
        {
            "$search": {
                "index": chunks_search_index_name,
                "text": {
                    "query": query,
                    "path": "content",
                    "fuzzy": {"maxEdits": 2, "prefixLength": 3},
                },
            }
        },
        {
            "$limit": match_count * 2  # Over-fetch for better RRF results
        },
        {
            "$lookup": {
                "from": files_collection_name,
                "localField": "file_id",
                "foreignField": "_id",
                "as": "file_info",
            }
        },
        {"$unwind": "$file_info"},
        {
            "$project": {
                "chunk_id": "$_id",
                "chunk_index": "$metadata.index",
                "file_id": 1,
                "file_name": "$file_info.name",
                "file_type": "$metadata.mime_type",
                "file_page": "$metadata.page",
                "content": 1,
                "similarity": {"$meta": "searchScore"},  # Text relevance score
            }
        },
    ]

    # Execute aggregation
    collection = get_db_client()["rag_db"][chunks_collection_name]
    cursor = await collection.aggregate(pipeline)
    results = [doc async for doc in cursor][: match_count * 2]

    search_results = [
        SearchResult(
            chunk_id=str(doc["chunk_id"]),
            chunk_index=doc["chunk_index"],
            file_id=str(doc["file_id"]),
            file_name=doc["file_name"],
            file_type=doc["file_type"],
            file_page=doc["file_page"],
            content=doc["content"],
            similarity=doc["similarity"],
        )
        for doc in results
    ]

    return search_results


def reciprocal_rank_fusion(
    search_results_list: List[List[SearchResult]], k: int = 60
) -> List[SearchResult]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF is a simple yet effective algorithm for combining results from different
    search methods. It works by scoring each document based on its rank position
    in each result list.

    Args:
        search_results_list: List of ranked result lists from different searches
        k: RRF constant (default: 60, standard in literature)

    Returns:
        Unified list of results sorted by combined RRF score

    Algorithm:
        For each document d appearing in result lists:
            RRF_score(d) = Σ(1 / (k + rank_i(d)))
        Where rank_i(d) is the position of document d in result list i.

    References:
        - Cormack et al. (2009): "Reciprocal Rank Fusion outperforms the best system"
        - Standard k=60 performs well across various datasets
    """
    # Build score dictionary by chunk_id
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, SearchResult] = {}

    # Process each search result list
    for results in search_results_list:
        for rank, result in enumerate(results):
            chunk_id = result.chunk_id

            # Calculate RRF contribution: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)

            # Accumulate score (automatic deduplication)
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += rrf_score
            else:
                rrf_scores[chunk_id] = rrf_score
                chunk_map[chunk_id] = result

    # Sort by combined RRF score (descending)
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Build final result list with updated similarity scores
    merged_results = []
    for chunk_id, rrf_score in sorted_chunks:
        result = chunk_map[chunk_id]
        # Create new result with updated similarity (RRF score)
        merged_result = SearchResult(
            chunk_id=result.chunk_id,
            chunk_index=result.chunk_index,
            file_id=result.file_id,
            file_name=result.file_name,
            file_type=result.file_type,
            file_page=result.file_page,
            content=result.content,
            similarity=rrf_score,  # Combined RRF score
        )
        merged_results.append(merged_result)

    return merged_results


async def _hybrid_search(
    query: str,
) -> List[SearchResult]:
    """
    Perform hybrid search combining semantic and keyword matching.

    Uses manual Reciprocal Rank Fusion (RRF) to merge vector and text search results.
    Works on all Atlas tiers including M0 (free tier) - no M10+ required!

    Args:
        query: Search query text
        match_count: Number of results to return (default: 10)
        text_weight: Weight for text matching (0-1, not used with RRF)

    Returns:
        List of search results sorted by combined RRF score

    Algorithm:
        1. Run semantic search (vector similarity)
        2. Run text search (keyword/fuzzy matching)
        3. Merge results using Reciprocal Rank Fusion
        4. Return top N results by combined score
    """
    # Over-fetch for better RRF results (2x requested count)
    fetch_count = match_count * 2

    # Run both searches concurrently for performance
    semantic_results, text_results = await asyncio.gather(
        _semantic_search(query),
        _text_search(query),
        return_exceptions=True,  # Don't fail if one search errors
    )

    # Handle errors gracefully
    if isinstance(semantic_results, Exception):
        print(f"Semantic search failed: {semantic_results}, using text results only")
        semantic_results = []
    if isinstance(text_results, Exception):
        print(f"Text search failed: {text_results}, using semantic results only")
        text_results = []

    # If both failed, return empty
    if not semantic_results and not text_results:
        print("Both semantic and text search failed")
        return []

    # Merge results using Reciprocal Rank Fusion
    merged_results = reciprocal_rank_fusion(
        [semantic_results, text_results],
        k=60,  # Standard RRF constant
    )

    # Return top N results
    final_results = merged_results[:match_count]

    print(
        f"hybrid_search_completed: query='{query}', "
        f"semantic={len(semantic_results)}, text={len(text_results)}, "
        f"merged={len(merged_results)}, returned={len(final_results)}"
    )

    return final_results


@tool(response_format="content_and_artifact")
async def semantic_search(query: str) -> tuple[str, List[SearchResult]]:
    """Perform semantic search using vector similarity. Best for conceptual or abstract queries."""
    results = await _semantic_search(query)
    return str(results), results


@tool(response_format="content_and_artifact")
async def text_search(query: str) -> tuple[str, List[SearchResult]]:
    """Perform full-text keyword search with fuzzy matching. Best for exact terms or phrases."""
    results = await _text_search(query)
    return str(results), results


@tool(response_format="content_and_artifact")
async def hybrid_search(query: str) -> tuple[str, List[SearchResult]]:
    """Perform hybrid search combining semantic and keyword matching. Use as the default search."""
    results = await _hybrid_search(query)
    return str(results), results

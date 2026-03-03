# Search Pipeline

This document describes the search pipeline — how user queries are matched against stored document chunks and synthesized into answers.

## Overview

```
User Query
    │
    ▼
LangChain Agent (gpt-4o-mini)
    │
    ├── Selects search tool based on query type
    │
    ▼
┌─────────────────────────────────────────────┐
│           Search Tool Execution             │
│                                             │
│  hybrid_search (default)                    │
│    ├── semantic_search ──► Vector Search     │
│    └── text_search ──────► Atlas Search      │
│    └── Reciprocal Rank Fusion ──► Top 10    │
│                                             │
│  semantic_search (conceptual queries)       │
│    └── Vector Search ──► Top 10             │
│                                             │
│  text_search (exact terms)                  │
│    └── Atlas Search ──► Top 10              │
└─────────────────────────────────────────────┘
    │
    ▼
Agent synthesizes answer from retrieved chunks
```

## Agent Setup

Defined in `src/core/agent/agent.py`

The agent is a LangChain tool-calling agent powered by the model specified in `AGENT_MODEL` (currently `gpt-4o-mini`, temperature 0.1).

### System Prompt

The agent is instructed to:

1. **Always** search the knowledge base before answering.
2. Select the appropriate search tool based on query characteristics.
3. Synthesize results into a coherent answer.
4. Cite source filenames.
5. Acknowledge when information is not found rather than fabricating answers.

### Tool Selection Guidance

| Tool              | When to Use                                                  |
| ----------------- | ------------------------------------------------------------ |
| `hybrid_search`   | **Default** — most queries. Combines semantic + keyword.     |
| `semantic_search`  | Conceptual or abstract queries, exploring ideas/relationships |
| `text_search`      | Specific terms, proper names, acronyms, exact phrases         |

## Search Strategies

All search tools are defined in `src/core/agent/tools.py`.

### Semantic Search

Function: `_semantic_search(query: str)`

1. **Embed the query** using `get_embedding()` (OpenAI `text-embedding-3-small`).
2. **MongoDB `$vectorSearch`** aggregation pipeline:
   - Index: `chunks_embedding_index`
   - Search space: 100 candidates (`numCandidates`)
   - Return limit: 10 results
3. **`$lookup`** join with the `files` collection to get file metadata.
4. **Score**: `vectorSearchScore` (cosine similarity, 0–1).

Best for queries about concepts, relationships, or paraphrased content where exact wording doesn't match.

### Text Search

Function: `_text_search(query: str)`

1. **MongoDB Atlas `$search`** aggregation pipeline:
   - Index: `chunks_search_index`
   - Searches the `content` field
   - Fuzzy matching: up to 2 edits, prefix length 3
2. **`$limit`**: 20 results (over-fetches for better RRF merging).
3. **`$lookup`** join with the `files` collection.
4. **Score**: Atlas `searchScore`.

Best for queries with specific terminology, names, or acronyms that need exact lexical matching.

### Hybrid Search (Default)

Function: `_hybrid_search(query: str)`

Combines both search strategies for the best of both worlds:

1. **Run both searches concurrently** via `asyncio.gather()`.
2. Each search over-fetches (20 results each) to provide better input for fusion.
3. **Merge via Reciprocal Rank Fusion (RRF)**.
4. Return the top 10 merged results.

**Graceful degradation**: If one search fails, the other's results are used alone. Only raises an error if both fail.

## Reciprocal Rank Fusion (RRF)

Function: `reciprocal_rank_fusion(search_results_list, k=60)`

RRF merges multiple ranked result lists into a single ranking. For each document `d`:

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
```

Where:
- `k = 60` (standard constant from Cormack et al., 2009)
- `rank_i(d)` is the rank of document `d` in search result list `i` (1-indexed)

**Properties:**
- Documents that appear in multiple lists get boosted scores.
- The constant `k` dampens the effect of high rankings in any single list.
- Results are deduplicated by `chunk_id`.

**Example**: A chunk ranked #1 in semantic and #3 in text search:
- Semantic contribution: `1 / (60 + 1) = 0.0164`
- Text contribution: `1 / (60 + 3) = 0.0159`
- Combined RRF score: `0.0323`

## Search Result Format

Each result is a `SearchResult` object:

```python
{
    "chunk_id": "507f1f77bcf86cd799439011",
    "chunk_index": 5,          # position within source file
    "file_id": "507f1f77bcf86cd799439012",
    "file_name": "Introduction to Agents.pdf",
    "file_type": "application/pdf",
    "file_page": "3",
    "content": "Agents combine language models with...",
    "similarity": 0.8742       # search score
}
```

## Agent Response Modes

### Standard Mode (`query_agent`)

Returns only the answer:

```json
{
    "content": "Based on the documents, agents combine..."
}
```

### Include Info Mode (`query_agent_include_info`)

Returns the answer plus diagnostics:

```json
{
    "content": "Based on the documents, agents combine...",
    "search_tool": "hybrid_search",
    "chunks": [
        {
            "chunk_id": "...",
            "chunk_index": 5,
            "file_name": "Introduction to Agents.pdf",
            "content": "...",
            "similarity": 0.8742
        }
    ]
}
```

This mode post-processes the LangChain message history to extract which tool was called and the artifact (search results) returned by that tool.

## Configuration

| Environment Variable | Purpose                          | Current Value            |
| -------------------- | -------------------------------- | ------------------------ |
| `AGENT_MODEL`        | LLM for agent reasoning          | `gpt-4o-mini`           |
| `EMBEDDING_MODEL`    | Embedding model for queries       | `text-embedding-3-small` |
| `OPENAI_API_KEY`     | OpenAI API authentication         | —                        |

| Constant             | Value | Description                              |
| -------------------- | ----- | ---------------------------------------- |
| `match_count`        | 10    | Final results returned per search        |
| RRF `k`              | 60    | Rank fusion dampening constant           |
| `numCandidates`      | 100   | Vector search candidate pool             |
| Fuzzy `maxEdits`     | 2     | Max edit distance for text search        |
| Fuzzy `prefixLength` | 3     | Characters that must match exactly       |

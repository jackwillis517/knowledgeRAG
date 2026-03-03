# Ingestion Pipeline

This document describes the end-to-end ingestion pipeline — from file upload to stored, searchable chunks in MongoDB.

## Overview

```
File Upload (bytes / URL / path)
        │
        ▼
  MIME Detection (mimes.py)
        │
        ├── Document ──► Docling Converter ──► HybridChunker
        │                                          │
        └── Image ──► OpenAI Vision API ──► Text ──► HybridChunker
                                                       │
                                                       ▼
                                              Batch Embedding (OpenAI)
                                                       │
                                                       ▼
                                              MongoDB Insert (file + chunks)
```

## Step 1: File Reception

Entry point: `chunk_file()` in `src/core/chunker.py`

Accepts exactly one input source:

| Parameter    | Description                        |
| ------------ | ---------------------------------- |
| `file_bytes` | Raw bytes from multipart upload    |
| `url`        | Remote URL — fetched via HTTP GET  |
| `file_path`  | Local filesystem path              |

MIME type is inferred from the filename extension via `mime_from_filename()` if not explicitly provided. URL fetches extract MIME from the `Content-Type` response header.

## Step 2: Document Processing

Routing is based on MIME type:

- **Documents** (PDF, DOCX, PPTX, HTML, Markdown, plain text) are processed by Docling's `DocumentConverter`.
- **Images** (PNG, JPEG, TIFF, BMP, WebP) are sent to the OpenAI Vision API for description, then processed as text.

### Document Conversion (Docling)

Function: `process_document()` in `src/core/chunker.py`

The Docling `DocumentConverter` is configured with:

- **OCR**: Enabled (`do_ocr=True`) for embedded images within PDFs.
- **Table structure detection**: Enabled (`do_table_structure=True`) for extracting tabular data.
- **Supported input formats**: PDF, DOCX, PPTX, HTML, XHTML, Markdown, plain text.

Since Docling requires a file path, uploaded bytes are written to a temporary file, converted, and the temp file is cleaned up afterward.

### Image Processing

Function: `process_image()` in `src/core/chunker.py`

1. Image bytes are base64-encoded and sent to the OpenAI chat completions API using the vision model (`IMAGE_UNDERSTANDING_MODEL` env var, currently `gpt-4o-mini`).
2. The model is prompted to extract all visible text, tables, charts, diagrams, and describe the image content in detail.
3. The description is wrapped in a markdown document with the image filename as a header.
4. This markdown is then processed through the standard document chunking path.

## Step 3: Chunking

Function: `chunk()` in `src/core/chunker.py`

Uses Docling's `HybridChunker` with the following configuration:

| Setting       | Value                                       |
| ------------- | ------------------------------------------- |
| Tokenizer     | `sentence-transformers/all-MiniLM-L6-v2`   |
| Max tokens    | 256 per chunk                               |
| Merge peers   | `True` (merges small adjacent chunks)       |

Each chunk is output as a dictionary:

```python
{
    "content": "chunk text...",
    "metadata": {
        "filename": "document.pdf",
        "mime_type": "application/pdf",
        "index": 0,           # position in document
        "element_type": "paragraph",  # heading, paragraph, table, etc.
        "page": "3",          # page number or "-1" if unavailable
    }
}
```

The chunker also extracts **provenance** metadata from Docling — tracking which structural element (heading, paragraph, list, table) each chunk originated from and which page it appears on.

## Step 4: Embedding

Function: `get_embeddings()` in `src/core/llm.py`

All chunk texts are embedded in a **single batch API call** to OpenAI:

| Setting | Value                        |
| ------- | ---------------------------- |
| Model   | `text-embedding-3-small`     |
| Input   | All chunk texts as a list    |

Results are sorted by index to guarantee ordering matches the input list. This batch approach minimizes API calls — one request regardless of chunk count.

## Step 5: Storage

Function: `ingest()` in `src/core/ingest.py`

Two database operations:

### File Record

Inserted into the `files` collection:

```python
{
    "name": "document.pdf",
    "content": "full markdown export of the document...",
    "num_chunks": 94,
    "metadata": None,       # reserved for future use
    "created_at": datetime
}
```

### Chunk Records

Batch-inserted into the `chunks` collection:

```python
{
    "file_id": ObjectId("..."),     # reference to parent file
    "content": "chunk text...",
    "embedding": [0.012, -0.034, ...],  # float vector
    "metadata": {
        "filename": "document.pdf",
        "mime_type": "application/pdf",
        "index": 0,
        "element_type": "paragraph",
        "page": "3"
    },
    "created_at": datetime
}
```

Chunks are inserted with `ordered=False` for maximum write throughput.

## MongoDB Indexes

Two indexes are required on the `chunks` collection for search:

1. **`chunks_embedding_index`** — MongoDB Atlas Vector Search index on the `embedding` field.
2. **`chunks_search_index`** — MongoDB Atlas Search index on the `content` field for full-text search.

## File Deletion

When a file is deleted (`DELETE /file/{file_id}`), both the file record and all associated chunks are removed in a single operation.

import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
import torch  # noqa: F401 — must be imported before docling to avoid DLL load error on Windows
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from llm import describe_image
from mimes import (
    DOCX_MIME,
    HTML_MIMES,
    IMAGE_MIMES,
    MD_MIMES,
    PDF_MIME,
    PPTX_MIME,
    mime_from_filename,
)


def build_converter() -> DocumentConverter:
    """
    Build a DocumentConverter with sensible defaults.
    For PDFs: enables table structure detection and embedded image OCR.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # OCR embedded images in PDFs
    pipeline_options.do_table_structure = True  # Detect table layouts

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    return converter


file_converter = build_converter()


def chunk(docling_doc, source_label: str, metadata: dict):
    """
    Run HybridChunker on a DoclingDocument and print each chunk.
    Returns list of chunk dicts (ready to be stored later).
    """
    chunker = HybridChunker(
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=256,  # max tokens per chunk
        merge_peers=True,  # merge small adjacent chunks
    )

    chunks = list(chunker.chunk(docling_doc))
    file_content = docling_doc.export_to_markdown()

    # print(f"\n{'=' * 60}")
    # print(f"SOURCE: {source_label}")
    # print(f"TOTAL CHUNKS: {len(chunks)}")
    # print(f"{'=' * 60}")

    results = []
    for i, chunk in enumerate(chunks):
        text = chunk.text.strip()
        if not text:
            continue

        # Docling provides provenance info: page, bounding box, element type
        chunk_meta = {**metadata}
        if chunk.meta:
            chunk_meta["index"] = i
            if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                item = chunk.meta.doc_items[0]
                chunk_meta["element_type"] = (
                    item.label.value
                    if hasattr(item.label, "value")
                    else str(item.label)
                )
                # Page available
                if item.prov:
                    prov = item.prov[0]
                    chunk_meta["page"] = prov.page_no

        # print(f"\n--- Chunk {i + 1} ---")
        # print(f"Type    : {chunk_meta.get('element_type', 'text')}")
        # if "page" in chunk_meta:
        #     print(f"Page    : {chunk_meta['page']}")
        # print(f"Text    : {text[:300]}{'...' if len(text) > 300 else ''}")

        results.append(
            {
                "content": text,
                "metadata": chunk_meta,
            }
        )

    return (results, file_content)


def process_document(
    file_bytes: bytes, filename: str, mime_type: str
) -> tuple[list[dict], str]:
    """
    Process a document (PDF (with images), DOCX, PPTX, HTML, Markdown/text)
    through Docling and return chunks.
    """
    suffix = Path(filename).suffix.lower() or ".bin"
    metadata = {
        "filename": filename,
        "mime_type": mime_type,
    }

    # Write to temp file — Docling requires a file path
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        result = file_converter.convert(tmp_path)
        doc = result.document
    finally:
        tmp_path.unlink(missing_ok=True)

    return chunk(doc, filename, metadata)


def process_image(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
) -> tuple[list[dict], str]:
    """
    Process an image file.
      → Send image to OpenAI API for text description
      → Wrap description as a Docling text document
      → Chunk normally
    """
    metadata = {
        "filename": filename,
        "mime_type": mime_type,
    }

    description = describe_image(file_bytes, mime_type)

    # Wrap the description as a plain text document for Docling to chunk
    # We create a temp .txt so Docling's text pipeline handles it
    with tempfile.NamedTemporaryFile(
        suffix=".md", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(f"[Image: {filename}]\n\n{description}")
        tmp_path = Path(tmp.name)

    try:
        result = file_converter.convert(tmp_path)
        doc = result.document
    finally:
        tmp_path.unlink(missing_ok=True)

    return chunk(doc, filename, metadata)


def _fetch_url(url: str) -> tuple[bytes, str, str]:
    """
    Download a URL and return (content_bytes, filename, mime_type).
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Extract MIME type from Content-Type header, stripping params like "; charset=..."
    content_type = response.headers.get("Content-Type", "application/octet-stream")
    mime_type = content_type.split(";")[0].strip()

    # Extract filename from URL path, fallback to "download"
    parsed = urlparse(url)
    path = Path(parsed.path)
    filename = path.name if path.name else "download"

    return response.content, filename, mime_type


def chunk_file(
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    url: Optional[str] = None,
    file_path: Optional[str | Path] = None,
) -> tuple[list[dict], str]:
    """
    Main entry point. Detects file type and routes to correct processor.

    Provide exactly one of `file_bytes`, `url`, or `file_path`.
    """
    sources = sum(x is not None for x in (file_bytes, url, file_path))
    if sources != 1:
        raise ValueError("Provide exactly one of 'file_bytes', 'url', or 'file_path'.")

    if file_path is not None:
        p = Path(file_path)
        file_bytes = p.read_bytes()
        if filename is None:
            filename = p.name

    if url:
        file_bytes, url_filename, url_mime = _fetch_url(url)
        if filename is None:
            filename = url_filename
        if mime_type is None:
            mime_type = url_mime

    if mime_type is None:
        mime_type = mime_from_filename(filename or "")

    # At this point file_bytes and filename are guaranteed set
    assert file_bytes is not None
    assert filename is not None

    if mime_type in IMAGE_MIMES:
        return process_image(file_bytes, filename, mime_type)
    elif mime_type in (PDF_MIME, DOCX_MIME, PPTX_MIME, *HTML_MIMES, *MD_MIMES):
        return process_document(file_bytes, filename, mime_type)
    else:
        raise ValueError(f"Unsupported file type: {mime_type} ({filename})")


# results = process_file(file_path="Introduction to Agents.pdf")
# idx = 0
# for res in results:
#     print(f"|========== Chunk #{idx + 1} ==========|")
#     print("|Content:", res["content"])
#     print("|")
#     print("|File Name:", res["metadata"]["filename"])
#     print("|")
#     print("|Mime Type:", res["metadata"]["mime_type"])
#     print("|")
#     print("|Page:", res["metadata"]["page"] or -1)
#     print("\n")
#     print("\n")

#     idx += 1

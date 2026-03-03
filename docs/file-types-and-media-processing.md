# Supported File Types & Media Processing

This document covers all currently supported file types, how image processing works, and future plans for video and multimodal pipelines.

## Supported File Types

### Documents

| Format       | Extensions     | MIME Type                                                                           | Processing Method        |
| ------------ | -------------- | ----------------------------------------------------------------------------------- | ------------------------ |
| PDF          | `.pdf`         | `application/pdf`                                                                   | Docling with OCR         |
| Word         | `.docx`        | `application/vnd.openxmlformats-officedocument.wordprocessingml.document`            | Docling                  |
| PowerPoint   | `.pptx`        | `application/vnd.openxmlformats-officedocument.presentationml.presentation`          | Docling                  |
| HTML         | `.html`, `.htm`| `text/html`                                                                         | Docling                  |
| XHTML        | `.xhtml`       | `application/xhtml+xml`                                                             | Docling                  |
| Markdown     | `.md`          | `text/markdown`, `text/x-markdown`                                                  | Docling                  |
| Plain Text   | `.txt`         | `text/plain`                                                                        | Docling                  |

### Images

| Format | Extensions      | MIME Type      | Processing Method      |
| ------ | --------------- | -------------- | ---------------------- |
| PNG    | `.png`          | `image/png`    | OpenAI Vision API      |
| JPEG   | `.jpg`, `.jpeg` | `image/jpeg`   | OpenAI Vision API      |
| TIFF   | `.tiff`         | `image/tiff`   | OpenAI Vision API      |
| BMP    | `.bmp`          | `image/bmp`    | OpenAI Vision API      |
| WebP   | `.webp`         | `image/webp`   | OpenAI Vision API      |

## Document Processing Details

All document types are processed through [Docling](https://github.com/DS4SD/docling), an intelligent document parsing library.

### PDF Processing

PDFs receive the most advanced processing pipeline:

- **OCR** (`do_ocr=True`): Extracts text from scanned pages and embedded images within the PDF.
- **Table structure detection** (`do_table_structure=True`): Identifies and preserves tabular layouts.
- **Multi-page support**: Page numbers are tracked per chunk via Docling's provenance metadata.

### Office Documents (DOCX, PPTX)

Parsed natively by Docling. Structural elements (headings, paragraphs, lists, tables) are preserved and tagged in chunk metadata.

### Web & Text Formats (HTML, Markdown, Plain Text)

Parsed by Docling with structure-aware handling. HTML tags and markdown formatting are interpreted to extract semantic structure.

## Image Processing Pipeline

Images follow a fundamentally different path from documents since they contain visual rather than textual information.

### Current Implementation

```
Image Upload (bytes)
       │
       ▼
Base64 Encode
       │
       ▼
OpenAI Vision API (gpt-4o-mini)
       │
       ├── Prompt: "Extract all visible text, describe
       │    tables, charts, diagrams, and visual content"
       │
       ▼
Text Description (markdown)
       │
       ▼
Wrap in markdown document with filename header
       │
       ▼
Standard chunking pipeline (Docling → HybridChunker)
       │
       ▼
Embedding + Storage
```

### Vision Prompt

The vision model is instructed to provide detailed descriptions covering:

- All visible **text** (headings, labels, captions, body text)
- **Tables**: Structure and data
- **Charts/graphs**: Type, axes, data points, trends
- **Diagrams**: Components, relationships, flow
- **General visual content**: Layout, notable elements

### Configuration

| Setting                       | Value          |
| ----------------------------- | -------------- |
| Vision model                  | `gpt-4o-mini`  |
| Max output tokens             | 2048           |
| Environment variable          | `IMAGE_UNDERSTANDING_MODEL` |

### Limitations

- Image descriptions are limited to what the vision model can interpret in a single pass.
- No multi-image context — each image is processed independently.
- The vision model output quality depends on image resolution and clarity.
- Max token output (2048) may truncate descriptions of very dense images.

## Future Plans

### Video Processing Pipeline

**Goal**: Enable ingestion and search over video content (lectures, tutorials, presentations, screencasts).

Planned approach:

1. **Frame extraction**: Sample keyframes at configurable intervals (e.g., every N seconds or on scene change detection).
2. **Audio transcription**: Extract audio track and transcribe via Whisper or equivalent ASR model.
3. **Frame analysis**: Send keyframes to a vision model for visual description (slides, diagrams, whiteboard content).
4. **Temporal alignment**: Align transcription segments with keyframe descriptions using timestamps.
5. **Chunking**: Chunk the combined transcript + visual descriptions, preserving timestamp metadata for seeking.
6. **Storage**: Each chunk stores a timestamp range, enabling "jump to source" in the original video.

Supported formats (planned): MP4, WebM, MOV, AVI, MKV.

### Enhanced Image Pipeline

Planned improvements to image processing:

1. **Multi-pass analysis**: Run multiple specialized prompts (text extraction, diagram understanding, data extraction) and merge results for richer descriptions.
2. **OCR fallback**: For images with dense text, use dedicated OCR (Tesseract / EasyOCR) alongside the vision model and merge outputs.
3. **Image-to-image context**: When multiple images are uploaded as part of the same document, provide cross-image context to the vision model.
4. **Structured output**: Extract tables and charts into structured formats (JSON/CSV) alongside the text description for more precise retrieval.

### Multimodal Retrieval

Longer-term plans for a unified multimodal search pipeline:

1. **Multimodal embeddings**: Use models like CLIP or similar to embed images directly as vectors alongside text embeddings, enabling cross-modal search (text query finds relevant images and vice versa).
2. **Unified chunk format**: A chunk could contain text, image references, and video timestamps, all embedded in a shared vector space.
3. **Source linking**: Search results link back to the original media — page in a PDF, timestamp in a video, or region in an image.
4. **Streaming ingestion**: Process large video files in streaming fashion rather than loading entirely into memory.

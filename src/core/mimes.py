from pathlib import Path

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
HTML_MIMES = {"text/html", "application/xhtml+xml"}
MD_MIMES = {"text/markdown", "text/x-markdown", "text/plain"}
IMAGE_MIMES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
    "image/bmp",
    "image/webp",
}
SUPPORTED_MIMES = {
    PDF_MIME,
    DOCX_MIME,
    PPTX_MIME,
    *HTML_MIMES,
    *MD_MIMES,
    *IMAGE_MIMES,
}


def mime_from_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    mapping = {
        ".pdf": PDF_MIME,
        ".docx": DOCX_MIME,
        ".pptx": PPTX_MIME,
        ".html": "text/html",
        ".htm": "text/html",
        ".xhtml": "application/xhtml+xml",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    return mapping.get(suffix, "application/octet-stream")

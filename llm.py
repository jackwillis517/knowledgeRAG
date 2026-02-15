import base64
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

VISION_PROMPT = """You are a document analysis assistant.

Describe the content of this image completely and precisely. Your output will be used for
semantic search, so include:

- All visible text, preserving hierarchy (titles, headings, body text)
- Table data row by row with column headers
- Chart/graph data: axes labels, data points, trends
- Diagrams: components, labels, relationships
- Any numbers, dates, or statistics
- The overall purpose or subject of the image

Format your response as clear structured prose. Do not say "I see" or "The image shows" —
just describe the content directly.
"""


def describe_image(image_bytes: bytes, mime_type: str) -> str:
    """
    Send image to OpenAI gpt-4o-mini vision API and get a rich text description.
    This description replaces the image for downstream Docling chunking.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_image}",
                        },
                    },
                    {
                        "type": "text",
                        "text": VISION_PROMPT,
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content or ""


def get_embedding(text: str) -> list[float]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    )
    return response.data[0].embedding


def get_embeddings(texts: list[str]) -> list[list[float]]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts,
    )
    # Sort by index to guarantee order matches input
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]

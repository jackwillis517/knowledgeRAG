from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

from src.core.agent.tools import (
    SearchResult,
    hybrid_search,
    semantic_search,
    text_search,
)

SYSTEM_PROMPT = """You are a knowledge retrieval assistant. Your job is to answer user questions \
by searching a document knowledge base using the tools available to you.

## Tool Selection

- **hybrid_search**: Your DEFAULT tool. Use this for most queries. It combines semantic \
understanding with keyword matching for the best results. When in doubt, use hybrid search.
- **semantic_search**: Use when the query is conceptual, abstract, or asks about meaning, \
relationships, or ideas (e.g. "What is the role of memory in agents?", "How do tools relate \
to planning?"). Best when exact keywords may not appear in the text.
- **text_search**: Use when the query targets specific terms, names, acronyms, or exact phrases \
(e.g. "What does ReAct stand for?", "Find mentions of LangChain"). Best for precise keyword \
lookups.

## Instructions

1. Always search the knowledge base before answering. Never answer from general knowledge alone.
2. If search results are relevant, synthesize them into a clear, concise answer.
3. Cite the source file name when providing information.
4. If no relevant results are found, say so honestly rather than guessing.
5. Keep answers grounded in the retrieved content.
"""


async def query_agent(model: str, prompt: str) -> Dict[str, Any]:
    tools = [semantic_search, text_search, hybrid_search]
    agent = create_agent(model, tools=tools, system_prompt=SYSTEM_PROMPT)

    result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
    return {"content": result["messages"][-1].content}


async def query_agent_include_info(model: str, prompt: str) -> Dict[str, Any]:
    tools = [semantic_search, text_search, hybrid_search]
    agent = create_agent(model, tools=tools, system_prompt=SYSTEM_PROMPT)

    result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result["messages"]

    content = messages[-1].content

    search_tool: str | None = None
    chunks: List[SearchResult] = []

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["name"] in (
                    "hybrid_search",
                    "semantic_search",
                    "text_search",
                ):
                    search_tool = tool_call["name"]

        if isinstance(msg, ToolMessage) and msg.name in (
            "hybrid_search",
            "semantic_search",
            "text_search",
        ):
            if msg.artifact:
                chunks = msg.artifact

    return {
        "content": content,
        "search_tool": search_tool,
        "chunks": [chunk.model_dump() for chunk in chunks],
    }

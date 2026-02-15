import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools import hybrid_search, semantic_search, text_search

load_dotenv()

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

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

tools = [semantic_search, text_search, hybrid_search]

agent = create_agent(model, tools=tools, system_prompt=SYSTEM_PROMPT)


async def main():
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What's an AI agent?"}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

from fastapi import Body
from rag import vector_service
from rag import embed
from schemas import TextModelRequest, TextModelResponse
from scraper import extract_urls, fetch_all


async def get_rag_content(body: TextModelRequest = Body(...)) -> str:
    rag_content = await vector_service.search("knowledgebase", embed(body.prompt), 3, 0.7)
    rag_content_str = "\n".join([c.payload["original_text"] for c in rag_content])

    return rag_content_str


async def get_urls_content(body: TextModelRequest = Body(...)) -> str:
    urls = extract_urls(body.prompt)
    if urls:
        try:
            urls_content = await fetch_all(urls)
            return urls_content
        except Exception as e:
            logger.warning(f"Failed to fetch one or several URls - Error: {e}")
    return ""

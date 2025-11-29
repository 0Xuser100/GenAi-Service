from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from stream import openai_chat_client

app = FastAPI()

app.mount("/pages", StaticFiles(directory="pages"), name="pages")

class Prompt(BaseModel):
    prompt: str

@app.post("/generate/text/stream")
async def serve_text_to_text_stream_controller(data: Prompt)-> StreamingResponse:
    return StreamingResponse(
        openai_chat_client.chat_stream(data.prompt),
        media_type="text/event-stream",
    )

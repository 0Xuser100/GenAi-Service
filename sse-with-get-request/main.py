
# # main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from stream import openai_chat_client
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/pages", StaticFiles(directory="pages"), name="pages")

@app.get("/generate/text/stream")
async def serve_text_to_text_stream_controller(prompt: str):
    return StreamingResponse(
        openai_chat_client.chat_stream(prompt),
        media_type="text/event-stream",
    )







# main.py for huggingface API
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# from stream import chat_stream
# from fastapi.staticfiles import StaticFiles

# app = FastAPI()

# # Serve HTML pages
# app.mount("/pages", StaticFiles(directory="pages"), name="pages")

# @app.get("/generate/text/stream")
# async def serve_text_to_text_stream_controller(prompt: str):
#     return StreamingResponse(
#         chat_stream(prompt),
#         media_type="text/event-stream",
#     )


















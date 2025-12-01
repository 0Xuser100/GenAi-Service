import asyncio
import os
from typing import AsyncGenerator
from openai import AsyncOpenAI
from dotenv import load_dotenv
from fastapi.websockets import WebSocket

# تحميل متغيرات البيئة من ملف .env
load_dotenv()


class OpenAIChatClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not self.api_key:
            raise ValueError("❌ Missing OPENAI_API_KEY in .env file")

        self.client = AsyncOpenAI(api_key=self.api_key)

    async def chat_stream(
        self, prompt: str, model: str | None = None,mode:str="sse"
    ) -> AsyncGenerator[str, None]:

        model_name = model or self.model

        stream = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            stream=True,
        )

        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            yield (
            f"data: {token}\n\n"
            if mode=="sse"
            else token
        
            )
            await asyncio.sleep(0.02)

        if mode=="sse":
            yield "data: [DONE]\n\n"

            
class WSConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)
        await websocket.close()

    @staticmethod
    async def receive(websocket: WebSocket) -> str:
        return await websocket.receive_text()
    
    async def broadcast(self, message: str | bytes | list | dict) -> None:
        for connection in self.active_connections:
            await self.send(message, connection)

    @staticmethod
    async def send(message: str | bytes | list | dict, websocket: WebSocket) -> None:
        if isinstance(message, str):
            await websocket.send_text(message)
        elif isinstance(message, bytes):
            await websocket.send_bytes(message)
        else:
            await websocket.send_json(message)


openai_chat_client = OpenAIChatClient()
ws_manager = WSConnectionManager()
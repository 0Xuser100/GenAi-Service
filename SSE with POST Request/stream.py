# # stream.py for OpenAI API
import asyncio
import os
from typing import AsyncGenerator
from openai import AsyncOpenAI
from dotenv import load_dotenv

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
        self, prompt: str, model: str | None = None
    ) -> AsyncGenerator[str, None]:

        model_name = model or self.model

        stream = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            stream=True,
        )

        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            yield f"data: {token}\n\n"
            await asyncio.sleep(0.02)

        yield "data: [DONE]\n\n"


openai_chat_client = OpenAIChatClient()




# # stream.py for Hugging Face Inference API

# import asyncio
# from typing import AsyncGenerator
# from huggingface_hub import AsyncInferenceClient

# client = AsyncInferenceClient("http://localhost:8080")


# async def chat_stream(prompt: str) -> AsyncGenerator[str, None]:
#     stream = await client.text_generation(prompt, stream=True)
#     async for token in stream:
#         yield token
#         await asyncio.sleep(0.05)



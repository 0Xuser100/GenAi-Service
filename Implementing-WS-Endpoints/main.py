# main.py

import asyncio
from fastapi import FastAPI
from fastapi.websockets import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from loguru import logger
from stream import ws_manager, openai_chat_client

app = FastAPI()

app.mount("/pages", StaticFiles(directory="pages"), name="pages")


def build_message(msg_type: str, data=None, message=None):
    """Standard WS protocol message"""
    return {
        "type": msg_type,
        "data": data,
        "message": message
    }


@app.websocket("/generate/text/streams")
async def websocket_endpoint(websocket: WebSocket):

    # accept with protocol
    await websocket.accept(subprotocol="llm.chat.stream.v1")
    logger.info("Client connected using protocol: llm.chat.stream.v1")

    try:
        while True:

            # Receive message
            client_input = await websocket.receive_text()

            # STOP command
            if client_input.lower().strip() == "stop":
                await websocket.send_json(build_message("info", message="Stopping stream"))
                await websocket.close(code=1000, reason="Client requested stop")
                return

            # Announce start of stream
            await websocket.send_json(build_message("start", message="Stream beginning"))

            # Stream tokens
            async for token in openai_chat_client.chat_stream(client_input, mode="ws"):

                if websocket.application_state != WebSocketState.CONNECTED:
                    logger.warning("Client disconnected mid-stream.")
                    return

                await websocket.send_json(build_message("token", data=token))
                await asyncio.sleep(0.02)

            # Announce stream finished
            await websocket.send_json(build_message("done"))

    except WebSocketDisconnect:
        logger.info("Client disconnected (code 1001).")

    except Exception as e:
        logger.error(f"Internal server error: {e}")

        # send structured error
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json(build_message("error", message="Internal server error"))
            await websocket.close(code=1011, reason="Server error")

    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close(code=1000, reason="Normal closure")

        logger.info("WebSocket closed cleanly.")

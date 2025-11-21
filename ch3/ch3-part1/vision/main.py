
# main.py

from fastapi import FastAPI, Body, Depends, Request,BackgroundTasks, File, UploadFile, status, HTTPException
from schemas import TextModelRequest, TextModelResponse
from dependencies import get_rag_content, get_urls_content
from models import generate_text,load_text_model
from typing import Annotated
from rag import pdf_text_extractor, vector_service
from upload import save_file

app = FastAPI()

@app.post("/upload")
async def file_upload_controller(
    file: Annotated[UploadFile, File(description="A file read as UploadFile")],
    bg_text_processor: BackgroundTasks,
):
    ...  # Raise a HTTPException if data upload is not a PDF file
    try:
        filepath = await save_file(file)
        bg_text_processor.add_task(pdf_text_extractor, filepath)
        bg_text_processor.add_task(
            vector_service.store_file_content_in_db,
            filepath.replace("pdf", "txt"),
            512,
            "knowledgebase",
            768,
        )

    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred while saving file - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {"filename": file.filename, "message": "File uploaded successfully"}


@app.post("/generate/text", response_model_exclude_defaults=True)
async def serve_text_to_text_controller(
    request: Request,
    body: TextModelRequest = Body(...),
    urls_content: str = Depends(get_urls_content),
    rag_content: str = Depends(get_rag_content),
) -> TextModelResponse:
    ...  # Raise HTTPException for invalid models
    prompt = body.prompt + " " + urls_content + rag_content
    load_text_model()
    output =  generate_text( prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host)

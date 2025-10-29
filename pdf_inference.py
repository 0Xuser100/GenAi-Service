import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv
from openai import OpenAI

HF_GATEWAY_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct:fireworks-ai"
PDF_FILENAME = "document.pdf"
OUTPUT_FILENAME = "pdf_inference_result.txt"
DEFAULT_PROMPT = "Summarize the important points from this page of the document."

# Rendering constraints
ATTEMPT_DPIS: List[int] = [120, 96, 80, 72, 60, 48, 36]
MAX_BASE64_IMAGE_BYTES: int = 900_000  # keep each payload well under router limits
MAX_PAGE_TEXT_CHARS: int = 1500  # fallback snippet size if page must be sent as text


@dataclass
class PagePayload:
    page_number: int
    mode: str  # "image" or "text"
    content: str  # base64 data URL for images, raw text for text mode
    dpi: Optional[int] = None  # actual DPI used when mode == "image"
    fallback_text: Optional[str] = None  # extracted text snippet for image fallback


def _ensure_hf_token() -> str:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in your environment or .env file before running.")
    return token


def _message_to_text(message) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if content:
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")
            if text:
                parts.append(text.strip())
        combined = "\n\n".join(filter(None, parts))
        if combined:
            return combined
    return str(message)


def _render_page_as_image(page: fitz.Page, page_number: int) -> Optional[PagePayload]:
    for dpi in ATTEMPT_DPIS:
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=matrix)
        png_bytes = pix.tobytes("png")
        encoded = base64.b64encode(png_bytes)
        if len(encoded) <= MAX_BASE64_IMAGE_BYTES:
            data_url = f"data:image/png;base64,{encoded.decode('ascii')}"
            return PagePayload(
                page_number=page_number,
                mode="image",
                content=data_url,
                dpi=dpi,
            )
    return None


def _extract_page_text(page: fitz.Page, page_number: int) -> Optional[PagePayload]:
    text = page.get_text("text").strip()
    if not text:
        return None
    snippet = text[:MAX_PAGE_TEXT_CHARS]
    return PagePayload(page_number=page_number, mode="text", content=snippet)


def iter_page_payloads(pdf_path: Path) -> Iterable[PagePayload]:
    doc = fitz.open(pdf_path)
    try:
        for index, page in enumerate(doc, start=1):
            payload = _render_page_as_image(page, index)
            if payload is not None:
                text_payload = _extract_page_text(page, index)
                if text_payload is not None:
                    payload.fallback_text = text_payload.content
                yield payload
                continue
            text_payload = _extract_page_text(page, index)
            if text_payload is None:
                raise ValueError(
                    f"Page {index} could not be rendered as an image and contained no extractable text."
                )
            yield text_payload
    finally:
        doc.close()


def _send_page_request(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    payload: PagePayload,
) -> tuple[str, str, Optional[int]]:
    header_text = f"{prompt}\n\nThis is page {payload.page_number}."

    content = [{"type": "text", "text": header_text}]
    if payload.mode == "image":
        content.append({"type": "image_url", "image_url": {"url": payload.content}})
    else:
        content.append(
            {
                "type": "text",
                "text": f"Extracted text from page {payload.page_number}:\n{payload.content}",
            }
        )
    try:
        completion = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}]
        )
        return _message_to_text(completion.choices[0].message), payload.mode, payload.dpi
    except openai.APIStatusError as exc:
        if getattr(exc, "status_code", None) == 413 and payload.mode == "image":
            if payload.fallback_text:
                fallback_content = [
                    {"type": "text", "text": header_text},
                    {
                        "type": "text",
                        "text": (
                            f"Image upload exceeded limit. Here is extracted text from page "
                            f"{payload.page_number}:\n{payload.fallback_text}"
                        ),
                    },
                ]
                completion = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": fallback_content}]
                )
                return (
                    _message_to_text(completion.choices[0].message),
                    "text-fallback",
                    None,
                )
            raise ValueError(
                f"Page {payload.page_number} still exceeds the gateway limit and no text fallback is available. "
                "Try lowering ATTEMPT_DPIS or split the PDF."
            ) from exc
        raise


def run_sequential_inference(
    pdf_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = HF_GATEWAY_URL,
    prompt: str = DEFAULT_PROMPT,
    hf_token: Optional[str] = None,
) -> List[str]:
    client = OpenAI(base_url=base_url, api_key=hf_token or _ensure_hf_token())
    responses: List[str] = []
    for payload in iter_page_payloads(pdf_path):
        response_text, mode_used, dpi_used = _send_page_request(
            client, model=model, prompt=prompt, payload=payload
        )
        page_header = f"Page {payload.page_number} ({mode_used}"
        if mode_used == "image" and dpi_used is not None:
            page_header += f", {dpi_used} DPI"
        page_header += "):"
        formatted = f"{page_header}\n{response_text}".strip()
        responses.append(formatted)
    return responses


def main() -> None:
    pdf_path = Path(__file__).with_name(PDF_FILENAME)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected {PDF_FILENAME} next to pdf_inference.py.")

    responses = run_sequential_inference(pdf_path)
    output_path = Path(__file__).with_name(OUTPUT_FILENAME)
    output_path.write_text("\n\n---\n\n".join(responses), encoding="utf-8")

    print(f"Inference complete. {len(responses)} page responses saved to {output_path}")


if __name__ == "__main__":
    main()

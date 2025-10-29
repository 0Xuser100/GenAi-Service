import argparse
import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv
from openai import OpenAI

HF_GATEWAY_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct:fireworks-ai"
PDF_FILENAME = "document.pdf"
OUTPUT_FILENAME = "pdf_inference_result.txt"
DEFAULT_PROMPT = "Summarize the important points from this page of the document."

DEFAULT_ATTEMPT_DPIS: Sequence[int] = (120, 96, 80, 72, 60, 48, 36)
DEFAULT_MAX_BASE64_IMAGE_BYTES: int = 900_000  # per request payload guard
DEFAULT_MAX_PAGE_TEXT_CHARS: int = 1500  # fallback snippet size if page must be sent as text
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_START_PAGE: int = 1
DEFAULT_STRATEGY: str = "auto"  # auto, image, text


@dataclass
class InferenceOptions:
    strategy: str = DEFAULT_STRATEGY
    attempt_dpis: Sequence[int] = DEFAULT_ATTEMPT_DPIS
    max_image_bytes: int = DEFAULT_MAX_BASE64_IMAGE_BYTES
    max_text_chars: int = DEFAULT_MAX_PAGE_TEXT_CHARS
    batch_size: int = DEFAULT_BATCH_SIZE
    start_page: int = DEFAULT_START_PAGE
    end_page: Optional[int] = None


def _parse_dpi_list(raw: Optional[str]) -> Sequence[int]:
    if not raw:
        return DEFAULT_ATTEMPT_DPIS
    values: List[int] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        dpi = int(stripped)
        if dpi <= 0:
            raise ValueError("All DPI values must be positive integers.")
        values.append(dpi)
    if not values:
        raise ValueError("At least one DPI value must be provided.")
    return tuple(values)


def _count_target_pages(pdf_path: Path, options: InferenceOptions) -> int:
    doc = fitz.open(pdf_path)
    try:
        total_pages = doc.page_count
    finally:
        doc.close()
    start_page = max(1, options.start_page)
    end_page = options.end_page or total_pages
    end_page = min(end_page, total_pages)
    if end_page < start_page:
        return 0
    return end_page - start_page + 1


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


def _render_page_as_image(
    page: fitz.Page, page_number: int, options: InferenceOptions
) -> Optional[PagePayload]:
    for dpi in options.attempt_dpis:
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=matrix)
        png_bytes = pix.tobytes("png")
        encoded = base64.b64encode(png_bytes)
        if len(encoded) <= options.max_image_bytes:
            data_url = f"data:image/png;base64,{encoded.decode('ascii')}"
            return PagePayload(
                page_number=page_number,
                mode="image",
                content=data_url,
                dpi=dpi,
            )
    return None


def _extract_page_text(
    page: fitz.Page, page_number: int, options: InferenceOptions
) -> Optional[PagePayload]:
    text = page.get_text("text").strip()
    if not text:
        return None
    snippet = text[: options.max_text_chars]
    return PagePayload(page_number=page_number, mode="text", content=snippet)


def iter_page_payloads(pdf_path: Path, options: InferenceOptions) -> Iterable[PagePayload]:
    doc = fitz.open(pdf_path)
    try:
        for index, page in enumerate(doc, start=1):
            logging.debug("Preparing page %d", index)
            if index < options.start_page:
                logging.debug(
                    "Skipping page %d (start_page=%d)", index, options.start_page
                )
                continue
            if options.end_page is not None and index > options.end_page:
                logging.debug(
                    "Stopping at page %d (end_page=%d)", index, options.end_page
                )
                break

            image_payload: Optional[PagePayload] = None
            text_payload: Optional[PagePayload] = None

            if options.strategy in ("auto", "image"):
                image_payload = _render_page_as_image(page, index, options)

            if options.strategy in ("auto", "text") or (
                options.strategy == "image" and image_payload is None
            ):
                text_payload = _extract_page_text(page, index, options)

            if options.strategy == "text":
                if text_payload is None:
                    raise ValueError(
                        f"Page {index} has no extractable text for text-only strategy."
                    )
                yield text_payload
                continue

            if options.strategy == "image":
                if image_payload is None:
                    raise ValueError(
                        f"Page {index} could not be rendered as an image within limits."
                    )
                logging.debug("Yielding page %d as image", index)
                yield image_payload
                continue

            # auto strategy
            if image_payload is not None:
                if text_payload is not None:
                    image_payload.fallback_text = text_payload.content
                yield image_payload
            elif text_payload is not None:
                yield text_payload
            else:
                raise ValueError(
                    f"Page {index} has neither renderable image nor extractable text."
                )
            logging.debug(
                "Yielded page %d with mode %s",
                index,
                image_payload.mode if image_payload else "text",
            )
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
        logging.debug(
            "Sending request for page %d using %s mode",
            payload.page_number,
            payload.mode,
        )
        completion = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}]
        )
        return _message_to_text(completion.choices[0].message), payload.mode, payload.dpi
    except openai.APIStatusError as exc:
        if getattr(exc, "status_code", None) == 413 and payload.mode == "image":
            if payload.fallback_text:
                logging.warning(
                    "Page %d exceeded image payload limit; falling back to text",
                    payload.page_number,
                )
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


def _send_text_batch_request(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    payloads: Sequence[PagePayload],
) -> str:
    page_numbers = [payload.page_number for payload in payloads]
    if not page_numbers:
        return ""
    if len(page_numbers) == 1:
        header_text = (
            f"{prompt}\n\nThis is page {page_numbers[0]} (text batch of size 1)."
        )
    else:
        header_text = (
            f"{prompt}\n\nThis batch covers pages {page_numbers[0]} to {page_numbers[-1]}."
        )
    combined_text = "\n\n".join(
        f"Page {payload.page_number}:\n{payload.content}" for payload in payloads
    )
    content = [
        {"type": "text", "text": header_text},
        {"type": "text", "text": combined_text},
    ]
    logging.debug(
        "Sending text batch covering pages %d-%d (%d pages)",
        page_numbers[0],
        page_numbers[-1],
        len(payloads),
    )
    completion = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}]
    )
    return _message_to_text(completion.choices[0].message)


def run_sequential_inference(
    pdf_path: Path,
    options: InferenceOptions,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = HF_GATEWAY_URL,
    prompt: str = DEFAULT_PROMPT,
    hf_token: Optional[str] = None,
) -> List[str]:
    client = OpenAI(base_url=base_url, api_key=hf_token or _ensure_hf_token())
    responses: List[str] = []
    payload_iter = iter_page_payloads(pdf_path, options)

    total_expected = _count_target_pages(pdf_path, options)
    logging.info(
        "Starting inference over %d page(s) with strategy=%s (batch_size=%d)",
        total_expected,
        options.strategy,
        options.batch_size,
    )

    start_time = time.perf_counter()
    processed_items = 0

    if options.strategy == "text" and options.batch_size > 1:
        batch: List[PagePayload] = []
        for payload in payload_iter:
            batch.append(payload)
            if len(batch) == options.batch_size:
                batch_start = time.perf_counter()
                response_text = _send_text_batch_request(
                    client, model=model, prompt=prompt, payloads=batch
                )
                batch_elapsed = time.perf_counter() - batch_start
                start_page = batch[0].page_number
                end_page = batch[-1].page_number
                label = (
                    f"Pages {start_page}-{end_page} (text-batch, {len(batch)} pages)"
                    if len(batch) > 1
                    else f"Page {start_page} (text)"
                )
                responses.append(f"{label}:\n{response_text}".strip())
                processed_items += len(batch)
                logging.info(
                    "Processed text batch %s-%s (%d pages) in %.2fs",
                    start_page,
                    end_page,
                    len(batch),
                    batch_elapsed,
                )
                batch = []
        if batch:
            batch_start = time.perf_counter()
            response_text = _send_text_batch_request(
                client, model=model, prompt=prompt, payloads=batch
            )
            batch_elapsed = time.perf_counter() - batch_start
            start_page = batch[0].page_number
            end_page = batch[-1].page_number
            label = (
                f"Pages {start_page}-{end_page} (text-batch, {len(batch)} pages)"
                if len(batch) > 1
                else f"Page {start_page} (text)"
            )
            responses.append(f"{label}:\n{response_text}".strip())
            processed_items += len(batch)
            logging.info(
                "Processed text batch %s-%s (%d pages) in %.2fs",
                start_page,
                end_page,
                len(batch),
                batch_elapsed,
            )
        elapsed = time.perf_counter() - start_time
        average = elapsed / processed_items if processed_items else 0.0
        logging.info(
            "Processed %d/%d text pages in %.2fs (average %.2fs per page)",
            processed_items,
            total_expected,
            elapsed,
            average,
        )
        logging.info(
            "Estimated remaining time: %.2fs",
            max(total_expected - processed_items, 0) * average,
        )
        return responses

    for payload in payload_iter:
        item_start = time.perf_counter()
        response_text, mode_used, dpi_used = _send_page_request(
            client, model=model, prompt=prompt, payload=payload
        )
        item_elapsed = time.perf_counter() - item_start
        processed_items += 1
        logging.info(
            "Page %d processed in %.2fs using %s%s",
            payload.page_number,
            item_elapsed,
            mode_used,
            f" at {dpi_used} DPI" if mode_used == "image" and dpi_used else "",
        )
        if total_expected:
            avg = (time.perf_counter() - start_time) / processed_items
            remaining = max(total_expected - processed_items, 0)
            logging.debug(
                "Average per page: %.2fs, estimated remaining time: %.2fs",
                avg,
                remaining * avg,
            )
        page_header = f"Page {payload.page_number} ({mode_used}"
        if mode_used == "image" and dpi_used is not None:
            page_header += f", {dpi_used} DPI"
        page_header += "):"
        formatted = f"{page_header}\n{response_text}".strip()
        responses.append(formatted)
    elapsed = time.perf_counter() - start_time
    logging.info(
        "Processed %d/%d page requests in %.2fs (average %.2fs each)",
        processed_items,
        total_expected,
        elapsed,
        elapsed / processed_items if processed_items else 0.0,
    )
    if processed_items and total_expected and processed_items < total_expected:
        avg = elapsed / processed_items
        logging.info(
            "Estimated time remaining for %d page(s): %.2fs",
            total_expected - processed_items,
            (total_expected - processed_items) * avg,
        )
    return responses


def _build_options_from_args(args: argparse.Namespace) -> InferenceOptions:
    return InferenceOptions(
        strategy=args.strategy,
        attempt_dpis=_parse_dpi_list(args.dpi_list),
        max_image_bytes=args.max_image_bytes,
        max_text_chars=args.max_text_chars,
        batch_size=args.batch_size,
        start_page=args.start_page,
        end_page=args.end_page,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multimodal inference over PDF pages with adjustable strategies."
    )
    parser.add_argument(
        "--pdf",
        default=PDF_FILENAME,
        help="Path to the PDF file (default: document.pdf next to this script).",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILENAME,
        help="Where to save the combined responses (default: pdf_inference_result.txt).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt sent with each request (default summarizes each page).",
    )
    parser.add_argument(
        "--strategy",
        choices=("auto", "image", "text"),
        default=DEFAULT_STRATEGY,
        help="auto tries images then text fallback, image enforces image-only, text extracts text only.",
    )
    parser.add_argument(
        "--dpi-list",
        dest="dpi_list",
        default=",".join(str(dpi) for dpi in DEFAULT_ATTEMPT_DPIS),
        help="Comma-separated DPI values to try for rendering (highest to lowest).",
    )
    parser.add_argument(
        "--max-image-bytes",
        type=int,
        default=DEFAULT_MAX_BASE64_IMAGE_BYTES,
        help="Maximum base64-encoded image size per request (default: 900000).",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=DEFAULT_MAX_PAGE_TEXT_CHARS,
        help="Maximum characters per page when extracting text (default: 1500).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of pages to combine per request in text mode (default: 1).",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=DEFAULT_START_PAGE,
        help="First page to process (1-indexed, default: 1).",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Last page to process (inclusive). Leave unset to read until the end.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model identifier to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        default=HF_GATEWAY_URL,
        help=f"OpenAI-compatible base URL (default: {HF_GATEWAY_URL}).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HF token override. Falls back to HF_TOKEN env/.env if omitted.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (e.g., DEBUG, INFO, WARNING). Default: INFO.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.batch_size <= 0:
        raise ValueError("batch-size must be >= 1.")
    if args.start_page <= 0:
        raise ValueError("start-page must be >= 1.")
    if args.end_page is not None and args.end_page < args.start_page:
        raise ValueError("end-page must be >= start-page when provided.")
    options = _build_options_from_args(args)
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    responses = run_sequential_inference(
        pdf_path,
        options,
        model=args.model,
        base_url=args.base_url,
        prompt=args.prompt,
        hf_token=args.hf_token,
    )
    output_path = Path(args.output).resolve()
    output_path.write_text("\n\n---\n\n".join(responses), encoding="utf-8")

    print(
        f"Inference complete. {len(responses)} response block(s) saved to {output_path}"
    )


if __name__ == "__main__":
    main()

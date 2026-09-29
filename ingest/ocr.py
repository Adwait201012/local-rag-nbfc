"""Shared Hindi/English Docling configuration for ingestion and OCR evaluation."""
from __future__ import annotations

import os
from pathlib import Path


def make_converter(force_full_page: bool | None = None):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    languages = [s.strip() for s in os.getenv("OCR_LANGUAGES", "hin,eng").split(",") if s.strip()]
    if not languages:
        raise ValueError("OCR_LANGUAGES must contain a Tesseract language code, e.g. hin,eng")
    if force_full_page is None:
        force_full_page = os.getenv("OCR_FORCE_FULL_PAGE", "false").lower() in {"1", "true", "yes"}
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.ocr_options = TesseractCliOcrOptions(
        lang=languages, force_full_page_ocr=force_full_page,
    )
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
    })


def convert_document(path: Path, converter=None):
    from docling.datamodel.base_models import ConversionStatus

    result = (converter or make_converter()).convert(str(path))
    # A document with one unreadable page used to be indexed minus that page.
    # Keep that behaviour rather than rejecting the whole file, but say so.
    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        print(f"[ocr] partial conversion, some pages skipped: {path.name}", flush=True)
    elif result.status != ConversionStatus.SUCCESS:
        raise RuntimeError(f"Document conversion failed: {path.name}: {result.status}")
    return result.document

# -*- coding: utf-8 -*-
"""
Unit tests for PDF extraction and clause chunking.
"""
from pathlib import Path
import pytest
from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.chunker import LegalClauseChunker
import config

SAMPLE_PDF = config.SAMPLE_DOCS_DIR / "Commercial_Vendor_Agreement.pdf"

def test_pdf_extraction():
    assert SAMPLE_PDF.exists(), f"Sample PDF missing at {SAMPLE_PDF}"
    extractor = PDFExtractor(enable_ocr=False)
    pages = extractor.extract_from_file(SAMPLE_PDF)

    assert len(pages) >= 2, "Expected at least 2 pages extracted"
    assert pages[0]["page_number"] == 1
    assert "Vendor Services Agreement" in pages[0]["text"]

    summary = extractor.get_document_summary(pages)
    assert summary["total_pages"] == len(pages)
    assert summary["total_words"] > 100

def test_legal_clause_chunker():
    extractor = PDFExtractor(enable_ocr=False)
    pages = extractor.extract_from_file(SAMPLE_PDF)

    chunker = LegalClauseChunker(chunk_size=300, overlap=40)
    chunks = chunker.chunk_pages(pages)

    assert len(chunks) > 0, "No chunks generated"
    first_chunk = chunks[0]
    assert "chunk_id" in first_chunk
    assert "section_title" in first_chunk
    assert "page_number" in first_chunk
    assert "text" in first_chunk
    assert len(first_chunk["text"]) >= config.MIN_CHUNK_CHARS

    section_titles = [c["section_title"] for c in chunks]
    has_target_section = any("termination" in s.lower() or "liability" in s.lower() or "services" in s.lower() for s in section_titles)
    assert has_target_section, f"Expected key section titles, got: {section_titles}"
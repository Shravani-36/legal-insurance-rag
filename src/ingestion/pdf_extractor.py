"""
PDF Text and Document Extractor using PyMuPDF (fitz) with OCR fallback.
"""
from typing import List, Dict, Any, Union, BinaryIO
from pathlib import Path
import io
import os
import shutil

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None


class PDFExtractor:
    """
    Extracts text, metadata, and page numbers from PDF files.
    Supports digital PDFs and gracefully handles scanned pages with OCR fallback.
    """

    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        self._tesseract_available = shutil.which("tesseract") is not None

    def is_tesseract_installed(self) -> bool:
        return self._tesseract_available

    def extract_from_file(self, file_source: Union[str, Path, BinaryIO, bytes], filename: str = "document.pdf") -> List[Dict[str, Any]]:
        """
        Extracts structured page content from a file path or in-memory stream.

        Returns:
            List[Dict[str, Any]]: Each item has 'page_number', 'text', 'char_count', 'source_file', 'is_scanned'
        """
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is not installed. Please install it via pip install pymupdf.")

        if isinstance(file_source, (str, Path)):
            doc = fitz.open(str(file_source))
            source_name = Path(file_source).name
        elif isinstance(file_source, bytes):
            doc = fitz.open(stream=file_source, filetype="pdf")
            source_name = filename
        elif hasattr(file_source, "read"):
            stream_data = file_source.read()
            doc = fitz.open(stream=stream_data, filetype="pdf")
            source_name = getattr(file_source, "name", filename)
        else:
            raise ValueError(f"Unsupported file source type: {type(file_source)}")

        pages: List[Dict[str, Any]] = []

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_num = page_idx + 1
                text = page.get_text("text").strip()
                is_scanned = False

                # If text is extremely short, check if it contains images (likely scanned)
                if len(text) < 40:
                    image_list = page.get_images(full=True)
                    if image_list:
                        is_scanned = True
                        if self.enable_ocr and self._tesseract_available:
                            ocr_text = self._perform_ocr_on_page(page)
                            if len(ocr_text) > len(text):
                                text = ocr_text

                pages.append({
                    "page_number": page_num,
                    "text": text,
                    "char_count": len(text),
                    "source_file": source_name,
                    "is_scanned": is_scanned
                })
        finally:
            doc.close()

        return pages

    def _perform_ocr_on_page(self, page) -> str:
        """
        Render page pixmap and run pytesseract OCR.
        """
        try:
            import pytesseract
            from PIL import Image

            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_result = pytesseract.image_to_string(img)
            return ocr_result.strip()
        except Exception:
            return ""

    def get_document_summary(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns high-level statistics for the extracted document.
        """
        total_pages = len(pages)
        total_chars = sum(p["char_count"] for p in pages)
        total_words = sum(len(p["text"].split()) for p in pages)
        scanned_pages = sum(1 for p in pages if p["is_scanned"])

        return {
            "total_pages": total_pages,
            "total_characters": total_chars,
            "total_words": total_words,
            "scanned_pages": scanned_pages,
            "filename": pages[0]["source_file"] if pages else "unknown"
        }

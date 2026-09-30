# -*- coding: utf-8 -*-
"""
Smart legal-clause and section-aware chunker.
Preserves legal structure, section headers, clause numbers, and page references.
"""
from typing import List, Dict, Any, Optional
import re
import config


class LegalClauseChunker:
    """
    Chunks legal and insurance text while respecting clause and section boundaries.
    Maintains section title headers and page provenance for citation.
    """

    def __init__(self, chunk_size: int = config.DEFAULT_CHUNK_SIZE, overlap: int = config.DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.header_patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in config.LEGAL_HEADER_PATTERNS]

    def _detect_section_header(self, line: str) -> Optional[str]:
        clean_line = line.strip()
        if not clean_line or len(clean_line) > 120:
            return None

        for pattern in self.header_patterns:
            match = pattern.match(clean_line)
            if match:
                groups = [g.strip() for g in match.groups() if g]
                return " - ".join(groups) if groups else clean_line

        # Heuristic for short uppercase lines that often represent legal headers
        if clean_line.isupper() and 4 < len(clean_line) < 60 and not clean_line.endswith("."):
            return clean_line

        return None

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes extracted pages and returns structured, section-aware chunks.
        """
        chunks: List[Dict[str, Any]] = []
        current_section = "Preamble / General Provisions"
        buffer_words: List[str] = []
        buffer_page = pages[0]["page_number"] if pages else 1
        source_file = pages[0]["source_file"] if pages else "document.pdf"
        chunk_counter = 0

        for page in pages:
            page_num = page["page_number"]
            text = page["text"]
            lines = text.split("\n")

            for line in lines:
                header = self._detect_section_header(line)
                
                # If a new section header is found and buffer already has substantial content, flush it
                if header:
                    if len(buffer_words) >= 80:
                        chunk_text = " ".join(buffer_words).strip()
                        if len(chunk_text) >= config.MIN_CHUNK_CHARS:
                            chunk_counter += 1
                            chunks.append({
                                "chunk_id": f"{source_file}_chk_{chunk_counter:04d}",
                                "section_title": current_section,
                                "page_number": buffer_page,
                                "text": chunk_text,
                                "word_count": len(buffer_words),
                                "source_file": source_file
                            })
                            # Keep overlap
                            buffer_words = buffer_words[-self.overlap:] if self.overlap < len(buffer_words) else []
                            buffer_page = page_num

                    current_section = header

                # Add line words to buffer
                words = line.strip().split()
                if words:
                    if not buffer_words:
                        buffer_page = page_num
                    buffer_words.extend(words)

                # If buffer exceeds target chunk size, create a chunk
                if len(buffer_words) >= self.chunk_size:
                    chunk_text = " ".join(buffer_words).strip()
                    chunk_counter += 1
                    chunks.append({
                        "chunk_id": f"{source_file}_chk_{chunk_counter:04d}",
                        "section_title": current_section,
                        "page_number": buffer_page,
                        "text": chunk_text,
                        "word_count": len(buffer_words),
                        "source_file": source_file
                    })
                    # Overlap
                    buffer_words = buffer_words[-self.overlap:] if self.overlap < len(buffer_words) else []
                    buffer_page = page_num

        # Flush any remaining words in buffer
        if buffer_words:
            chunk_text = " ".join(buffer_words).strip()
            if len(chunk_text) >= config.MIN_CHUNK_CHARS or not chunks:
                chunk_counter += 1
                chunks.append({
                    "chunk_id": f"{source_file}_chk_{chunk_counter:04d}",
                    "section_title": current_section,
                    "page_number": buffer_page,
                    "text": chunk_text,
                    "word_count": len(buffer_words),
                    "source_file": source_file
                })

        return chunks
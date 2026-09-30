# -*- coding: utf-8 -*-
"""
Unit tests for FAISS dense search, BM25 keyword search, and Hybrid search.
"""
import pytest
from src.retrieval.bm25_search import BM25Retriever
from src.retrieval.vector_search import FAISSRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

MOCK_CHUNKS = [
    {
        "chunk_id": "doc_chk_001",
        "section_title": "SECTION 7 - TERMINATION",
        "page_number": 2,
        "text": "Either Party may terminate this Agreement for convenience at any time without cause by providing thirty (30) days prior written notice."
    },
    {
        "chunk_id": "doc_chk_002",
        "section_title": "SECTION 6 - LIMITATION OF LIABILITY",
        "page_number": 2,
        "text": "In no event shall either Party aggregate cumulative liability exceed the total fees paid in the twelve months preceding the incident, not to exceed $1,000,000."
    },
    {
        "chunk_id": "doc_chk_003",
        "section_title": "SECTION 8 - INSURANCE REQUIREMENTS",
        "page_number": 3,
        "text": "Vendor shall maintain Commercial General Liability insurance with limits not less than $2,000,000 per occurrence and $4,000,000 aggregate."
    },
    {
        "chunk_id": "doc_chk_004",
        "section_title": "SCHEDULE B - DEDUCTIBLES",
        "page_number": 1,
        "text": "The annual individual deductible is $1,500. The family aggregate deductible is $3,000. Maximum out-of-pocket is $6,000."
    }
]

def test_bm25_retriever():
    retriever = BM25Retriever()
    retriever.fit(MOCK_CHUNKS)

    results = retriever.search("terminate convenience 30 days notice", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "TERMINATION" in top_chunk["section_title"]
    assert score > 0.0

def test_faiss_retriever():
    retriever = FAISSRetriever()
    retriever.fit(MOCK_CHUNKS)

    # Semantic query without exact keywords
    results = retriever.search("how much is the maximum damage payout cap?", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "LIABILITY" in top_chunk["section_title"] or "DEDUCTIBLES" in top_chunk["section_title"]
    assert 0.0 <= score <= 1.0

def test_hybrid_retriever():
    hybrid = HybridRetriever()
    hybrid.fit(MOCK_CHUNKS)

    # Test RRF
    rrf_results = hybrid.search("insurance requirements commercial general liability", top_k=2, use_rrf=True)
    assert len(rrf_results) > 0
    assert "INSURANCE" in rrf_results[0]["section_title"]
    assert "hybrid_score" in rrf_results[0]
    assert "vector_score" in rrf_results[0]
    assert "bm25_score" in rrf_results[0]

    # Test Convex Linear Combination
    linear_results = hybrid.search("deductible and out of pocket limit", top_k=2, alpha=0.5, use_rrf=False)
    assert len(linear_results) > 0
    assert "DEDUCTIBLES" in linear_results[0]["section_title"]
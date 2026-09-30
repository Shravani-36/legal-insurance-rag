# -*- coding: utf-8 -*-
"""
Unit tests for the Legal RAG generator and evaluation suite.
"""
import pytest
from src.rag.generator import LegalRAGGenerator
from src.evaluation.rag_evaluator import RAGEvaluator

MOCK_CHUNKS = [
    {
        "chunk_id": "doc_chk_001",
        "section_title": "Section 7 - Termination",
        "page_number": 2,
        "primary_category": "Termination For Convenience",
        "risk_level": "MEDIUM",
        "text": "Either Party may terminate this Agreement for convenience at any time without cause by providing thirty (30) days prior written notice."
    }
]

def test_rag_generation_offline():
    generator = LegalRAGGenerator(provider="offline")
    question = "What is the termination notice period?"
    result = generator.generate(question, MOCK_CHUNKS)

    assert "answer" in result
    assert "sources" in result
    ans = result["answer"]

    # Verify structured sections
    assert "Plain English" in ans or "Explanation" in ans
    assert "30 days" in ans or "thirty (30) days" in ans
    assert "Section 7" in ans

def test_rag_evaluator_metrics():
    evaluator = RAGEvaluator()
    question = "What is the termination notice period?"
    answer = "The notice period for termination is thirty (30) days as stated in Section 7 - Termination."

    metrics = evaluator.evaluate_single_turn(
        question=question,
        answer=answer,
        retrieved_chunks=MOCK_CHUNKS,
        target_section="Section 7 - Termination",
        expected_keywords=["thirty (30) days", "written notice"]
    )

    assert "context_recall" in metrics
    assert "faithfulness" in metrics
    assert "answer_relevance" in metrics
    assert "citation_precision" in metrics
    assert "overall_rag_score" in metrics

    assert metrics["context_recall"] > 0.5
    assert metrics["overall_rag_score"] > 0.5
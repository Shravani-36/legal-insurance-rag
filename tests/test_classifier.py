# -*- coding: utf-8 -*-
"""
Unit tests for the Legal and Insurance Clause Classifier (CUAD 41 + Insurance).
"""
import pytest
from src.ingestion.legal_classifier import LegalClauseClassifier

def test_termination_classification():
    classifier = LegalClauseClassifier()
    text = "Either party may terminate this agreement for convenience at any time upon 30 days prior written notice."
    result = classifier.classify_text(text, section_title="Section 7 - Termination")

    assert result["primary_category"] == "Termination For Convenience"
    assert result["confidence"] >= 0.4
    assert result["risk_level"] in ["MEDIUM", "HIGH"]

def test_liability_cap_classification():
    classifier = LegalClauseClassifier()
    text = "In no event shall total cumulative liability exceed the aggregate fees paid in the preceding twelve months, capped at $1,000,000."
    result = classifier.classify_text(text, section_title="Section 6 - Limitation of Liability")

    assert result["primary_category"] == "Cap On Liability"
    assert result["confidence"] >= 0.4

def test_insurance_deductible_classification():
    classifier = LegalClauseClassifier()
    text = "The calendar year deductible is $1,500 for individuals. Out-of-pocket maximum is $6,000."
    result = classifier.classify_text(text, section_title="Schedule B - Deductibles")

    assert result["primary_category"] == "Deductible & Out-of-Pocket" or result["primary_category"] == "Coverage Limit"
    assert result["confidence"] >= 0.3

def test_enrich_chunks():
    classifier = LegalClauseClassifier()
    chunks = [
        {
            "chunk_id": "c1",
            "section_title": "Section 9 - Indemnification",
            "text": "Vendor agrees to defend, indemnify, and hold harmless Customer against third-party patent infringement claims."
        }
    ]
    enriched = classifier.enrich_chunks(chunks)
    assert enriched[0]["primary_category"] == "Indemnity"
    assert enriched[0]["risk_level"] == "HIGH"
# -*- coding: utf-8 -*-
"""
Configuration settings for RAG-Based Legal and Insurance Document Intelligence.
"""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = BASE_DIR / "sample_documents"
BENCHMARK_PATH = BASE_DIR / "src" / "evaluation" / "benchmark_dataset.json"

DEFAULT_CHUNK_SIZE = 450
DEFAULT_CHUNK_OVERLAP = 50
MIN_CHUNK_CHARS = 80

LEGAL_HEADER_PATTERNS = [
    r'^(?:ARTICLE|SECTION|CLAUSE)\s+([0-9IVXLCDM]+(?:\.[0-9]+)*)[:\.\s\-\–]+([^\n\r]+)',
    r'^([0-9]{1,2}\.[0-9]{1,2}(?:\.[0-9]+)*)\s+([^\n\r]+)',
    r'^([A-Z][A-Za-z0-9\s,\-\–\(\)]{3,50}):\s*$',
    r'^(?:SCHEDULE|EXHIBIT|APPENDIX|ATTACHMENT)\s+([A-Z0-9]+)[:\.\s\-\–]+([^\n\r]+)',
]

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 4
DEFAULT_HYBRID_ALPHA = 0.65
RRF_K = 60

CUAD_CATEGORIES = {
    "Document Name": "Name or title of the legal agreement or contract",
    "Parties": "Entities, corporations, or individuals entering into the agreement",
    "Agreement Date": "Date on which the agreement was executed or formed",
    "Effective Date": "Date on which the agreement becomes operational or binding",
    "Expiration Date": "Date specifying when the contract term naturally ends",
    "Renewal Term": "Conditions, notice periods, and terms for auto-renewing or extending",
    "Notice Period To Terminate Renewal": "Notice deadline required to prevent automatic contract renewal",
    "Governing Law": "Jurisdiction, state, or federal law governing dispute resolution",
    "Most Favored Nation": "Covenant ensuring terms are as favorable as those granted to other clients",
    "Non-Compete": "Restriction prohibiting a party from engaging in competitive business",
    "Exclusivity": "Exclusive dealing, sole supplier, or exclusive distributor requirements",
    "No-Solicit Of Customers": "Restriction preventing solicitation or poaching of customer base",
    "No-Solicit Of Employees": "Restriction preventing hiring or soliciting counterpart employees",
    "Non-Disparagement": "Prohibition against making derogatory or disparaging statements",
    "Termination For Convenience": "Right of a party to terminate without breach or specific cause",
    "Termination For Cause": "Provisions allowing termination due to default, material breach, or bankruptcy",
    "RoFR/RoFO/RoFN": "Right of First Refusal, First Offer, or First Negotiation rights",
    "Change Of Control": "Consequences or termination triggers if a party changes ownership/merges",
    "Anti-Assignment": "Restrictions on transferring, selling, or delegating contractual rights/obligations",
    "Cap On Liability": "Dollar limit, multiple of fees, or ceiling capping maximum liability",
    "Uncapped Liability": "Carve-outs or situations where liability limits do not apply",
    "Liquidated Damages": "Pre-estimated monetary damages payable upon specified contractual breach",
    "Warranty Duration": "Timeframe and scope of warranties provided on goods, software, or services",
    "Insurance": "Mandatory coverage types and minimal policy amounts",
    "Indemnity": "Obligation to defend, hold harmless, and indemnify against third-party claims",
    "IP Ownership Assignment": "Provisions defining assignment of work product, inventions, or IP",
    "Jointly Owned IP": "Clauses regulating co-developed or jointly owned intellectual property",
    "License Grant": "Scope, territory, exclusivity, and nature of rights granted to IP",
    "Non-Transferable License": "Specification that the granted license cannot be sublicensed or assigned",
    "Affiliate License Grant": "Clause extending software or IP license to company subsidiaries/affiliates",
    "Irrevocable License": "Terms stating the license cannot be unilaterally rescinded or cancelled",
    "Source Code Escrow": "Deposit of source code with third-party escrow agent upon trigger event",
    "Post-Termination Services": "Transition assistance, migration support, or wind-down duties",
    "Audit Rights": "Right to inspect financial records, security controls, or contract compliance",
    "Revenue / Profit Sharing": "Percentages, royalties, or fee splits on generated revenue",
    "Volume Restriction": "Limits or minimum quotas on transaction volume or usage metrics",
    "Minimum Commitment": "Guaranteed minimum payments, purchase volumes, or spend commitments",
    "Price Restriction": "Clauses restricting price hikes, indexing inflation caps, or fee modifications",
    "Confidentiality": "Obligations to protect proprietary information and trade secrets",
    "Force Majeure": "Excuses performance due to acts of God, strikes, war, or natural disasters",
    "Dispute Resolution": "Arbitration, mediation, escalation, and venue requirements",
}

INSURANCE_CATEGORIES = {
    "Deductible and Out-of-Pocket": "Out-of-pocket amount the insured must pay before coverage applies",
    "Coverage Limit": "Maximum dollar aggregate or per-incident payout by insurer",
    "Exclusions": "Events, conditions, damages, or perils explicitly excluded from coverage",
    "Premium and Payment Schedule": "Cost of insurance, installment intervals, and grace period",
    "Claim Notice and Proof of Loss": "Deadlines and procedures required to file and validate claims",
    "Conditions Precedent": "Requirements that must be satisfied for insurance coverage to remain in force",
    "Subrogation": "Insurer's right to pursue third parties responsible for the covered loss",
    "Cancellation and Non-Renewal": "Terms under which insurer or policyholder may cancel or refuse renewal",
    "Waiting Period / Pre-existing Conditions": "Moratorium period before certain benefits become eligible",
    "Endorsements and Riders": "Amendments expanding, altering, or limiting base policy coverage",
}

SUPPORTED_PROVIDERS = ['offline', 'gemini', 'groq', 'openai', 'ollama']
DEFAULT_PROVIDER = 'offline'
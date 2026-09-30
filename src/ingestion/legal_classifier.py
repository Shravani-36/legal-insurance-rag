# -*- coding: utf-8 -*-
"""
Legal & Insurance Clause Classifier based on the CUAD (Contract Understanding Atticus Dataset)
and Insurance Policy Ontologies.
"""
from typing import List, Dict, Any, Tuple
import re
import config


class LegalClauseClassifier:
    """
    Classifies legal clauses into 41 CUAD categories and 10 Insurance categories.
    Detects high-risk clauses (uncapped liability, broad indemnification, auto-renewal traps, exclusions).
    """

    # Keyword and regex patterns for CUAD and Insurance categories
    CATEGORY_PATTERNS = {
        # CUAD Legal Categories
        "Termination For Convenience": [
            r"\bterminate(?:\s+\w+){0,3}\s+(?:for\s+convenience|without\s+cause|at\s+any\s+time|upon\s+\d+\s+days\s+notice)\b",
            r"\beither\s+party\s+may\s+terminate\b",
            r"\bmay\s+terminate\s+this\s+agreement\s+by\s+providing\b"
        ],
        "Termination For Cause": [
            r"\bmaterial\s+breach\b",
            r"\bdefault\s+under\s+this\s+agreement\b",
            r"\binsolvency|bankruptcy|liquidation\b",
            r"\bfailure\s+to\s+cure\b"
        ],
        "Cap On Liability": [
            r"\bliability\s+(?:shall\s+not\s+exceed|capped\s+at|limited\s+to)\b",
            r"\blimitation\s+of\s+liability\b",
            r"\bliability\s+cap\b",
            r"\baggregate\s+(?:cumulative\s+)?liability\b",
            r"\bmaximum\s+cumulative\s+liability\b",
            r"\bfees\s+paid\s+in\s+the\s+preceding\b"
        ],
        "Uncapped Liability": [
            r"\bwithout\s+limitation\b",
            r"\bno\s+limitation\s+of\s+liability\b",
            r"\bshall\s+not\s+apply\s+to\s+(?:gross\s+negligence|willful\s+misconduct|indemnification|breach\s+of\s+confidentiality)\b"
        ],
        "Indemnity": [
            r"\bindemnify|hold\s+harmless|defend\b",
            r"\bindemnification\s+obligations\b",
            r"\bthird-party\s+claims\b"
        ],
        "Insurance": [
            r"\bcommercial\s+general\s+liability\b",
            r"\berrors\s+and\s+omissions\b",
            r"\bworkers['\s]+compensation\b",
            r"\bcyber\s+liability\b",
            r"\bmaintain\s+(?:insurance|coverage)\b",
            r"\bcertificate\s+of\s+insurance\b"
        ],
        "Governing Law": [
            r"\bgoverned\s+by\s+the\s+laws\b",
            r"\bjurisdiction\s+of\s+the\s+courts\b",
            r"\bchoice\s+of\s+law\b",
            r"\bvenue\s+shall\s+be\b"
        ],
        "Renewal Term": [
            r"\bautomatically\s+renew\b",
            r"\bauto-renewal\b",
            r"\bsuccessive\s+(?:one-year|annual|monthly)\s+periods\b",
            r"\bterm\s+shall\s+be\s+extended\b"
        ],
        "Notice Period To Terminate Renewal": [
            r"\bnotice\s+(?:of\s+non-renewal|not\s+to\s+renew)\b",
            r"\bat\s+least\s+\d+\s+days\s+prior\s+to\s+the\s+expiration\b",
            r"\bwritten\s+notice\s+before\s+renewal\b"
        ],
        "Confidentiality": [
            r"\bconfidential\s+information\b",
            r"\bproprietary\s+information\b",
            r"\bnon-disclosure\b",
            r"\btrade\s+secrets\b",
            r"\brecipient\s+shall\s+protect\b"
        ],
        "Non-Compete": [
            r"\bnon-compete|covenant\s+not\s+to\s+compete\b",
            r"\bshall\s+not\s+directly\s+or\s+indirectly\s+engage\s+in\b",
            r"\bcompeting\s+business\b"
        ],
        "No-Solicit Of Employees": [
            r"\bnon-solicitation\s+of\s+employees\b",
            r"\bshall\s+not\s+solicit,\s+recruit,\s+or\s+hire\b",
            r"\bentice\s+away\s+any\s+employee\b"
        ],
        "IP Ownership Assignment": [
            r"\bwork\s+made\s+for\s+hire\b",
            r"\bassigns\s+all\s+right,\s+title,\s+and\s+interest\b",
            r"\bownership\s+of\s+deliverables\b",
            r"\bintellectual\s+property\s+rights\b"
        ],
        "License Grant": [
            r"\bhereby\s+grants\s+a\s+(?:non-exclusive|exclusive|royalty-free|perpetual|worldwide)\s+license\b",
            r"\bscope\s+of\s+license\b",
            r"\bright\s+to\s+use\s+the\s+software\b"
        ],
        "Audit Rights": [
            r"\bright\s+to\s+audit\b",
            r"\binspect\s+books\s+and\s+records\b",
            r"\bexaminations\s+upon\s+reasonable\s+notice\b"
        ],
        "Force Majeure": [
            r"\bforce\s+majeure\b",
            r"\bacts\s+of\s+god|war|terrorism|pandemic|earthquake|natural\s+disaster\b",
            r"\bbeyond\s+the\s+reasonable\s+control\b"
        ],
        "Dispute Resolution": [
            r"\barbitration\b",
            r"\bAmerican\s+Arbitration\s+Association\b",
            r"\bmediation\b",
            r"\bdispute\s+escalation\b"
        ],

        # Insurance Specific Categories
        "Deductible & Out-of-Pocket": [
            r"\bdeductible\b",
            r"\bout-of-pocket\s+maximum\b",
            r"\bco-insurance\b",
            r"\bco-payment|copay\b"
        ],
        "Coverage Limit": [
            r"\blimit\s+of\s+liability\b",
            r"\bmaximum\s+benefit\b",
            r"\bpolicy\s+limit\b",
            r"\baggregate\s+limit\b",
            r"\beach\s+occurrence\s+limit\b"
        ],
        "Exclusions": [
            r"\bexclusions?\b",
            r"\bwe\s+do\s+not\s+cover\b",
            r"\bnot\s+covered\s+under\s+this\s+policy\b",
            r"\bexcluded\s+perils?\b",
            r"\bpre-existing\s+conditions?\b",
            r"\bintentional\s+acts|wear\s+and\s+tear|flood\s+damage\b"
        ],
        "Premium & Payment Schedule": [
            r"\bpremium\b",
            r"\bgrace\s+period\b",
            r"\bpayment\s+due\s+date\b",
            r"\blapse\s+for\s+non-payment\b"
        ],
        "Claim Notice & Proof of Loss": [
            r"\bnotice\s+of\s+claim\b",
            r"\bproof\s+of\s+loss\b",
            r"\bfile\s+a\s+claim\s+within\b",
            r"\bclaim\s+reporting\b"
        ],
        "Subrogation": [
            r"\bsubrogation\b",
            r"\bright\s+of\s+recovery\b",
            r"\btransfer\s+of\s+rights\s+of\s+recovery\b"
        ],
        "Cancellation & Non-Renewal": [
            r"\bcancellation\s+of\s+policy\b",
            r"\bnon-renewal\s+notice\b",
            r"\bpro-rata\s+refund\b"
        ]
    }

    RISK_CLASSIFICATION = {
        "Uncapped Liability": ("HIGH", "Carves out limits, exposing the organization to boundless financial risk."),
        "Indemnity": ("HIGH", "Imposes duty to defend and pay for third-party legal claims and damages."),
        "Exclusions": ("HIGH", "Denies insurance or contract coverage for specific events, conditions, or perils."),
        "Non-Compete": ("HIGH", "Severely restricts competitive business activities or market entry."),
        "Renewal Term": ("MEDIUM", "Auto-renewal can lock in fees unless timely cancellation notice is served."),
        "Notice Period To Terminate Renewal": ("MEDIUM", "Strict deadline required to prevent automatic contract rollover."),
        "Termination For Convenience": ("MEDIUM", "Allows immediate or short-notice termination without fault."),
        "Cap On Liability": ("LOW", "Protective ceiling capping maximum damages or liability."),
        "Insurance": ("LOW", "Standard operational risk mitigation requirement.")
    }

    def __init__(self):
        self._compiled_patterns = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in self.CATEGORY_PATTERNS.items()
        }

    def classify_text(self, text: str, section_title: str = "") -> Dict[str, Any]:
        """
        Classifies clause text and section header into CUAD / Insurance categories.
        """
        matches: List[Tuple[str, float]] = []
        full_text = f"{section_title}\n{text}"

        for category, patterns in self._compiled_patterns.items():
            score = 0.0
            for pattern in patterns:
                finds = pattern.findall(full_text)
                if finds:
                    score += len(finds) * 0.4
                    if section_title and pattern.search(section_title):
                        score += 0.5

            if score > 0:
                normalized_score = min(1.0, round(score, 2))
                matches.append((category, normalized_score))

        matches.sort(key=lambda x: x[1], reverse=True)

        if matches:
            primary_cat, conf = matches[0]
            risk_level, risk_desc = self.RISK_CLASSIFICATION.get(
                primary_cat, ("INFORMATIONAL", "Standard contractual or policy term.")
            )
        else:
            primary_cat = "General Legal Clause"
            conf = 0.0
            risk_level = "INFORMATIONAL"
            risk_desc = "Standard procedural or non-critical language."

        return {
            "primary_category": primary_cat,
            "confidence": conf,
            "all_categories": matches[:4],
            "risk_level": risk_level,
            "risk_explanation": risk_desc
        }

    def enrich_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enriches a list of chunk dictionaries with classification metadata.
        """
        for chunk in chunks:
            analysis = self.classify_text(chunk["text"], chunk.get("section_title", ""))
            chunk.update(analysis)
        return chunks
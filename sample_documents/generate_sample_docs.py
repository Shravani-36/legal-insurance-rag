# -*- coding: utf-8 -*-
"""
Generates realistic sample legal contracts and insurance policies in PDF format
using ReportLab for testing and demonstration purposes.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

SAMPLE_DIR = Path(__file__).resolve().parent

def build_pdf(filepath: Path, title: str, sections: list):
    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=14,
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=18,
        alignment=1
    )
    h1_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=14,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'LegalBody',
        parent=styles['BodyText'],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=8
    )

    story = [
        Paragraph(title, title_style),
        Paragraph("CONFIDENTIAL & LEGALLY BINDING DOCUMENT", subtitle_style),
        Spacer(1, 10)
    ]

    for item in sections:
        if item.get("page_break"):
            story.append(PageBreak())
            continue
        if item.get("header"):
            story.append(Paragraph(item["header"], h1_style))
        if item.get("text"):
            story.append(Paragraph(item["text"], body_style))

    doc.build(story)
    print(f"Generated: {filepath.name}")

def generate_vendor_agreement():
    sections = [
        {
            "header": "RECITALS AND PARTIES",
            "text": "This Master Vendor Services Agreement (the 'Agreement') is entered into as of January 15, 2026 (the 'Effective Date'), by and between Apex Enterprise Solutions Inc., a Delaware corporation ('Customer'), and CloudSphere Systems LLC, a California limited liability company ('Vendor'). Customer and Vendor are individually referred to as a 'Party' and collectively as the 'Parties'."
        },
        {
            "header": "SECTION 1 -- SCOPE OF SERVICES AND DELIVERABLES",
            "text": "1.1 Services. Vendor shall perform the software integration, cloud migration, and data management services set forth in one or more Statements of Work ('SOW') executed pursuant to this Agreement. 1.2 Performance Standards. Vendor warrants that all services will be executed in a professional, workmanlike manner conforming to industry best practices and defined specifications."
        },
        {
            "header": "SECTION 2 -- PAYMENT TERMS AND INVOICING",
            "text": "2.1 Fees. Customer agrees to pay all undisputed invoices within forty-five (45) calendar days of receipt. 2.2 Late Payments. Undisputed past due balances shall accrue interest at the lesser of one percent (1.0%) per month or the highest rate permitted by applicable law. All fees are exclusive of applicable state and federal taxes."
        },
        {
            "header": "SECTION 3 -- INTELLECTUAL PROPERTY AND ASSIGNMENTS",
            "text": "3.1 Work Made For Hire. All custom deliverables, documentation, source code modifications, and inventions conceived or created by Vendor exclusively for Customer shall be deemed 'work made for hire' owned exclusively by Customer. 3.2 Pre-Existing IP. Vendor retains sole ownership of its pre-existing tools, libraries, and background technology."
        },
        {"page_break": True},
        {
            "header": "SECTION 4 -- CONFIDENTIALITY AND TRADE SECRETS",
            "text": "4.1 Non-Disclosure. Each Party agrees to protect the Confidential Information of the disclosing Party with the same degree of care it uses for its own proprietary information, but no less than reasonable care. Confidential Information shall not be disclosed to any third party without prior written consent for a period of five (5) years following termination."
        },
        {
            "header": "SECTION 5 -- NON-SOLICITATION",
            "text": "5.1 Employees. During the term of this Agreement and for a period of twelve (12) months following the date of expiration or termination, neither Party shall directly or indirectly solicit, recruit, entice away, or hire any personnel or contractor of the other Party who was involved in the performance of services under this Agreement without express written approval."
        },
        {
            "header": "SECTION 6 -- LIMITATION OF LIABILITY",
            "text": "6.1 Liability Cap. In no event shall either Party's aggregate cumulative liability arising out of or related to this Agreement exceed the total fees paid by Customer to Vendor in the twelve (12) months immediately preceding the incident giving rise to liability, not to exceed $1,000,000. 6.2 Exclusions from Liability Cap. The limitations in Section 6.1 shall not apply to breach of Section 4 (Confidentiality), gross negligence, or indemnification obligations under Section 9."
        },
        {
            "header": "SECTION 7 -- TERMINATION",
            "text": "7.1 Termination for Convenience. Either Party may terminate this Agreement or any active SOW for convenience at any time without cause by providing thirty (30) days prior written notice to the other Party. 7.2 Termination for Cause. Either Party may terminate immediately upon written notice if the other Party commits a material breach and fails to cure such breach within fifteen (15) days of receiving written notice thereof."
        },
        {"page_break": True},
        {
            "header": "SECTION 8 -- INSURANCE REQUIREMENTS",
            "text": "8.1 Coverage. Vendor shall at all times during the term maintain: (a) Commercial General Liability insurance with limits not less than $2,000,000 per occurrence and $4,000,000 general aggregate; (b) Professional Errors and Omissions (E&O) and Cyber Liability insurance of at least $1,000,000 per claim; and (c) Statutory Workers' Compensation. Vendor shall name Customer as an additional insured upon request."
        },
        {
            "header": "SECTION 9 -- INDEMNIFICATION",
            "text": "9.1 Vendor Indemnity. Vendor agrees to defend, indemnify, and hold harmless Customer, its officers, directors, and employees against any third-party claims, liabilities, losses, or legal expenses arising from allegations that Vendor deliverables infringe any patent, copyright, or trademark of a third party."
        },
        {
            "header": "SECTION 10 -- GOVERNING LAW & DISPUTE RESOLUTION",
            "text": "10.1 Governing Law. This Agreement shall be construed and enforced in accordance with the laws of the State of Delaware, without regard to conflicts of law principles. 10.2 Arbitration. Any controversy or claim arising out of this Agreement shall be settled by final and binding arbitration administered by the American Arbitration Association (AAA) in Wilmington, Delaware."
        }
    ]
    build_pdf(SAMPLE_DIR / "Commercial_Vendor_Agreement.pdf", "MASTER VENDOR SERVICES AGREEMENT", sections)

def generate_insurance_policy():
    sections = [
        {
            "header": "POLICY DECLARATIONS & GENERAL TERMS",
            "text": "Policy Number: POL-2026-HLT-99214. Insurer: Guardian Beacon Life & Casualty Assurance Co. Named Insured: Horizon Logistics Corp and eligible dependents. Policy Term: January 1, 2026 to December 31, 2026. This comprehensive medical and hospitalization policy provides healthcare benefits subject to the terms, limitations, exclusions, and cost-sharing schedules detailed herein."
        },
        {
            "header": "SCHEDULE B -- DEDUCTIBLES & COST SHARING",
            "text": "1. Annual Deductible: The individual calendar year deductible is $1,500. The family aggregate deductible is $3,000. Covered expenses must satisfy the annual deductible before insurance benefits become payable. 2. Maximum Out-of-Pocket: The maximum annual out-of-pocket limit is $6,000 for individual coverage and $12,000 for family coverage. Once reached, eligible medical expenses are covered at 100% for the remainder of the calendar year."
        },
        {
            "header": "SECTION 1 -- COVERED MEDICAL BENEFITS",
            "text": "1.1 Hospitalization. Inpatient room and board, intensive care unit charges, operating room expenses, and inpatient physician visits are covered at 80% co-insurance after the deductible. 1.2 Outpatient Services. Diagnostic laboratory tests, imaging (MRI, CT scans, X-rays), and specialist office visits are covered subject to a $40 co-payment per visit."
        },
        {"page_break": True},
        {
            "header": "SECTION 3 -- EMERGENCY & AMBULANCE SERVICES",
            "text": "3.1 Emergency Care. True medical emergencies are covered at in-network benefit levels regardless of hospital network affiliation, subject to a $250 emergency room co-pay. Co-payment is waived if the insured is directly admitted to the hospital as an inpatient."
        },
        {
            "header": "SECTION 4 -- POLICY EXCLUSIONS",
            "text": "4.1 Non-Covered Perils and Conditions. This policy does not cover: (a) Cosmetic surgery or elective procedures not medically necessary; (b) Experimental, investigational, or unproven medical treatments; (c) Injuries resulting from declared war, military action, or civil insurrection; (d) Damages or losses resulting from flood, earthquake, surface water, or environmental contamination; (e) Services obtained without required prior authorization."
        },
        {
            "header": "SECTION 5 -- CLAIMS FILING AND NOTICE REQUIREMENTS",
            "text": "5.1 Notice of Claim. Written notice of claim must be submitted to the Insurer within thirty (30) days after the occurrence or commencement of any loss covered by the policy. 5.2 Proof of Loss. Full formal written proof of loss, including itemized medical bills, diagnosis codes, and physician notes, must be furnished to the claims department within sixty (60) days after the date of loss."
        },
        {"page_break": True},
        {
            "header": "SECTION 6 -- SUBROGATION AND REIMBURSEMENT",
            "text": "6.1 Subrogation Rights. If the Insurer makes payment under this policy and the covered person has a right to recover damages from a negligent third party, the Insurer is subrogated to all rights of recovery to the full extent of benefits paid. The insured shall cooperate fully and execute all legal documents required to protect the Insurer's subrogation interests."
        },
        {
            "header": "SECTION 7 -- CANCELLATION AND GRACE PERIOD",
            "text": "7.1 Premium Grace Period. A grace period of thirty (30) calendar days will be granted for the payment of each premium falling due after the first premium. During this grace period, the policy remains in full force. If payment is not received prior to the expiration of the 30-day grace period, coverage will terminate automatically effective on the due date."
        }
    ]
    build_pdf(SAMPLE_DIR / "Comprehensive_Health_Insurance_Policy.pdf", "COMPREHENSIVE HEALTH & CASUALTY INSURANCE POLICY", sections)

if __name__ == "__main__":
    generate_vendor_agreement()
    generate_insurance_policy()
    print("All sample PDF documents generated successfully.")

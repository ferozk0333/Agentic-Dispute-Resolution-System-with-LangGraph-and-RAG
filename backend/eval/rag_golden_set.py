"""
Golden evaluation set — 20 query → expected chunk pairs.
Derived from Visa Core Rules and Visa Product and Service Rules, 18 April 2026 edition.
"""

RAG_GOLDEN_SET = [

    # ── Dispute time limits ───────────────────────────────────────────────

    {
        "id": "rag_001",
        "query": "dispute condition 10.4 card absent environment time limit",
        "expected_section": "11.7.5.4",
        "expected_text_fragment": "120 calendar days from the Transaction Processing Date",
        "page": 705,
        "notes": "Core time window check for unauthorized CNP disputes. "
                 "Table 11-29. Applies to all regions.",
    },
    {
        "id": "rag_002",
        "query": "reason code 10.4 unauthorized transaction card not present",
        "expected_section": "11.7.5.1",
        "expected_text_fragment": "Cardholder denies authorization of or participation in a "
                                  "Transaction conducted in a Card-Absent Environment",
        "page": 698,
        "notes": "Dispute reason definition for 10.4. Table 11-26.",
    },
    {
        "id": "rag_003",
        "query": "dispute condition 10.3 card present fraud time limit 120 days",
        "expected_section": "11.7.4.4",
        "expected_text_fragment": "120 calendar days from the Transaction Processing Date",
        "page": 695,
        "notes": "Card-present fraud time limit. Table 11-23.",
    },
    {
        "id": "rag_004",
        "query": "dispute condition 13.1 merchandise not received time limit",
        "expected_section": "11.10.2.4",
        "expected_text_fragment": "120 calendar days",
        "page": 747,
        "notes": "Not-received dispute time limit. Table 11-92.",
    },
    {
        "id": "rag_005",
        "query": "dispute condition 12.5 incorrect amount time limit",
        "expected_section": "11.9.4",
        "expected_text_fragment": "120 calendar days from either",
        "page": 733,
        "notes": "Incorrect amount dispute. Table 11-74.",
    },
    {
        "id": "rag_006",
        "query": "dispute condition 12.6 duplicate processing paid by other means dispute time limit",
        "expected_section": "11.9.5.4",
        "expected_text_fragment": "120 calendar days from either",
        "page": 738,
        "notes": "Duplicate dispute time limit. Table 11-80 on page 738. "
                 "Section is §11.9.5.4, not the parent §11.9.5. "
                 "Query must include '12.6' and 'time limit' to target the sub-section.",
    },
    {
        "id": "rag_007",
        "query": "card recovery bulletin dispute condition 11.1 time limit 75 days",
        "expected_section": "11.8.1.3",
        "expected_text_fragment": "75 calendar days from the Transaction Processing Date",
        "page": 712,
        "notes": "Shorter 75-day window for CRB disputes. Table 11-38.",
    },

    # ── Dispute rights and conditions ─────────────────────────────────────

    {
        "id": "rag_008",
        "query": "issuer must attempt to settle before initiating dispute",
        "expected_section": "1.10.1.1",
        "expected_text_fragment": "Before initiating a Dispute, the Issuer must attempt to honor the Transaction",
        "page": 147,
        "notes": "Core rule: attempt to settle first. ID# 0003287.",
    },
    {
        "id": "rag_009",
        "query": "cardholder must not be credited twice same transaction dispute",
        "expected_section": "1.10.1.1",
        "expected_text_fragment": "Issuer must not be reimbursed twice for the same Transaction",
        "page": 147,
        "notes": "Anti-double-credit rule. Critical for duplicate dispute detection.",
    },
    {
        "id": "rag_010",
        "query": "dispute invalid 10.4 more than 35 disputes same account 120 days",
        "expected_section": "11.7.5.3",
        "expected_text_fragment": "Transaction on an Account Number for which the Issuer has initiated "
                                  "more than 35 Disputes within the previous 120 calendar days",
        "page": 699,
        "notes": "High-velocity dispute abuse check. Table 11-28. Important fraud signal.",
    },
    {
        "id": "rag_011",
        "query": "10.4 invalid dispute cryptocurrency non-fungible token NFT cardholder deceived",
        "expected_section": "11.7.5.3",
        "expected_text_fragment": "Transaction for the acquisition of non-fiat currency",
        "page": 700,
        "notes": "Crypto/NFT carve-out from 10.4 dispute rights. Table 11-28.",
    },
    {
        "id": "rag_012",
        "query": "compelling evidence card absent environment pre-arbitration",
        "expected_section": "11.5.1",
        "expected_text_fragment": "Acquirer may submit Compelling Evidence with a pre-Arbitration attempt",
        "page": 677,
        "notes": "Compelling evidence rules. Table 11-6. Relevant to reject decisions.",
    },
    {
        "id": "rag_013",
        "query": "EMV liability shift counterfeit fraud dispute condition 10.1",
        "expected_section": "11.7.2",
        "expected_text_fragment": "EMV Liability Shift Counterfeit Fraud",
        "page": 688,
        "notes": "10.1 dispute condition definition.",
    },

    # ── Dispute amounts ───────────────────────────────────────────────────

    {
        "id": "rag_014",
        "query": "minimum dispute amount T&E transactions USD 25",
        "expected_section": "11.4.3",
        "expected_text_fragment": "USD 25 (or local currency equivalent)",
        "page": 676,
        "notes": "Minimum dispute threshold. Table 11-5.",
    },
    {
        "id": "rag_015",
        "query": "dispute amount must not exceed transaction amount",
        "expected_section": "11.4.1",
        "expected_text_fragment": "Dispute amount must not exceed the Transaction amount",
        "page": 675,
        "notes": "Dispute amount ceiling. ID# 0030217.",
    },

    # ── Compliance and arbitration ────────────────────────────────────────

    {
        "id": "rag_016",
        "query": "arbitration compliance decision financial liability visa",
        "expected_section": "1.10.2.3",
        "expected_text_fragment": "responsible Member is financially liable",
        "page": 149,
        "notes": "Financial liability assignment. ID# 0003623.",
    },
    {
        "id": "rag_017",
        "query": "non-compliance assessment tier 1 violation USD 25000",
        "expected_section": "1.11.2.2",
        "expected_text_fragment": "Level 1 non-compliance assessment of USD 25,000",
        "page": 152,
        "notes": "Tier 1 non-compliance schedule. Table 1-13.",
    },
    {
        "id": "rag_018",
        "query": "enforcement appeal member 30 days new evidence violation",
        "expected_section": "1.11.2.7",
        "expected_text_fragment": "appeal letter must be received by Visa within 30 calendar days",
        "page": 157,
        "notes": "Appeal window for compliance violations. ID# 0025975.",
    },

    # ── Dispute processing requirements ──────────────────────────────────

    {
        "id": "rag_019",
        "query": "dispute 10.4 processing requirements cardholder certification denies authorization",
        "expected_section": "11.7.5.5",
        "expected_text_fragment": "Certification that the Cardholder denies authorization of or "
                                  "participation in the",
        "page": 705,
        "notes": "Required documentation for 10.4 disputes. Table 11-30. "
                 "Fragment trimmed: PDF table splits 'Transaction' into next row.",
    },
    {
        "id": "rag_020",
        "query": "13.1 not received processing requirements cardholder attempted resolve merchant",
        "expected_section": "11.10.2.5",
        "expected_text_fragment": "Cardholder attempted to resolve with Merchant",
        "page": 748,
        "notes": "Required documentation for 13.1 disputes. Table 11-93 (page 749). "
                 "PDF text is 'Cardholder attempted to resolve with Merchant' — no 'the Dispute'.",
    },
    

    {
        "id": "rag_021",
        "query": "dispute condition 10.4 card absent cardholder denies authorization reason",
        "expected_section": "11.7.5.1",
        "expected_text_fragment": "Cardholder denies authorization of or participation in a Transaction conducted in a Card-Absent Environment",
        "page": 698,
        "notes": "Core reason definition for 10.4. Table 11-26. Should be top-1 hit — "
                "exact match to the most common dispute type in the system.",
    },
    {
        "id": "rag_022",
        "query": "issuer must report fraud activity before initiating dispute 10.4",
        "expected_section": "11.7.5.2",
        "expected_text_fragment": "Before initiating a Dispute, an Issuer must report the Fraud Activity to Visa",
        "page": 698,
        "notes": "Dispute rights prerequisite for 10.4. Table 11-27. "
                "Short, distinctive phrase — should be easy Hit@1.",
    },
    {
        "id": "rag_023",
        "query": "dispute amount must not exceed transaction amount partial dispute",
        "expected_section": "11.4.1",
        "expected_text_fragment": "Dispute amount must not exceed the Transaction amount",
        "page": 675,
        "notes": "Amount ceiling rule. ID# 0030217. Short standalone sentence — "
                "very easy retrieval, should be Hit@1.",
    },
    {
        "id": "rag_024",
        "query": "cardholder financial loss required before issuer processes dispute",
        "expected_section": "1.10.1.1",
        "expected_text_fragment": "Issuer must not process a Dispute unless the Cardholder has suffered a financial loss",
        "page": 148,
        "notes": "Core rule ID# 0003287. Section §1.10.1.1 starts p.147; financial-loss sentence "
                "appears on p.148 continuation. Fundamental eligibility gate.",
    },
    {
        "id": "rag_025",
        "query": "cancelled recurring transaction dispute condition 13.2 time limit 120 days",
        "expected_section": "11.10.3",
        "expected_text_fragment": "120 calendar days from the Transaction Processing Date",
        "page": 755,
        "notes": "Recurring charge dispute. Table 11-99. "
                "Common scenario — subscription cancellations.",
    },
    {
        "id": "rag_026",
        "query": "dispute condition 10.5 visa fraud monitoring program time limit",
        "expected_section": "11.7.6.3",
        "expected_text_fragment": "120 calendar days from the date of the Visa Fraud Monitoring Program report",
        "page": 710,
        "notes": "10.5 time limit. Table 11-34 on page 710 (§11.7.6.3 heading starts p.710; "
                "§11.7.6.1-2 are on p.709). Distinctive phrase — "
                "different trigger date from other 120-day windows.",
    },
    {
        "id": "rag_027",
        "query": "arbitration compliance decision based on visa rules transaction date",
        "expected_section": "1.10.2.2",
        "expected_text_fragment": "Visa bases its Arbitration or Compliance decision on all information available",
        "page": 149,
        "notes": "Arbitration decision standard. ID# 0027133. "
                "Short section, clear language — easy retrieval.",
    },
    {
        "id": "rag_028",
        "query": "EMV liability shift counterfeit fraud card present environment regions",
        "expected_section": "1.10.1.2",
        "expected_text_fragment": "EMV liability shift applies to qualifying Transactions",
        "page": 148,
        "notes": "EMV liability shift participation table. ID# 0008190. "
                "Relevant to card-present disputes and liability assignment.",
    },
]

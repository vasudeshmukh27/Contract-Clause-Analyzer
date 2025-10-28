# clause_detector.py
# Purpose: Classify "Unknown" spans into 8 target clause types using Groq API

import os
import json
import re
import psycopg2
from typing import List, Dict, Tuple
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "contract_analyzer"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "YourPwd123"),
}

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is required. Get it from https://console.groq.com/keys")
groq_client = Groq(api_key=GROQ_API_KEY)

# Target clause types
TARGET_CLAUSES = [
    "Data Sharing",
    "Termination for Convenience",
    "Notice to Terminate",
    "Governing Law",
    "Cap on Liability",
    "Auto Renewal",
    "Audit Rights",
    "Indemnification",
]

# Clause type definitions for rule-based detection
CLAUSE_DEFINITIONS = {
    "Data Sharing": {
        "keywords": ["data", "information", "confidential", "share", "disclose", "personal", "privacy", "processing"],
        "heading_patterns": ["data", "confidentiality", "information"],
    },
    "Termination for Convenience": {
        "keywords": ["terminate", "convenience", "without cause", "at will", "any reason", "termination"],
        "heading_patterns": ["termination", "convenience", "discontinuance"],
    },
    "Notice to Terminate": {
        "keywords": ["notice", "days", "written notice", "advance notice", "notify", "terminate"],
        "heading_patterns": ["notice", "termination"],
    },
    "Governing Law": {
        "keywords": ["governing law", "jurisdiction", "laws of", "courts", "state law", "applicable law"],
        "heading_patterns": ["governing", "jurisdiction", "law"],
    },
    "Cap on Liability": {
        "keywords": ["liability", "limit", "exceed", "damages", "cap", "maximum", "not liable"],
        "heading_patterns": ["liability", "cap", "limit", "damages"],
    },
    "Auto Renewal": {
        "keywords": ["renew", "automatic", "successive", "term", "renewal", "continues"],
        "heading_patterns": ["renewal", "renewal term", "automatic"],
    },
    "Audit Rights": {
        "keywords": ["audit", "inspect", "examine", "records", "books", "access", "review"],
        "heading_patterns": ["audit", "inspection", "review"],
    },
    "Indemnification": {
        "keywords": ["indemnify", "hold harmless", "defend", "indemnification", "indemnified"],
        "heading_patterns": ["indemnification", "indemnify", "hold harmless"],
    },
}

def detect_clause_candidates(span_text: str, heading: str) -> List[Tuple[str, int]]:
    """Rule-based detection of clause type candidates"""
    text_lower = span_text.lower()
    heading_lower = heading.lower() if heading else ""
    candidates = []
    
    for clause_type, defn in CLAUSE_DEFINITIONS.items():
        kw_matches = sum(1 for kw in defn["keywords"] if kw in text_lower)
        hd_matches = sum(1 for hp in defn["heading_patterns"] if hp in heading_lower)
        score = hd_matches * 3 + kw_matches
        if score > 0:
            candidates.append((clause_type, score))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates

def extract_numeric_attributes(text: str) -> Dict[str, any]:
    """Extract numeric attributes from text (e.g., notice days)"""
    attrs = {}
    t = text.lower()
    
    # Extract notice days
    notice_match = re.search(r'(\d+)\s*(?:business\s+)?days?(?:\s+(?:prior|advance|written))?', t)
    if notice_match:
        attrs["notice_days"] = int(notice_match.group(1))
    
    # Detect termination fee
    if any(kw in t for kw in ["termination fee", "termination charge", "early termination", "penalty"]):
        attrs["termination_fee_required"] = True
    else:
        attrs["termination_fee_required"] = False
    
    # Detect convenience termination
    if any(kw in t for kw in ["convenience", "without cause", "at will", "for any reason"]):
        attrs["termination_for_convenience"] = True
    else:
        attrs["termination_for_convenience"] = False
    
    return attrs

def groq_classify_and_extract(span_text: str, heading: str, candidates: List[str]) -> Dict[str, any]:
    """Use Groq API to classify and extract attributes from span"""
    
    prompt = f"""Classify this contract text into ONE of: Data Sharing, Termination for Convenience, Notice to Terminate, Governing Law, Cap on Liability, Auto Renewal, Audit Rights, Indemnification, or Other.

Heading: "{heading}"
Content: "{span_text[:700]}"

Respond ONLY with this JSON (no markdown, no extra text):
{{"clause_type": "type name", "confidence": 0.8, "attributes": {{}}, "reasoning": "why"}}"""
    
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a legal analyst. Respond ONLY with valid JSON, no markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        
        txt = resp.choices[0].message.content.strip()
        
        # Simple extraction: find first { and last }
        start = txt.find('{')
        end = txt.rfind('}')
        
        if start == -1 or end == -1:
            print(f"No JSON found in response: {txt[:100]}")
            return {"clause_type": "Other", "confidence": 0.2, "attributes": {}, "reasoning": "No JSON in response"}
        
        json_str = txt[start:end+1]
        
        # Unescape if needed
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            # Try unescaping
            json_str = json_str.replace('\\"', '"').replace('\\\\', '\\')
            result = json.loads(json_str)
        
        ct = result.get("clause_type", "Other")
        if ct not in TARGET_CLAUSES + ["Other"]:
            result["clause_type"] = "Other"
        
        return result
        
    except Exception as e:
        print(f"Groq error: {e}")
        return {"clause_type": "Other", "confidence": 0.2, "attributes": {}, "reasoning": "API error"}

def classify_clause_span(span_id: int, span_text: str, heading: str) -> Dict[str, any]:
    """Classify a single span using rule-based + LLM approach"""
    cand_scores = detect_clause_candidates(span_text, heading)
    candidates = [c for c, _ in cand_scores[:3]]
    
    # Get rule-based attributes
    rule_attrs = extract_numeric_attributes(span_text)
    
    # Get LLM classification
    llm_res = groq_classify_and_extract(span_text, heading, candidates)
    
    # Merge attributes
    final_attrs = {**rule_attrs, **llm_res.get("attributes", {})}
    
    return {
        "span_id": span_id,
        "clause_type": llm_res.get("clause_type"),
        "confidence": llm_res.get("confidence", 0.5),
        "attributes": final_attrs,
        "reasoning": llm_res.get("reasoning", ""),
    }

def reset_processed_spans():
    """Reset all non-Unknown spans back to Unknown for reclassification"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("UPDATE clause_spans SET clause_type='Unknown' WHERE clause_type != 'Unknown'")
    conn.commit()
    cur.close()
    conn.close()
    print("Reset spans to Unknown")

def process_unknown_spans(limit: int = None, min_conf: float = 0.1):
    """Process Unknown spans and classify them"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    cur.execute(f"""
        SELECT span_id, text, heading
        FROM clause_spans
        WHERE clause_type = 'Unknown'
        ORDER BY span_id
        {limit_clause}
    """)
    rows = cur.fetchall()
    
    processed = skipped = 0
    
    for i, (sid, text, hd) in enumerate(rows, 1):
        if i % 50 == 0:
            print(f"Processing span {i}/{len(rows)}...")
        
        res = classify_clause_span(sid, text, hd or "")
        
        if res["confidence"] < min_conf:
            skipped += 1
            continue
        
        # Update database
        cur.execute("""
            UPDATE clause_spans
            SET clause_type=%s, attributes=%s::jsonb
            WHERE span_id=%s
        """, (res["clause_type"], json.dumps(res["attributes"]), sid))
        
        processed += 1
        
        # Commit every 10 spans
        if processed % 10 == 0:
            conn.commit()
    
    conn.commit()
    cur.close()
    conn.close()
    print(f"Processed: {processed}, Skipped: {skipped}")

def verify_results():
    """Display classification summary"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    print("\n=== Classification Summary ===")
    cur.execute("""
        SELECT clause_type, COUNT(*) as count
        FROM clause_spans 
        GROUP BY clause_type
        ORDER BY count DESC
    """)
    
    for ctype, cnt in cur.fetchall():
        print(f"{ctype:30}: {cnt}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    print("Resetting and re-processing spans with 8 target categories")
    reset_processed_spans()
    process_unknown_spans(limit=None, min_conf=0.1)
    verify_results()

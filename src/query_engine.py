import os
import psycopg2
import json
from typing import List, Dict, Any
from pgvector.psycopg2 import register_vector

# Load DB config from environment
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT",  "5432")),
    "database": os.getenv("DB_NAME", "contract_analyzer"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "YourPwd123"),
}

def get_data_sharing_contracts() -> List[Dict[str, Any]]:
    """
    Returns a list of contracts that have a Data Sharing clause,
    with vendor, title, page, heading, and excerpt.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("""
        SELECT d.vendor, d.title, cs.page, cs.heading,
               LEFT(cs.text, 200) AS excerpt
        FROM clause_spans cs
        JOIN documents d ON d.doc_id = cs.doc_id
        WHERE cs.clause_type = 'Data Sharing'
        ORDER BY d.vendor, d.title;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {"vendor": v, "title": t, "page": p, "heading": h, "excerpt": e}
        for v, t, p, h, e in rows
    ]

def compute_termination_score(attrs: Dict[str, Any]) -> float:
    """
    Termination flexibility score:
      S = 0.4*TFC + 0.3*(1 - min(notice_days/60,1))
          + 0.2*(1 - fee_penalty) + 0.1*(1 - auto_renew)
    """
    tfc = 1.0 if attrs.get("termination_for_convenience") else 0.0
    notice = attrs.get("notice_days", 60)
    fee_req = 1.0 if attrs.get("termination_fee_required") else 0.0
    auto_renew = 1.0 if attrs.get("auto_renewal") else 0.0

    score = (
        0.4 * tfc +
        0.3 * (1 - min(notice / 60, 1)) +
        0.2 * (1 - fee_req) +
        0.1 * (1 - auto_renew)
    )
    return round(score, 4)

def get_most_flexible_termination(top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Returns the top_k vendors/contracts ranked by termination flexibility
    with score and cited span.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    # Retrieve spans with attributes
    cur.execute("""
        SELECT d.vendor, d.title, cs.page, cs.heading, cs.attributes
        FROM clause_spans cs
        JOIN documents d ON d.doc_id = cs.doc_id
        WHERE cs.clause_type = 'Termination for Convenience'
    """)
    entries = cur.fetchall()
    cur.close()
    conn.close()

    # Compute scores
    scored = []
    for vendor, title, page, heading, attrs_json in entries:
        attrs = json.loads(attrs_json)
        score = compute_termination_score(attrs)
        excerpt = attrs.get("termination_for_convenience") and f"{heading} (page {page})" or ""
        scored.append({
            "vendor": vendor,
            "title": title,
            "page": page,
            "heading": heading,
            "attributes": attrs,
            "score": score
        })

    # Sort descending by score
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

if __name__ == "__main__":
    print("Contracts with Data Sharing clauses:")
    ds = get_data_sharing_contracts()
    for c in ds:
        print(f"- {c['vendor']} | {c['title']} | {c['heading']} (p.{c['page']})")

    print("\nTop flexible termination contracts:")
    flex = get_most_flexible_termination(top_k=5)
    for f in flex:
        print(f"- {f['vendor']} | {f['title']} | Score: {f['score']} | {f['heading']} (p.{f['page']})")
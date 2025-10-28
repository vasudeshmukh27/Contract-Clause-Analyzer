# app.py - Contract Clause Analyzer with PDF Upload and JSON Attribute Handling

import os
import sys
import json
import streamlit as st
import psycopg2
from pgvector.psycopg2 import register_vector
import subprocess
from pathlib import Path

# Load DB config
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "contract_analyzer"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "YourPwd123"),
}

st.set_page_config(page_title="Contract Clause Analyzer", layout="wide")

# Sidebar: Upload New Contract
st.sidebar.title("Contract Clause Analyzer")
st.sidebar.header("📁 Upload New Contract")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file", type="pdf",
    help="Upload a contract PDF to analyze clauses automatically"
)

if uploaded_file is not None:
    if st.sidebar.button("🔍 Analyze Contract"):
        with st.spinner("Processing contract..."):
            contracts_dir = Path("data/contracts")
            contracts_dir.mkdir(parents=True, exist_ok=True)

            file_path = contracts_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            vendor_name = file_path.stem
            title = uploaded_file.name

            try:
                st.info(f"📄 Parsing {uploaded_file.name}...")
                ingest_cmd = [
                    sys.executable,
                    "ingest_one.py",
                    "--pdf", str(file_path),
                    "--vendor", vendor_name,
                    "--title", title,
                ]
                result = subprocess.run(
                    ingest_cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    cwd="."
                )
                if result.returncode != 0:
                    st.error("❌ Error in parsing:")
                    st.code(result.stderr, language="bash")
                else:
                    st.info("🤖 Classifying clauses...")
                    classify_cmd = [sys.executable, "clause_detector.py"]
                    result2 = subprocess.run(
                        classify_cmd,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        cwd="."
                    )
                    if result2.returncode != 0:
                        st.error("❌ Error in classification:")
                        st.code(result2.stderr, language="bash")
                    else:
                        st.success(f"✅ Successfully processed {uploaded_file.name}!")
                        st.balloons()
                        st.rerun()
            except Exception as e:
                st.error(f"❌ Error processing contract: {e}")

# Sidebar: Query Clauses
st.sidebar.header("🔍 Query Clauses")
clause_type = st.sidebar.selectbox(
    "Select Clause Type",
    [
        "Data Sharing",
        "Termination for Convenience",
        "Notice to Terminate",
        "Governing Law",
        "Cap on Liability",
        "Auto Renewal",
        "Audit Rights",
        "Indemnification"
    ]
)

# Main Content
st.title("Contract Clause Analyzer")
st.markdown("Upload contracts and analyze key clauses for procurement decisions")

# Database query functions
def get_contracts_by_clause_type(clause_type):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT d.vendor, d.title, cs.page, cs.heading,
                   LEFT(cs.text, 200) AS excerpt, cs.attributes
            FROM clause_spans cs
            JOIN documents d ON d.doc_id = cs.doc_id
            WHERE cs.clause_type = %s
            ORDER BY d.vendor, d.title;
        """, (clause_type,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Database error: {e}")
        return []

def get_database_stats():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT doc_id) FROM clause_spans")
        total_contracts = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM clause_spans")
        total_spans = cur.fetchone()[0]
        cur.execute("""
            SELECT clause_type, COUNT(*) 
            FROM clause_spans 
            WHERE clause_type != 'Unknown'
            GROUP BY clause_type 
            ORDER BY COUNT(*) DESC
        """)
        clause_distribution = cur.fetchall()
        cur.close()
        conn.close()
        return total_contracts, total_spans, clause_distribution
    except Exception as e:
        st.error(f"Database error: {e}")
        return 0, 0, []

def get_termination_flexibility_score(attributes):
    if not attributes:
        return 0.0
    attrs = attributes if isinstance(attributes, dict) else json.loads(attributes)
    tfc = 1.0 if attrs.get("termination_for_convenience") else 0.0
    notice_days = attrs.get("notice_days", 60)
    fee_req = 1.0 if attrs.get("termination_fee_required") else 0.0
    score = (
        0.5 * tfc +
        0.3 * (1 - min(notice_days / 60, 1)) +
        0.2 * (1 - fee_req)
    )
    return round(score, 3)

# Tabs
tab1, tab2 = st.tabs(["📋 Query Results", "📊 Analytics"])

with tab1:
    st.header(f"📄 Contracts with {clause_type} Clauses")
    results = get_contracts_by_clause_type(clause_type)
    if results:
        if clause_type == "Termination for Convenience":
            scored = []
            for vendor, title, page, heading, excerpt, attrs in results:
                score = get_termination_flexibility_score(attrs)
                scored.append((vendor, title, page, heading, excerpt, attrs, score))
            scored.sort(key=lambda x: x[6], reverse=True)
            for i, (vendor, title, page, heading, excerpt, attrs, score) in enumerate(scored, 1):
                with st.expander(f"#{i} {vendor} - {title} (Score: {score})"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Page:** {page}")
                        st.write(f"**Section:** {heading}")
                        st.write(f"**Excerpt:** {excerpt}...")
                    with c2:
                        st.metric("Flex Score", f"{score}/1.0")
                        if attrs:
                            st.write("**Attributes:**")
                            if isinstance(attrs, str):
                                try:
                                    st.json(json.loads(attrs))
                                except:
                                    st.write(attrs)
                            else:
                                st.json(attrs)
        else:
            for i, (vendor, title, page, heading, excerpt, attrs) in enumerate(results, 1):
                with st.expander(f"{i}. {vendor} - {title}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Page:** {page}")
                        st.write(f"**Section:** {heading}")
                        st.write(f"**Excerpt:** {excerpt}...")
                    with c2:
                        if attrs:
                            st.write("**Attributes:**")
                            if isinstance(attrs, str):
                                try:
                                    st.json(json.loads(attrs))
                                except:
                                    st.write(attrs)
                            else:
                                st.json(attrs)
    else:
        st.info(f"No {clause_type.lower()} clauses found in uploaded contracts.")

with tab2:
    st.header("📊 Contract Analytics")
    total_contracts, total_spans, distribution = get_database_stats()
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Contracts", total_contracts)
    with c2: st.metric("Total Spans", total_spans)
    with c3: st.metric("Classified Spans", sum(cnt for _, cnt in distribution))
    if distribution:
        st.subheader("📈 Clause Type Distribution")
        chart_data = {"Clause Type": [t for t, _ in distribution], "Count": [c for _, c in distribution]}
        st.bar_chart(data=chart_data, x="Clause Type", y="Count")
        st.subheader("📋 Detailed Breakdown")
        for t, c in distribution:
            s1, s2 = st.columns([3,1])
            with s1: st.write(f"**{t}**")
            with s2: st.write(c)

# Footer
st.markdown("---")
st.caption("Built with Streamlit, pgvector, and Groq AI • © 2025")

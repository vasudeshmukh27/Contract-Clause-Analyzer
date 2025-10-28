# 📋 Contract Clause Analyzer

> **AI-powered intelligent contract analysis system**  
> Automatically parse, classify, and extract key contractual clauses from PDFs using LLMs and vector embeddings.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791?logo=postgresql&logoColor=white)
![Groq API](https://img.shields.io/badge/Groq-LLM-FF6B00)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)

---

## 🎯 What This Project Does

Contract Clause Analyzer is a **full-stack AI document intelligence platform** that helps procurement teams, legal departments, and contract managers quickly understand and compare vendor contracts.

**Key Workflow:**
1. 📤 Upload PDF contracts
2. 🔍 Automatic parsing & semantic chunking
3. 🤖 LLM-powered clause classification
4. 📊 Interactive dashboard for insights
5. 📈 Analytics & comparison tools

---

## ✨ Core Features

### **8-Clause Classification System**
Automatically detect and extract:
- 🔓 **Termination for Convenience** - Exit flexibility with notice periods
- 📊 **Data Sharing** - Privacy, confidentiality, and data handling terms
- 📢 **Notice to Terminate** - Required notification periods
- ⚖️ **Governing Law** - Legal jurisdiction and venue
- 💰 **Cap on Liability** - Financial exposure limits
- 🔄 **Auto Renewal** - Automatic renewal terms and conditions
- 🔎 **Audit Rights** - Right to inspect and verify
- 🛡️ **Indemnification** - Indemnity and hold harmless clauses

### **Intelligent Processing Pipeline**
- ✅ PDF parsing with smart text extraction
- ✅ Semantic span generation (automatic chunking)
- ✅ Vector embeddings (384-dimensional)
- ✅ Hybrid classification (rule-based + LLM)
- ✅ Structured attribute extraction
- ✅ Confidence scoring

### **Interactive Web Dashboard**
- 📁 Real-time PDF upload & processing
- 🔍 Query & filter by clause type
- 📈 Termination flexibility scoring
- 📊 Contract analytics dashboard
- 💾 Persistent database storage

---

## 📦 Project Structure

```
contract-clause-analyzer/
│
├── 📄 Core Application Files
│   ├── app.py                    # Streamlit web dashboard
│   ├── ingest_one.py             # PDF → Database pipeline
│   ├── clause_detector.py        # LLM classification engine
│   ├── schema.py                 # Database schema setup
│   └── schema.sql                # SQL table definitions
│
├── 🛠️ Utility Scripts
│   ├── count_clause_types.py     # Clause distribution analyzer
│   ├── search_demo.py            # Vector search demonstration
│   ├── migrate_vector_384.py     # Embedding migration utility
│   └── verify_db.py              # Database integrity checker
│
├── 📁 Data Directories
│   └── data/
│       └── contracts/            # Upload PDFs here ⬅️
│
├── 🔧 Configuration
│   ├── requirements.txt          # Python dependencies
│   ├── .env                      # Environment variables (create manually)
│   └── README.md                 # This file
│
└── 📝 Documentation
    └── I want to work on this genAI project...  # Project notes
```

---

## 🚀 Quick Start (5 Minutes)

### **Step 1: Clone & Setup Environment**

```bash
# Clone repository
git clone https://github.com/yourusername/contract-clause-analyzer.git
cd contract-clause-analyzer

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

### **Step 2: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 3: Configure Environment**

Create a `.env` file in project root:

```env
# PostgreSQL Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=contract_analyzer
DB_USER=postgres
DB_PASSWORD=your_postgres_password

# Groq API (Get free key: https://console.groq.com/keys)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Sentence Transformer Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### **Step 4: Setup PostgreSQL Database**

```bash
# Start PostgreSQL (Windows: Services, Mac: brew, Linux: systemctl)
psql -U postgres

# Create database
CREATE DATABASE contract_analyzer;
\c contract_analyzer;

# Run schema
\i schema.sql
```

Or use Python:
```bash
python schema.py
```

### **Step 5: Launch Dashboard**

```bash
streamlit run app.py
```

✅ Open browser → `http://localhost:8501`

---

## 📖 Usage Guide

### **1. Upload & Analyze a Contract**

```
Sidebar → "📁 Upload New Contract"
  ↓
Drag & drop PDF or click "Browse files"
  ↓
Click "🔍 Analyze Contract"
  ↓
Wait 2-5 minutes (depending on PDF size)
  ↓
Results appear in Query Results tab
```

### **2. Query Clauses**

```
Sidebar → "Select Clause Type" dropdown
  ↓
Choose from 8 clause types
  ↓
View extracted clauses with:
  - Page number
  - Section heading
  - Text excerpt (200 characters)
  - Extracted attributes (JSON)
```

### **3. View Analytics**

```
Main → "📊 Analytics" tab
  ↓
See:
  - Total contracts processed
  - Total spans parsed
  - Classified vs. unclassified breakdown
  - Clause type distribution chart
```

### **4. Termination Flexibility Scoring**

When viewing "Termination for Convenience" clauses, they're ranked by **flexibility score**:

| Score Range | Meaning | Example |
|-----------|---------|---------|
| 0.9-1.0 | ✅ Very Flexible | 30-day notice, no fees, convenience termination |
| 0.5-0.7 | ⚠️ Moderate | 60-day notice, 20% termination fee |
| 0.1-0.3 | ❌ Restrictive | 180-day notice, 50% penalty, auto-renewal |

---

## 🔧 File Documentation

### **Core Scripts**

#### `app.py` (9 KB)
Streamlit web application - main user interface
- **Functions:** PDF upload, clause querying, analytics dashboard
- **Start:** `streamlit run app.py`

#### `ingest_one.py` (3 KB)
PDF ingestion pipeline
- **Functions:** Parse PDF → extract spans → generate embeddings → store in DB
- **Start:** `python ingest_one.py --pdf <path> --vendor <name> --title <title>`

#### `clause_detector.py` (10 KB)
LLM-powered classification engine
- **Functions:** Classify unknown spans using Groq API
- **Start:** `python clause_detector.py`
- **Uses:** Hybrid approach (keyword detection + Groq LLM)

#### `schema.py` & `schema.sql`
Database initialization
- **Tables:** `documents`, `clause_spans`, `span_embeddings`
- **Functions:** Create tables, indexes, extensions
- **Start:** `python schema.py` or `psql -U postgres -f schema.sql`

### **Utility Scripts**

#### `count_clause_types.py` (1 KB)
Analyze distribution of classified clauses
- **Output:** Count of each clause type in database

#### `search_demo.py` (2 KB)
Demonstrate vector similarity search
- **Example:** Find similar contract clauses

#### `verify_db.py` (1 KB)
Check database integrity and row counts

#### `migrate_vector_384.py` (1 KB)
Utility for migrating embeddings to 384-dimensional vectors

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER UPLOADS PDF                         │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              INGEST_ONE.PY (Parsing Phase)                  │
│  • Extract text from PDF                                    │
│  • Split into semantic spans                                │
│  • Generate 384-dim embeddings                              │
│  • Store in PostgreSQL                                      │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│            CLAUSE_DETECTOR.PY (Classification)              │
│  • Fetch "Unknown" spans                                    │
│  • Rule-based keyword detection                             │
│  • Query Groq API for LLM classification                    │
│  • Extract structured attributes                            │
│  • Update clause_type & attributes                          │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              APP.PY (Dashboard & Query)                     │
│  • Display classified clauses by type                       │
│  • Calculate flexibility scores                             │
│  • Show analytics & charts                                  │
│  • Enable clause comparison                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Web UI & dashboard |
| **Backend** | Python 3.10+ | Core logic |
| **Database** | PostgreSQL + pgvector | Data storage & vector search |
| **LLM** | Groq API (Llama 3.3 70B) | Clause classification |
| **Embeddings** | SentenceTransformers | Semantic vectors (384-dim) |
| **PDF Parsing** | Unstructured | Text extraction |
| **ORM** | psycopg2 | Database connection |

---

## 📊 Clause Type Reference

| Clause | Keywords | Attributes Extracted | Example |
|--------|----------|-------------------|---------|
| **Termination for Convenience** | terminate, convenience, without cause, at will | `notice_days`, `termination_fee_required`, `termination_for_convenience` | "Either party may terminate for any reason with 30 days' notice" |
| **Data Sharing** | data, confidential, share, disclose, privacy | `data_categories`, `sharing_restrictions` | "Vendor shall not share client data with third parties without written consent" |
| **Notice to Terminate** | notice, days, written, advance | `notice_days`, `notice_type` | "Termination requires 90 days' written notice to either party" |
| **Governing Law** | governing law, jurisdiction, courts, state law | `governing_jurisdiction`, `venue` | "This Agreement shall be governed by California law" |
| **Cap on Liability** | liability, limit, exceed, damages, cap | `liability_cap_amount`, `liability_cap_type` | "Liability capped at fees paid in prior 12 months" |
| **Auto Renewal** | renew, automatic, successive, renewal term | `renewal_period`, `auto_renewal_enabled` | "Automatically renews for successive one-year terms" |
| **Audit Rights** | audit, inspect, examine, records, books | `audit_frequency`, `audit_scope` | "Client may audit Vendor's records upon 30 days' notice" |
| **Indemnification** | indemnify, hold harmless, defend, indemnified | `indemnifying_party`, `indemnified_party` | "Vendor shall indemnify and hold Client harmless from third-party claims" |

---

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'psycopg2'`
```bash
pip install psycopg2-binary
```

### Error: `GROQ_API_KEY not found`
- ✅ Create `.env` file in project root
- ✅ Get key from https://console.groq.com/keys
- ✅ Set `GROQ_API_KEY=your_key`

### Error: `Rate limit exceeded (429)`
- Free tier: 100K tokens/day
- ✅ Wait for reset or upgrade at https://console.groq.com/settings/billing

### Error: `PostgreSQL connection refused`
```bash
# Verify PostgreSQL is running
# Windows: Services → PostgreSQL (running?)
# Mac: brew services list | grep postgres
# Linux: sudo systemctl status postgresql
```

### Error: `No clause types found`
- Threshold may be too high
- ✅ Reduce `min_conf=0.1` in `clause_detector.py`
- ✅ Verify contract contains target clause language

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Parsing Speed** | ~5-10 seconds per contract |
| **Classification Speed** | ~2-3 minutes per 100 spans (depends on Groq API) |
| **Embedding Dimension** | 384-dimensional vectors |
| **Database Size** | ~1-2GB per 1000 contracts |
| **Free Tier Limit** | 100K tokens/day |

---

## 🚀 Deployment

### **Local Development**
```bash
streamlit run app.py
```

### **Production (Ubuntu 20.04)**

```bash
# Install
sudo apt-get update && sudo apt-get install python3.10 postgresql

# Setup
cd /opt && sudo git clone <repo>
cd contract-analyzer && pip install -r requirements.txt

# Run with systemd
sudo systemctl start contract-analyzer
```

---

## 💡 Example Use Cases

### **1. Vendor Contract Review**
- Upload 5 vendor MSAs
- Compare termination flexibility scores
- Export results for procurement team
- Negotiate terms before signing

### **2. Legal Compliance Check**
- Audit all active contracts
- Identify data sharing risks
- Check liability caps
- Ensure governing law compliance

### **3. Contract Standardization**
- Analyze competitor contracts
- Extract best practices
- Create template terms
- Build procurement playbook

---

## 🎓 How It Works (Technical Deep Dive)

### **Phase 1: Ingestion** (ingest_one.py)
1. Parse PDF → Extract text elements
2. Create semantic spans (typically paragraph-sized)
3. Generate 384-dim embeddings via SentenceTransformer
4. Insert into PostgreSQL with pgvector

### **Phase 2: Classification** (clause_detector.py)
1. Query all "Unknown" spans
2. Rule-based keyword detection
3. Call Groq API for LLM classification
4. Extract JSON attributes (notice days, fees, etc.)
5. Update database with clause_type

### **Phase 3: Visualization** (app.py)
1. Query PostgreSQL by clause_type
2. Calculate termination flexibility scores
3. Render in Streamlit
4. Display analytics charts

---

## 📝 Environment Variables

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=contract_analyzer
DB_USER=postgres
DB_PASSWORD=YourPassword

# API
GROQ_API_KEY=your_groq_key

# Models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## 🔒 Security Notes

- ⚠️ Never commit `.env` file to Git
- ⚠️ Use strong PostgreSQL passwords
- ⚠️ Rotate Groq API keys regularly
- ⚠️ Store PDFs in secure location
- ⚠️ Audit who has database access

---

## 📞 Support & Contributing

### **Issues?**
- Check `#troubleshooting` section
- Open GitHub issue with error message
- Include `.env` excerpt (without sensitive data)

### **Want to Contribute?**
```bash
git checkout -b feature/amazing-feature
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

---

## 🎉 What's Next?

- [ ] Multi-contract comparison matrix
- [ ] Risk scoring system
- [ ] Custom clause templates
- [ ] REST API endpoints
- [ ] Batch processing UI
- [ ] Excel/PDF export
- [ ] Contract versioning
- [ ] Team collaboration

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👏 Acknowledgments

- **Groq** - Free LLM API access
- **PostgreSQL & pgvector** - Semantic search
- **Streamlit** - Rapid UI development
- **Unstructured** - PDF parsing
- **SentenceTransformers** - Embeddings

---

**Built with ❤️ for intelligent contract analysis**

*Last Updated: October 29, 2025*
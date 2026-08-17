# Enterprise ATS: AI Resume-Job Matcher and Candidate Ranker

An enterprise-grade Applicant Tracking System (ATS) engine designed to evaluate, score, and rank candidate resumes against role-specific Job Descriptions using contextual semantic embeddings, ontology-driven skill extraction, and multi-factor alignment algorithms.

---

## Technical Overview

Traditional resume screening systems rely heavily on brittle keyword matching or surface-level document classification, leading to high false-positive and false-negative rates. 

This platform implements an end-to-end modular pipeline combining:
1. **Multi-Format Document Ingestion:** Parsing and normalizing structured and unstructured text from PDF, DOCX, and TXT files.
2. **Taxonomy-Driven Skill Extraction:** Automated entity extraction utilizing an ontology of over 1,000 normalized technical and domain competencies.
3. **Contextual Semantic Matching:** Vector embedding cosine similarity via Sentence-BERT (`all-MiniLM-L6-v2`) to capture experiential alignment beyond raw keywords.
4. **Hybrid Quantitative Scoring:** Multi-factor scoring incorporating semantic similarity, skill gap coverage ratio, and terminology density.
5. **Automated Candidate Summarization & Reporting:** Extractive summarization and export capabilities for hiring workflows.

---

## Architectural Design

The codebase adheres to object-oriented, decoupled, and modular software engineering principles.

```
AI-Resume-Matcher-and-ATS-Ranker/
├── app.py                         # Presentation Layer (Streamlit Dashboard)
├── setup.py                       # Package Definition and Distribution Setup
├── requirements.txt               # Pinned Python Dependencies
├── README.md                      # Technical Documentation
├── LICENSE                        # MIT License
├── .gitignore                     # Version Control Exclusions
├── src/
│   ├── __init__.py
│   ├── constants/
│   │   ├── __init__.py
│   │   └── skill_taxonomy.py      # Normalized Skill Taxonomy & Entity Mappings
│   ├── entity/
│   │   ├── __init__.py
│   │   ├── config_entity.py       # Configuration Dataclasses
│   │   └── artifact_entity.py     # Evaluation and Document Data Artifacts
│   ├── exception/
│   │   ├── __init__.py
│   │   └── custom_exception.py    # Standardized Exception Handling
│   ├── logger/
│   │   ├── __init__.py
│   │   └── logging.py             # File and Console Logging Configuration
│   ├── components/
│   │   ├── __init__.py
│   │   ├── document_parser.py     # Multi-Format Text Extraction Component
│   │   ├── skill_extractor.py     # Skill Extraction and Gap Analysis Component
│   │   ├── semantic_matcher.py    # Vector Embeddings and Hybrid Scoring Component
│   │   └── candidate_evaluator.py # Profile Summarizer and Verdict Generator
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── matching_pipeline.py   # End-to-End Execution Pipeline
│   └── utils/
│       ├── __init__.py
│       └── export_manager.py      # Tabular and CSV Export Utilities
└── samples/
    ├── sample_job_description.txt
    └── sample_resumes/
        ├── candidate_1_lead_ai_engineer.txt
        ├── candidate_2_data_engineer.txt
        ├── candidate_3_frontend_developer.txt
        └── candidate_4_financial_analyst.txt
```

---

## Mathematical Formulation

The unified ATS Match Score is formulated as a weighted linear combination of three distinct feature spaces:

$$\text{ATS Match Score} = (w_{\text{sem}} \cdot S_{\text{sem}}) + (w_{\text{skill}} \cdot S_{\text{skill}}) + (w_{\text{kw}} \cdot S_{\text{kw}})$$

Where:
- $S_{\text{sem}}$ denotes the cosine similarity of the 384-dimensional normalized dense vectors:
  $$S_{\text{sem}} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
- $S_{\text{skill}}$ represents the Jaccard-based coverage ratio of mandatory job requirements:
  $$S_{\text{skill}} = \frac{|\mathcal{K}_{\text{job}} \cap \mathcal{K}_{\text{resume}}|}{|\mathcal{K}_{\text{job}}|}$$
- $S_{\text{kw}}$ denotes the Term Frequency-Inverse Document Frequency (TF-IDF) n-gram similarity.
- $w_{\text{sem}}, w_{\text{skill}}, w_{\text{kw}}$ represent configurable weighting hyperparameters (defaulting to 0.45, 0.40, and 0.15 respectively).

---

## Installation and Local Setup

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11
- Git

### 2. Environment Configuration
```bash
# Clone the repository
git clone https://github.com/<YOUR_ORGANIZATION>/AI-Resume-Matcher-and-ATS-Ranker.git
cd AI-Resume-Matcher-and-ATS-Ranker

# Initialize virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Linux / macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Application Execution
```bash
streamlit run app.py
```
Access the application interface at `http://localhost:8501`.

---

## Cloud Deployment

### Option A: Streamlit Community Cloud (Recommended)
1. Commit and push the codebase to a GitHub repository.
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and authenticate with GitHub.
3. Click **New app**, select the repository, and specify `app.py` as the entrypoint.
4. Select **Deploy**.

### Option B: Render.com (Web Service)
1. Connect the GitHub repository to a new Render Web Service.
2. Specify environment: **Python 3**.
3. Build Command: `pip install -r requirements.txt`.
4. Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

---

## License

This software is distributed under the [MIT License](LICENSE).

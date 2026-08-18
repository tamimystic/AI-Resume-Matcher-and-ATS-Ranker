# Universal ATS: AI-Powered Candidate Matcher and Ranker

An enterprise-grade, domain-agnostic Applicant Tracking System (ATS) matching engine that evaluates and ranks candidate resumes against target Job Descriptions using dynamic requirement extraction, passage-level semantic evidence retrieval, and calibrated vector similarity.

---

## Technical Highlights

1. **Zero Domain Bias:** Contains no predefined or hardcoded skill lists. Works dynamically for any job specification across all professional disciplines.
2. **Point-by-Point Requirement Audit:** Automatically splits target job posts into atomic criteria and retrieves best-matching candidate evidence passages.
3. **Calibrated Semantic Vector Space:** Uses high-dimensional transformer embeddings (`all-MiniLM-L6-v2`) with continuous calibration for contextual precision.
4. **Multi-Format Ingestion:** High-speed extraction for PDF, DOCX, and plain text documents.
5. **Data Privacy:** Operates entirely in-memory with zero external data transmission.

---

## Project Structure

```
AI-Resume-Matcher-and-ATS-Ranker/
├── app.py                         # Streamlit Web Application
├── setup.py                       # Packaging Configuration
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
├── LICENSE                        # MIT License
├── .gitignore                     # Version Control Exclusions
└── src/
    ├── __init__.py
    ├── entity/
    │   ├── __init__.py
    │   ├── config_entity.py       # Configuration Dataclasses
    │   └── artifact_entity.py     # Evaluation & Evidence Data Contracts
    ├── exception/
    │   ├── __init__.py
    │   └── custom_exception.py    # Standardized Exception Handling
    ├── logger/
    │   ├── __init__.py
    │   └── logging.py             # Timestamped File and Console Logger
    ├── components/
    │   ├── __init__.py
    │   ├── document_parser.py     # Document Ingestion (PDF, DOCX, TXT)
    │   ├── requirement_extractor.py # Dynamic Criterion & Keyphrase Extractor
    │   ├── semantic_matcher.py    # Hybrid Vector Alignment & Evidence Matcher
    │   └── candidate_evaluator.py # Profile Summarization & Decision Engine
    ├── pipeline/
    │   ├── __init__.py
    │   └── matching_pipeline.py   # End-to-End Orchestrator Pipeline
    └── utils/
        ├── __init__.py
        └── export_manager.py      # CSV & Tabular Report Generator
```

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/<YOUR_ORGANIZATION>/AI-Resume-Matcher-and-ATS-Ranker.git
cd AI-Resume-Matcher-and-ATS-Ranker

# Initialize and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

---

## Cloud Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and link your GitHub account.
3. Select your repository, specify `app.py` as the entry file, and click **Deploy**.

---

## License

Distributed under the [MIT License](LICENSE).

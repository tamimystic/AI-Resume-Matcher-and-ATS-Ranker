from setuptools import setup, find_packages

setup(
    name="ai_resume_matcher",
    version="1.0.0",
    author="Engineering Team",
    description="End-to-End Enterprise ATS Resume-Job Matching and Ranking System",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.35.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "PyMuPDF>=1.24.0",
        "python-docx>=1.1.0",
        "altair>=5.0.0"
    ]
)

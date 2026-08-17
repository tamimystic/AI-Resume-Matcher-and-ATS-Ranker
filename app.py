import os
import streamlit as st
import pandas as pd
from typing import List, Tuple

from src.pipeline.matching_pipeline import ResumeMatchingPipeline
from src.entity.config_entity import MatchingWeightsConfig, ModelConfig
from src.entity.artifact_entity import CandidateEvaluationArtifact
from src.utils.export_manager import ExportManager
from src.logger.logging import logger

st.set_page_config(
    page_title="Enterprise ATS - Resume Matcher and Ranker",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise UI styling without emojis
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .enterprise-header {
        background-color: #0f172a;
        padding: 2rem 2.2rem;
        border-radius: 8px;
        color: #ffffff;
        margin-bottom: 1.8rem;
        border: 1px solid #1e293b;
    }
    .enterprise-header h1 {
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff !important;
    }
    .enterprise-header p {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        margin-bottom: 0;
        line-height: 1.5;
    }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .skill-tag {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px 3px 2px 0;
    }
    .tag-matched {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .tag-missing {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .tag-neutral {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
    }

    .candidate-card-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .score-container {
        font-size: 1rem;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 6px;
        display: inline-block;
    }

    .summary-text-box {
        background: #f8fafc;
        border-radius: 6px;
        padding: 0.9rem 1rem;
        border: 1px solid #e2e8f0;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #334155;
        margin-top: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def initialize_pipeline(semantic_weight: float, skills_weight: float, keywords_weight: float) -> ResumeMatchingPipeline:
    weights = MatchingWeightsConfig(
        semantic_weight=semantic_weight,
        skills_weight=skills_weight,
        keywords_weight=keywords_weight
    )
    return ResumeMatchingPipeline(weights_config=weights, model_config=ModelConfig())


# Top Header
st.markdown("""
<div class="enterprise-header">
    <h1>Enterprise ATS - Resume Matcher and Ranker</h1>
    <p>Contextual semantic evaluation, taxonomy-driven skill gap assessment, and quantitative candidate ranking against target job specifications.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### Evaluation Parameters")
    st.caption("Adjust weight distributions across scoring dimensions.")

    weight_sem = st.slider("Semantic Context Weight", min_value=0.1, max_value=1.0, value=0.45, step=0.05)
    weight_sk = st.slider("Skill Coverage Weight", min_value=0.1, max_value=1.0, value=0.40, step=0.05)
    weight_kw = st.slider("Keyword Overlap Weight", min_value=0.0, max_value=0.5, value=0.15, step=0.05)

    st.markdown("---")
    st.markdown("### Demonstration Dataset")
    st.caption("Load verified reference job specifications and multi-tier candidate resumes.")

    load_sample_action = st.button("Load Reference Dataset", use_container_width=True)

    st.markdown("---")
    st.markdown("### Architecture Overview")
    st.markdown("""
    - Transformer Backbone: all-MiniLM-L6-v2
    - Ontology: 1,000+ Normalized Technical Skills
    - Scoring: Multi-Factor Hybrid Algorithm
    - Privacy: Local In-Memory Evaluation
    """)

pipeline = initialize_pipeline(weight_sem, weight_sk, weight_kw)

# Session state setup
if "job_description_input" not in st.session_state:
    st.session_state.job_description_input = ""
if "is_demo_active" not in st.session_state:
    st.session_state.is_demo_active = False

sample_base_path = os.path.join(os.path.dirname(__file__), "samples")
sample_jd_path = os.path.join(sample_base_path, "sample_job_description.txt")
sample_resumes_folder = os.path.join(sample_base_path, "sample_resumes")

if load_sample_action:
    if os.path.exists(sample_jd_path):
        with open(sample_jd_path, "r", encoding="utf-8") as f:
            st.session_state.job_description_input = f.read()
        st.session_state.is_demo_active = True
        st.success("Reference dataset loaded. Proceed to candidate analysis below.")

# Input Layout
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown("#### 1. Job Specification")
    job_title_input = st.text_input("Role Title (Optional)", value="Senior Full-Stack AI Engineer" if st.session_state.is_demo_active else "", placeholder="e.g. Senior Backend Engineer")
    jd_text = st.text_area(
        "Job Description and Requirements",
        value=st.session_state.job_description_input,
        height=300,
        placeholder="Paste full responsibilities, mandatory skills, and qualification criteria..."
    )

    if jd_text.strip():
        extracted_job_skills = pipeline.extract_job_skills(jd_text)
        with st.expander(f"Extracted Requirements ({len(extracted_job_skills)} Skills)", expanded=True):
            if extracted_job_skills:
                tags = " ".join([f'<span class="skill-tag tag-neutral">{s}</span>' for s in extracted_job_skills])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.info("No taxonomy-listed skills found. Semantic context evaluation will proceed.")
    else:
        extracted_job_skills = []

with col_right:
    st.markdown("#### 2. Candidate Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload candidate documents for batch evaluation."
    )

    if st.session_state.is_demo_active and not uploaded_files:
        st.info("Using 4 preloaded candidate profiles (Lead AI Engineer, Data Engineer, Frontend Developer, Financial Analyst).")

    st.markdown("<br>", unsafe_allow_html=True)
    execute_evaluation = st.button("Execute Candidate Evaluation", type="primary", use_container_width=True)

st.markdown("---")

# Execution Handler
if execute_evaluation:
    if not jd_text.strip():
        st.error("A valid Job Description is required to initiate matching.")
    else:
        batch_items: List[Tuple[any, str]] = []

        if uploaded_files:
            for file_item in uploaded_files:
                batch_items.append((file_item, file_item.name))
        elif st.session_state.is_demo_active and os.path.exists(sample_resumes_folder):
            for filename in sorted(os.listdir(sample_resumes_folder)):
                filepath = os.path.join(sample_resumes_folder, filename)
                if os.path.isfile(filepath):
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    formatted_name = filename.replace(".txt", "").replace("_", " ").title()
                    batch_items.append((content, formatted_name))
        else:
            st.error("Upload at least one resume document or load the demonstration dataset.")

        if batch_items:
            progress_bar = st.progress(0, text="Evaluating candidate documents...")

            def update_progress(current_idx: int, total_count: int, current_filename: str):
                progress_bar.progress(current_idx / total_count, text=f"Processing {current_idx}/{total_count}: {current_filename}")

            evaluation_artifacts = pipeline.evaluate_batch(
                resume_items=batch_items,
                job_description_text=jd_text,
                progress_callback=update_progress
            )

            progress_bar.empty()
            st.session_state.evaluation_results = evaluation_artifacts
            st.session_state.active_job_skills = extracted_job_skills

# Results Presentation
if "evaluation_results" in st.session_state and st.session_state.evaluation_results:
    artifacts: List[CandidateEvaluationArtifact] = st.session_state.evaluation_results
    total_count = len(artifacts)
    top_matches = len([a for a in artifacts if a.match_percentage >= 75.0])
    good_matches = len([a for a in artifacts if 50.0 <= a.match_percentage < 75.0])
    avg_score = round(sum(a.match_percentage for a in artifacts) / total_count, 1)

    st.markdown("### Evaluation Summary")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_count}</div>
            <div class="kpi-label">Processed Resumes</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #10b981;">
            <div class="kpi-value" style="color: #10b981;">{top_matches}</div>
            <div class="kpi-label">Top Tier Matches</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #f59e0b;">
            <div class="kpi-value" style="color: #f59e0b;">{good_matches}</div>
            <div class="kpi-label">Moderate Alignment</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_score}%</div>
            <div class="kpi-label">Average Match Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_table, tab_details, tab_export = st.tabs([
        "Ranked Leaderboard",
        "Candidate Profiles",
        "Export Report"
    ])

    with tab_table:
        filter_col, search_col = st.columns([1, 2])
        with filter_col:
            selected_tier = st.selectbox("Filter Tier", ["All Profiles", "Top Match (>=75%)", "Good Match (50-74%)", "Low Match (<50%)"])
        with search_col:
            candidate_search = st.text_input("Search Candidate", placeholder="Filter by document name...")

        filtered_artifacts = artifacts
        if selected_tier == "Top Match (>=75%)":
            filtered_artifacts = [a for a in filtered_artifacts if a.match_percentage >= 75.0]
        elif selected_tier == "Good Match (50-74%)":
            filtered_artifacts = [a for a in filtered_artifacts if 50.0 <= a.match_percentage < 75.0]
        elif selected_tier == "Low Match (<50%)":
            filtered_artifacts = [a for a in filtered_artifacts if a.match_percentage < 50.0]

        if candidate_search.strip():
            filtered_artifacts = [a for a in filtered_artifacts if candidate_search.lower() in a.filename.lower()]

        table_rows = []
        for rank, a in enumerate(filtered_artifacts, start=1):
            table_rows.append({
                "Rank": rank,
                "Candidate Identifier": a.filename,
                "Overall Score": f"{a.match_percentage}%",
                "Fit Tier": a.fit_tier,
                "Decision Verdict": a.verdict.verdict_badge,
                "Matched Skills": a.skill_gap.matched_skills_count,
                "Missing Skills": len(a.skill_gap.missing_skills)
            })

        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No candidates match the specified filter criteria.")

    with tab_details:
        for idx, a in enumerate(artifacts, start=1):
            with st.container():
                st.markdown(f"""
                <div class="candidate-card-container">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">#{idx}. {a.filename}</span>
                            <span style="margin-left: 10px; font-size: 0.8rem; font-weight: 600; padding: 3px 8px; border-radius: 4px; {a.verdict.badge_style}">
                                {a.verdict.verdict_badge}
                            </span>
                        </div>
                        <div class="score-container" style="background-color: {a.fit_color}15; color: {a.fit_color}; border: 1px solid {a.fit_color};">
                            {a.match_percentage}% Score
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"Inspection Details: #{idx}. {a.filename}", expanded=(idx == 1)):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Semantic Similarity", f"{a.semantic_score}%")
                    m2.metric("Skill Coverage", f"{a.skill_score}%")
                    m3.metric("Keyword Overlap", f"{a.keyword_score}%")

                    st.markdown("##### Assessment Verdict")
                    st.markdown(f"**Recommendation:** {a.verdict.verdict_description}")

                    if a.verdict.strengths:
                        st.markdown("**Demonstrated Strengths:**")
                        for str_item in a.verdict.strengths:
                            st.markdown(f"- {str_item}")

                    if a.verdict.gaps:
                        st.markdown("**Identified Discrepancies:**")
                        for gap_item in a.verdict.gaps:
                            st.markdown(f"- {gap_item}")

                    st.markdown("##### Skill Alignment Breakdown")
                    col_m, col_g = st.columns(2)
                    with col_m:
                        st.markdown(f"**Matched Competencies ({a.skill_gap.matched_skills_count})**")
                        if a.skill_gap.matched_skills:
                            matched_tags = " ".join([f'<span class="skill-tag tag-matched">{s}</span>' for s in a.skill_gap.matched_skills])
                            st.markdown(matched_tags, unsafe_allow_html=True)
                        else:
                            st.caption("No direct skill matches detected.")

                    with col_g:
                        st.markdown(f"**Missing Competencies ({len(a.skill_gap.missing_skills)})**")
                        if a.skill_gap.missing_skills:
                            missing_tags = " ".join([f'<span class="skill-tag tag-missing">{s}</span>' for s in a.skill_gap.missing_skills])
                            st.markdown(missing_tags, unsafe_allow_html=True)
                        else:
                            st.caption("All required skills satisfied.")

                    st.markdown("##### Executive Profile Summary")
                    st.markdown(f"""
                    <div class="summary-text-box">
                        {a.executive_summary}
                    </div>
                    """, unsafe_allow_html=True)

    with tab_export:
        st.markdown("#### Evaluation Report Export")
        st.caption("Download structured candidate metrics and skills assessment in CSV format.")

        df_export = ExportManager.to_dataframe(artifacts)
        st.dataframe(df_export, use_container_width=True)

        csv_data = ExportManager.to_csv_bytes(df_export)
        st.download_button(
            label="Download Evaluation Dataset (CSV)",
            data=csv_data,
            file_name="candidate_ats_evaluation_report.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

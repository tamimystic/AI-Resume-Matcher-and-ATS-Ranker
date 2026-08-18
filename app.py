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
    page_title="Universal ATS - AI Resume Matcher and Ranker",
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

    .req-item-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
    }
    .req-satisfied {
        border-left: 4px solid #10b981;
    }
    .req-partial {
        border-left: 4px solid #f59e0b;
    }
    .req-unmet {
        border-left: 4px solid #ef4444;
    }

    .evidence-quote {
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        padding: 0.4rem 0.8rem;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #475569;
        margin-top: 0.4rem;
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

    .keyphrase-tag {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px 3px 2px 0;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def initialize_pipeline(requirement_weight: float, macro_weight: float, terminology_weight: float) -> ResumeMatchingPipeline:
    weights = MatchingWeightsConfig(
        requirement_weight=requirement_weight,
        macro_semantic_weight=macro_weight,
        terminology_weight=terminology_weight
    )
    return ResumeMatchingPipeline(weights_config=weights, model_config=ModelConfig())


# Top Header
st.markdown("""
<div class="enterprise-header">
    <h1>Universal ATS - AI Resume Matcher and Ranker</h1>
    <p>Dynamic requirement extraction, contextual vector similarity, and multi-factor candidate evaluation for any industry domain.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration: Only weights and algorithmic settings (Zero hardcoded presets)
with st.sidebar:
    st.markdown("### Evaluation Weighting")
    st.caption("Adjust weight distributions across scoring dimensions.")

    w_req = st.slider("Requirement Evidence Coverage", min_value=0.2, max_value=0.8, value=0.55, step=0.05,
                      help="Evaluates point-by-point evidence across individual job criteria.")
    w_macro = st.slider("Macro Context Similarity", min_value=0.1, max_value=0.5, value=0.25, step=0.05,
                        help="Measures full document semantic contextual alignment.")
    w_term = st.slider("Domain Terminology Overlap", min_value=0.0, max_value=0.4, value=0.20, step=0.05,
                       help="Measures dynamic domain keyphrase density.")

    st.markdown("---")
    st.markdown("### System Characteristics")
    st.markdown("""
    - Domain Agnostic: No hardcoded rules or predefined skill lists
    - Dynamic Extraction: Requirements parsed directly from input
    - Semantic Neural Vector Space: 384-dimensional dense embeddings
    - Privacy: Local in-memory processing with zero external data transmission
    """)

pipeline = initialize_pipeline(w_req, w_macro, w_term)

# Main 2-Column Layout for Pure User Inputs
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown("#### 1. Target Job Description")
    jd_text = st.text_area(
        "Paste Job Description, Responsibilities, and Qualifications",
        height=340,
        placeholder="Paste any job posting text from any industry (e.g. Healthcare, Engineering, Finance, Education, Operations, Aviation, Legal, etc.)..."
    )

    if jd_text.strip():
        analyzed_jd = pipeline.analyze_job_description(jd_text)
        reqs = analyzed_jd["requirements"]
        kps = analyzed_jd["keyphrases"]

        with st.expander(f"Dynamically Extracted Criteria ({len(reqs)} Requirements, {len(kps)} Keyphrases)", expanded=True):
            st.markdown(f"**Identified Criteria ({len(reqs)}):**")
            for idx, r in enumerate(reqs[:10], start=1):
                st.markdown(f"- **R{idx}:** {r}")
            if len(reqs) > 10:
                st.caption(f"... and {len(reqs) - 10} additional criteria.")

            st.markdown(f"**Detected Domain Terminology ({len(kps)}):**")
            if kps:
                tags = " ".join([f'<span class="keyphrase-tag">{kp}</span>' for kp in kps])
                st.markdown(tags, unsafe_allow_html=True)
    else:
        analyzed_jd = {"requirements": [], "keyphrases": []}

with col_right:
    st.markdown("#### 2. Candidate Resumes")
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload single or multiple candidate resume documents for evaluation."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Execute Candidate Evaluation", type="primary", use_container_width=True)

st.markdown("---")

# Execution Handler
if run_btn:
    if not jd_text.strip():
        st.error("Please provide a Job Description to initiate matching.")
    elif not uploaded_files:
        st.error("Please upload at least one candidate resume (PDF, DOCX, or TXT).")
    else:
        batch_items: List[Tuple[any, str]] = []
        for file_item in uploaded_files:
            batch_items.append((file_item, file_item.name))

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
            <div class="kpi-label">Strong Alignment</div>
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
            <div class="kpi-label">Average ATS Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_table, tab_details, tab_export = st.tabs([
        "Ranked Leaderboard",
        "Candidate Evidence Inspection",
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
                "Overall ATS Score": f"{a.match_percentage}%",
                "Fit Tier": a.fit_tier,
                "Verdict": a.verdict.verdict_badge,
                "Satisfied Requirements": f"{a.requirement_analysis.satisfied_count} / {a.requirement_analysis.total_requirements}",
                "Coverage Score": f"{a.requirement_coverage_score}%",
                "Context Similarity": f"{a.macro_semantic_score}%"
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
                            {a.match_percentage}% Match Score
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"Audit Trail & Evidence Breakdown: #{idx}. {a.filename}", expanded=(idx == 1)):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Requirement Coverage", f"{a.requirement_coverage_score}%", f"{a.requirement_analysis.satisfied_count}/{a.requirement_analysis.total_requirements} Criteria")
                    m2.metric("Macro Context Similarity", f"{a.macro_semantic_score}%")
                    m3.metric("Domain Terminology", f"{a.domain_terminology_score}%", f"{len(a.requirement_analysis.matched_keyphrases)} Term(s)")

                    st.markdown("##### Assessment Verdict")
                    st.markdown(f"**Recommendation:** {a.verdict.verdict_description}")

                    if a.verdict.strengths:
                        st.markdown("**Validated Strengths:**")
                        for str_item in a.verdict.strengths:
                            st.markdown(f"- {str_item}")

                    if a.verdict.gaps:
                        st.markdown("**Identified Discrepancies:**")
                        for gap_item in a.verdict.gaps:
                            st.markdown(f"- {gap_item}")

                    # Point-by-Point Evidence Audit
                    st.markdown("##### Point-by-Point Requirement Evidence Audit")
                    for r_idx, ev in enumerate(a.requirement_analysis.requirement_evidence_list, start=1):
                        box_class = "req-satisfied" if ev.calibrated_score >= 65.0 else ("req-partial" if ev.calibrated_score >= 35.0 else "req-unmet")
                        st.markdown(f"""
                        <div class="req-item-box {box_class}">
                            <div style="display: flex; justify-content: space-between;">
                                <b>Criteria {r_idx}: {ev.requirement_text}</b>
                                <span style="font-weight: 600;">{ev.status_label} ({ev.calibrated_score}%)</span>
                            </div>
                            <div class="evidence-quote">
                                <i>Candidate Evidence:</i> "{ev.matched_evidence_snippet}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Executive Summary
                    st.markdown("##### Executive Profile Summary")
                    st.markdown(f"""
                    <div style="background: #f8fafc; border-radius: 6px; padding: 0.9rem 1rem; border: 1px solid #e2e8f0; font-size: 0.88rem; line-height: 1.6; color: #334155;">
                        {a.executive_summary}
                    </div>
                    """, unsafe_allow_html=True)

    with tab_export:
        st.markdown("#### Evaluation Report Export")
        st.caption("Download structured candidate metrics and requirement audit trail in CSV format.")

        df_export = ExportManager.to_dataframe(artifacts)
        st.dataframe(df_export, use_container_width=True)

        csv_data = ExportManager.to_csv_bytes(df_export)
        st.download_button(
            label="Download Evaluation Dataset (CSV)",
            data=csv_data,
            file_name="universal_ats_candidate_evaluation.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

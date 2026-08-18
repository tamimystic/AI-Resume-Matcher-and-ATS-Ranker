import re
import sys
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from src.entity.artifact_entity import (
    CandidateEvaluationArtifact,
    DynamicRequirementAnalysis,
    AIVerdict
)
from src.exception.custom_exception import CustomException
from src.logger.logging import logger


class CandidateEvaluator:
    """
    Universal component for tiered candidate assessment, executive summarization,
    and recruitment recommendations across any professional domain.
    """

    def __init__(self, summary_sentences_count: int = 4):
        self.summary_sentences_count = summary_sentences_count

    def summarize_profile(self, text: str) -> str:
        """
        Extracts the most salient factual sentences from a candidate's resume across any industry.
        """
        try:
            if not text.strip():
                return "No text available for summary."

            raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
            sentences = [
                s.strip() for s in raw_sentences 
                if len(s.strip().split()) >= 5 and len(s.strip()) <= 250
            ]

            if not sentences:
                return text[:300] + "..." if len(text) > 300 else text

            if len(sentences) <= self.summary_sentences_count:
                return "\n".join([f"- {s}" for s in sentences])

            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = tfidf_matrix.sum(axis=1).A1

            top_indices = sorted(sentence_scores.argsort()[-self.summary_sentences_count:])
            selected = [sentences[i] for i in top_indices]
            return "\n".join([f"- {s}" for s in selected])
        except Exception as e:
            logger.warning(f"Summarization fallback: {str(e)}")
            fallback = [s for s in sentences[:self.summary_sentences_count]] if 'sentences' in locals() else [text[:200]]
            return "\n".join([f"- {s}" for s in fallback])

    def evaluate_verdict(
        self,
        match_percentage: float,
        req_analysis: DynamicRequirementAnalysis,
        candidate_name: str
    ) -> AIVerdict:
        """
        Produces actionable hiring recommendations based on quantitative requirement satisfaction.
        """
        try:
            strengths = []
            gaps = []

            sat_count = len(req_analysis.satisfied_requirements)
            total_req = req_analysis.total_requirements
            unmet_count = len(req_analysis.unmet_requirements)

            if req_analysis.core_score >= 70.0:
                strengths.append(f"Demonstrates strong evidence satisfying core qualifications ({req_analysis.core_score}% core match).")
            elif req_analysis.core_score >= 45.0:
                strengths.append(f"Satisfies fundamental qualifications with minor areas for review ({req_analysis.core_score}% core match).")
            else:
                gaps.append(f"Core mandatory qualifications partially unaddressed ({req_analysis.core_score}% core match).")

            if req_analysis.matched_domain_terms:
                preview = ", ".join(req_analysis.matched_domain_terms[:6])
                strengths.append(f"Demonstrates validated domain terminology: {preview}.")

            if req_analysis.preferred_score >= 60.0:
                strengths.append(f"Provides positive coverage on preferred / secondary qualifications ({req_analysis.preferred_score}%).")
            elif req_analysis.preferred_requirements_count > 0:
                gaps.append(f"Secondary / preferred qualifications have minor gaps ({req_analysis.preferred_score}%).")

            if unmet_count > 0:
                gaps.append(f"Unaddressed or missing evidence in {unmet_count} requirement area(s).")
            else:
                strengths.append("Fully satisfies all explicitly stated mandatory and preferred criteria.")

            if match_percentage >= 75.0:
                badge = "Strong Fit"
                desc = "Strong candidate alignment across core mandatory qualifications. Highly recommended for technical / primary interview."
                style = "background-color: #d1fae5; color: #065f46; border: 1px solid #10b981;"
            elif match_percentage >= 50.0:
                badge = "Moderate Fit"
                desc = "Satisfies core competencies with minor secondary gaps. Recommended for screening review."
                style = "background-color: #fef3c7; color: #92400e; border: 1px solid #f59e0b;"
            elif match_percentage >= 35.0:
                badge = "Partial Fit"
                desc = "Partial alignment with substantial unaddressed criteria. Review against priority requirements."
                style = "background-color: #fee2e2; color: #991b1b; border: 1px solid #ef4444;"
            else:
                badge = "Low Alignment"
                desc = "Limited contextual and requirement alignment with current role specifications."
                style = "background-color: #f3f4f6; color: #374151; border: 1px solid #9ca3af;"

            return AIVerdict(
                verdict_badge=badge,
                verdict_description=desc,
                badge_style=style,
                strengths=strengths,
                gaps=gaps
            )
        except Exception as e:
            logger.error(f"Error evaluating candidate verdict: {str(e)}")
            raise CustomException(e, sys)

    def build_artifact(
        self,
        filename: str,
        raw_text: str,
        match_percentage: float,
        req_analysis: DynamicRequirementAnalysis
    ) -> CandidateEvaluationArtifact:
        """
        Assembles all evaluation metrics into a unified CandidateEvaluationArtifact.
        """
        try:
            if match_percentage >= 75.0:
                tier = "Top Match"
                color = "#10b981"
            elif match_percentage >= 50.0:
                tier = "Good Match"
                color = "#f59e0b"
            else:
                tier = "Low Match"
                color = "#ef4444"

            summary = self.summarize_profile(raw_text)
            verdict = self.evaluate_verdict(match_percentage, req_analysis, filename)

            return CandidateEvaluationArtifact(
                filename=filename,
                raw_text=raw_text,
                match_percentage=match_percentage,
                fit_tier=tier,
                fit_color=color,
                core_qualifications_score=req_analysis.core_score,
                preferred_qualifications_score=req_analysis.preferred_score,
                experiential_evidence_score=req_analysis.soft_skills_score,
                requirement_analysis=req_analysis,
                verdict=verdict,
                executive_summary=summary
            )
        except Exception as e:
            logger.error(f"Failed to build evaluation artifact for {filename}: {str(e)}")
            raise CustomException(e, sys)

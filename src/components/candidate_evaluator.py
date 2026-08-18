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
    Universal component for candidate assessment, executive summarization, and recruitment recommendations.
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

            sat_count = req_analysis.satisfied_count
            total_req = req_analysis.total_requirements
            unmet_count = req_analysis.unmet_count

            if sat_count > 0:
                strengths.append(f"Provides strong semantic evidence satisfying {sat_count} of {total_req} core job criteria.")
                if req_analysis.matched_keyphrases:
                    preview = ", ".join(req_analysis.matched_keyphrases[:5])
                    strengths.append(f"Demonstrates competence in target domain terminology: {preview}.")
            else:
                gaps.append("Minimal verifiable evidence found corresponding to stated role criteria.")

            if unmet_count > 0:
                gaps.append(f"Unaddressed or partially evidenced criteria in {unmet_count} requirement area(s).")
            else:
                strengths.append("Fully addresses all explicitly stated qualification and responsibility criteria.")

            if match_percentage >= 75.0:
                badge = "Strong Fit"
                desc = "Strong overall alignment with role requirements. Recommended for primary interview."
                style = "background-color: #d1fae5; color: #065f46; border: 1px solid #10b981;"
            elif match_percentage >= 50.0:
                badge = "Moderate Fit"
                desc = "Satisfies core competencies with minor gaps. Recommended for secondary review."
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
        macro_semantic_score: float,
        terminology_score: float,
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
                requirement_coverage_score=req_analysis.coverage_score,
                macro_semantic_score=macro_semantic_score,
                domain_terminology_score=terminology_score,
                requirement_analysis=req_analysis,
                verdict=verdict,
                executive_summary=summary
            )
        except Exception as e:
            logger.error(f"Failed to build evaluation artifact for {filename}: {str(e)}")
            raise CustomException(e, sys)

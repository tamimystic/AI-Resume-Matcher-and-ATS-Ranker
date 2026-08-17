import re
import sys
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from src.entity.artifact_entity import (
    CandidateEvaluationArtifact,
    SkillGapAnalysis,
    AIVerdict
)
from src.exception.custom_exception import CustomException
from src.logger.logging import logger


class CandidateEvaluator:
    """
    Component for generating executive summaries, strength/gap assessments, and recruitment verdicts.
    """

    def __init__(self, summary_sentences_count: int = 4):
        self.summary_sentences_count = summary_sentences_count

    def summarize_profile(self, text: str) -> str:
        """
        Extracts the most salient sentences from resume text to form a concise profile summary.
        """
        try:
            if not text.strip():
                return "No text available for summary."

            raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
            sentences = [
                s.strip() for s in raw_sentences 
                if len(s.strip().split()) >= 6 and len(s.strip()) <= 250
            ]

            if not sentences:
                return text[:300] + "..." if len(text) > 300 else text

            if len(sentences) <= self.summary_sentences_count:
                return "\n".join([f"- {s}" for s in sentences])

            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = tfidf_matrix.sum(axis=1).A1

            top_indices = sorted(sentence_scores.argsort()[-self.summary_sentences_count:])
            selected_sentences = [sentences[i] for i in top_indices]
            return "\n".join([f"- {s}" for s in selected_sentences])
        except Exception as e:
            logger.warning(f"Summarization fallback triggered: {str(e)}")
            fallback = [s for s in sentences[:self.summary_sentences_count]] if 'sentences' in locals() else [text[:200]]
            return "\n".join([f"- {s}" for s in fallback])

    def evaluate_verdict(
        self,
        match_percentage: float,
        skill_gap: SkillGapAnalysis,
        candidate_name: str
    ) -> AIVerdict:
        """
        Produces actionable hiring recommendations based on quantitative matching metrics.
        """
        try:
            strengths = []
            gaps = []

            matched = skill_gap.matched_skills
            missing = skill_gap.missing_skills

            if matched:
                top_matched = ", ".join(matched[:5])
                strengths.append(f"Demonstrates validated competence in {len(matched)} target skill(s): {top_matched}.")
                if len(matched) > 5:
                    strengths.append(f"Additional relevant capabilities in {', '.join(matched[5:9])}.")
            else:
                gaps.append("Did not demonstrate any of the primary technical skills listed in the job specification.")

            if missing:
                top_missing = ", ".join(missing[:5])
                gaps.append(f"Missing {len(missing)} requirement(s): {top_missing}.")
            else:
                strengths.append("Fully covers all required technical competencies specified in the job posting.")

            if match_percentage >= 80.0:
                badge = "Strong Hire"
                desc = "Strong candidate alignment. Recommended for initial technical screening."
                style = "background-color: #d1fae5; color: #065f46; border: 1px solid #10b981;"
            elif match_percentage >= 65.0:
                badge = "Consider"
                desc = "Solid foundational profile with minor skill gaps. Recommended for secondary evaluation."
                style = "background-color: #fef3c7; color: #92400e; border: 1px solid #f59e0b;"
            elif match_percentage >= 45.0:
                badge = "Moderate Fit"
                desc = "Partial alignment. Candidate may require substantial onboarding or domain transition."
                style = "background-color: #fee2e2; color: #991b1b; border: 1px solid #ef4444;"
            else:
                badge = "Not Recommended"
                desc = "Low contextual and technical alignment with current role requirements."
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
        semantic_sim: float,
        keyword_density: float,
        skill_gap: SkillGapAnalysis
    ) -> CandidateEvaluationArtifact:
        """
        Assembles all individual evaluation metrics into a unified CandidateEvaluationArtifact.
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
            verdict = self.evaluate_verdict(match_percentage, skill_gap, filename)

            return CandidateEvaluationArtifact(
                filename=filename,
                raw_text=raw_text,
                match_percentage=match_percentage,
                fit_tier=tier,
                fit_color=color,
                semantic_score=round(semantic_sim * 100, 1),
                skill_score=round(skill_gap.coverage_ratio * 100, 1),
                keyword_score=round(keyword_density * 100, 1),
                skill_gap=skill_gap,
                verdict=verdict,
                executive_summary=summary
            )
        except Exception as e:
            logger.error(f"Failed to build evaluation artifact for {filename}: {str(e)}")
            raise CustomException(e, sys)

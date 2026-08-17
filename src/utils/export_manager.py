import sys
import pandas as pd
from typing import List
from src.entity.artifact_entity import CandidateEvaluationArtifact
from src.exception.custom_exception import CustomException
from src.logger.logging import logger


class ExportManager:
    """
    Utility component for transforming candidate evaluation artifacts into structured tabular formats.
    """

    @staticmethod
    def to_dataframe(artifacts: List[CandidateEvaluationArtifact]) -> pd.DataFrame:
        try:
            records = []
            for rank, item in enumerate(artifacts, start=1):
                records.append({
                    "Rank": rank,
                    "Candidate File": item.filename,
                    "Overall Match Score (%)": item.match_percentage,
                    "Fit Category": item.fit_tier,
                    "Recommendation Verdict": item.verdict.verdict_badge,
                    "Semantic Similarity (%)": item.semantic_score,
                    "Skill Coverage (%)": item.skill_score,
                    "Keyword Overlap (%)": item.keyword_score,
                    "Matched Skills Count": item.skill_gap.matched_skills_count,
                    "Missing Skills Count": len(item.skill_gap.missing_skills),
                    "Matched Skills": ", ".join(item.skill_gap.matched_skills),
                    "Missing Skills": ", ".join(item.skill_gap.missing_skills),
                    "Executive Summary": item.executive_summary.replace("\n", " ")
                })
            return pd.DataFrame(records)
        except Exception as e:
            logger.error(f"Failed to generate evaluation dataframe: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
        try:
            return dataframe.to_csv(index=False).encode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode CSV bytes: {str(e)}")
            raise CustomException(e, sys)

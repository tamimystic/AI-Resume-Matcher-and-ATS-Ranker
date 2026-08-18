import sys
import pandas as pd
from typing import List
from src.entity.artifact_entity import CandidateEvaluationArtifact
from src.exception.custom_exception import CustomException
from src.logger.logging import logger


class ExportManager:
    """
    Utility component for exporting candidate evaluation artifacts to structured tabular data.
    """

    @staticmethod
    def to_dataframe(artifacts: List[CandidateEvaluationArtifact]) -> pd.DataFrame:
        try:
            records = []
            for rank, item in enumerate(artifacts, start=1):
                analysis = item.requirement_analysis
                records.append({
                    "Rank": rank,
                    "Candidate File": item.filename,
                    "Overall Match Score (%)": item.match_percentage,
                    "Fit Category": item.fit_tier,
                    "Recommendation Verdict": item.verdict.verdict_badge,
                    "Requirement Coverage (%)": item.requirement_coverage_score,
                    "Macro Context Similarity (%)": item.macro_semantic_score,
                    "Domain Terminology (%)": item.domain_terminology_score,
                    "Satisfied Requirements Count": analysis.satisfied_count,
                    "Unmet Requirements Count": analysis.unmet_count,
                    "Satisfied Criteria": " | ".join(analysis.satisfied_requirements),
                    "Unmet Criteria": " | ".join(analysis.unmet_requirements),
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

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
                    "Core Requirements Score (%)": item.core_qualifications_score,
                    "Preferred Qualifications Score (%)": item.preferred_qualifications_score,
                    "Soft Skills / General Score (%)": item.experiential_evidence_score,
                    "Satisfied Criteria Count": len(analysis.satisfied_requirements),
                    "Unmet Criteria Count": len(analysis.unmet_requirements),
                    "Satisfied Criteria": " | ".join(analysis.satisfied_requirements),
                    "Unmet Criteria": " | ".join(analysis.unmet_requirements),
                    "Matched Domain Terms": ", ".join(analysis.matched_domain_terms),
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

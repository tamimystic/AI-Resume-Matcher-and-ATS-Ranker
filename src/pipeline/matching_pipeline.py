import sys
from typing import List, Tuple, Union, Callable, Optional
from src.components.document_parser import DocumentParser
from src.components.skill_extractor import SkillExtractor
from src.components.semantic_matcher import SemanticMatcher
from src.components.candidate_evaluator import CandidateEvaluator
from src.entity.config_entity import MatchingWeightsConfig, ModelConfig
from src.entity.artifact_entity import CandidateEvaluationArtifact, ParsedDocument
from src.exception.custom_exception import CustomException
from src.logger.logging import logger


class ResumeMatchingPipeline:
    """
    Enterprise pipeline orchestrator for end-to-end resume ingestion, skill extraction,
    semantic vector alignment, and candidate scoring.
    """

    def __init__(
        self,
        weights_config: MatchingWeightsConfig = MatchingWeightsConfig(),
        model_config: ModelConfig = ModelConfig()
    ):
        logger.info("Initializing ResumeMatchingPipeline")
        self.weights_config = weights_config
        self.model_config = model_config
        
        self.document_parser = DocumentParser()
        self.skill_extractor = SkillExtractor()
        self.semantic_matcher = SemanticMatcher(
            model_config=self.model_config,
            weights_config=self.weights_config
        )
        self.candidate_evaluator = CandidateEvaluator(summary_sentences_count=4)

    def extract_job_skills(self, job_description_text: str) -> List[str]:
        return self.skill_extractor.extract(job_description_text)

    def evaluate_single_resume(
        self,
        file_source: Union[bytes, str],
        filename: str,
        job_description_text: str,
        job_skills: List[str]
    ) -> CandidateEvaluationArtifact:
        """
        Processes a single resume through the end-to-end evaluation pipeline.
        """
        try:
            # 1. Parse document text
            parsed_doc: ParsedDocument = self.document_parser.extract(file_source, filename)
            
            # 2. Extract candidate skills
            candidate_skills = self.skill_extractor.extract(parsed_doc.raw_text)
            
            # 3. Analyze skill gap
            skill_gap = self.skill_extractor.analyze_gap(job_skills, candidate_skills)
            
            # 4. Compute semantic and keyword similarities
            semantic_sim = self.semantic_matcher.compute_semantic_similarity(
                job_description_text,
                parsed_doc.raw_text
            )
            keyword_density = self.semantic_matcher.compute_keyword_overlap(
                job_description_text,
                parsed_doc.raw_text
            )
            
            # 5. Calculate hybrid ATS score
            match_percentage = self.semantic_matcher.calculate_hybrid_score(
                semantic_sim=semantic_sim,
                skill_coverage_ratio=skill_gap.coverage_ratio,
                keyword_density=keyword_density,
                has_job_skills=bool(job_skills)
            )
            
            # 6. Construct final evaluation artifact
            artifact = self.candidate_evaluator.build_artifact(
                filename=parsed_doc.filename,
                raw_text=parsed_doc.raw_text,
                match_percentage=match_percentage,
                semantic_sim=semantic_sim,
                keyword_density=keyword_density,
                skill_gap=skill_gap
            )
            
            return artifact
        except Exception as e:
            logger.error(f"Error in evaluate_single_resume for {filename}: {str(e)}")
            raise CustomException(e, sys)

    def evaluate_batch(
        self,
        resume_items: List[Tuple[Union[bytes, str], str]],
        job_description_text: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[CandidateEvaluationArtifact]:
        """
        Processes multiple resumes concurrently or sequentially, returning a ranked list of candidate artifacts.
        """
        try:
            logger.info(f"Starting batch evaluation for {len(resume_items)} candidate resumes")
            
            job_skills = self.extract_job_skills(job_description_text)
            total = len(resume_items)
            results: List[CandidateEvaluationArtifact] = []

            for idx, (source, filename) in enumerate(resume_items, start=1):
                artifact = self.evaluate_single_resume(
                    file_source=source,
                    filename=filename,
                    job_description_text=job_description_text,
                    job_skills=job_skills
                )
                results.append(artifact)

                if progress_callback is not None:
                    progress_callback(idx, total, filename)

            # Sort artifacts by match_percentage descending
            results.sort(key=lambda x: x.match_percentage, reverse=True)
            logger.info("Batch evaluation completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in evaluate_batch: {str(e)}")
            raise CustomException(e, sys)

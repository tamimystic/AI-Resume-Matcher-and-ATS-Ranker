import sys
from typing import List, Tuple, Union, Callable, Optional, Dict
from src.components.document_parser import DocumentParser
from src.components.requirement_extractor import RequirementExtractor
from src.components.semantic_matcher import SemanticMatcher
from src.components.candidate_evaluator import CandidateEvaluator
from src.entity.config_entity import MatchingWeightsConfig, ModelConfig
from src.entity.artifact_entity import CandidateEvaluationArtifact, ParsedDocument
from src.exception.custom_exception import CustomException
from src.logger.logging import logger


class ResumeMatchingPipeline:
    """
    Universal, Domain-Agnostic ATS Pipeline Orchestrator.
    Handles point-by-point requirement evidence extraction and multi-factor candidate scoring.
    """

    def __init__(
        self,
        weights_config: MatchingWeightsConfig = MatchingWeightsConfig(),
        model_config: ModelConfig = ModelConfig()
    ):
        logger.info("Initializing Domain-Agnostic ResumeMatchingPipeline")
        self.weights_config = weights_config
        self.model_config = model_config

        self.document_parser = DocumentParser()
        self.requirement_extractor = RequirementExtractor()
        self.semantic_matcher = SemanticMatcher(
            model_config=self.model_config,
            weights_config=self.weights_config
        )
        self.candidate_evaluator = CandidateEvaluator(summary_sentences_count=4)

    def analyze_job_description(self, job_description_text: str) -> Dict[str, List[str]]:
        """
        Extracts atomic requirements and domain keyphrases from any Job Description.
        """
        requirements = self.requirement_extractor.extract_job_requirements(job_description_text)
        keyphrases = self.requirement_extractor.extract_domain_keyphrases(job_description_text)
        return {
            "requirements": requirements,
            "keyphrases": keyphrases
        }

    def evaluate_single_resume(
        self,
        file_source: Union[bytes, str],
        filename: str,
        job_description_text: str,
        job_requirements: List[str],
        job_keyphrases: List[str]
    ) -> CandidateEvaluationArtifact:
        """
        Evaluates a single candidate resume against extracted job criteria.
        """
        try:
            # 1. Parse document and extract semantic passages
            parsed_doc: ParsedDocument = self.document_parser.extract(file_source, filename)
            passages = self.requirement_extractor.chunk_document_into_passages(parsed_doc.raw_text)

            # 2. Point-by-Point Requirement Evidence Evaluation
            req_analysis = self.semantic_matcher.evaluate_requirements(
                job_requirements=job_requirements,
                resume_passages=passages,
                extracted_keyphrases=job_keyphrases,
                resume_text=parsed_doc.raw_text
            )

            # 3. Macro Context and Terminology Scores
            macro_sim = self.semantic_matcher.compute_macro_semantic_similarity(
                job_description_text,
                parsed_doc.raw_text
            )
            terminology_score = self.semantic_matcher.compute_terminology_score(
                matched_kp_count=len(req_analysis.matched_keyphrases),
                total_kp_count=len(job_keyphrases)
            )

            # 4. Calculate Unified ATS Score
            match_percentage = self.semantic_matcher.calculate_unified_ats_score(
                requirement_coverage=req_analysis.coverage_score,
                macro_semantic_score=macro_sim,
                terminology_score=terminology_score
            )

            # 5. Build Artifact
            artifact = self.candidate_evaluator.build_artifact(
                filename=parsed_doc.filename,
                raw_text=parsed_doc.raw_text,
                match_percentage=match_percentage,
                macro_semantic_score=macro_sim,
                terminology_score=terminology_score,
                req_analysis=req_analysis
            )

            return artifact
        except Exception as e:
            logger.error(f"Error evaluating single resume {filename}: {str(e)}")
            raise CustomException(e, sys)

    def evaluate_batch(
        self,
        resume_items: List[Tuple[Union[bytes, str], str]],
        job_description_text: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[CandidateEvaluationArtifact]:
        """
        Processes a batch of resumes against a target job description.
        """
        try:
            logger.info(f"Initiating universal batch matching for {len(resume_items)} candidates")
            jd_data = self.analyze_job_description(job_description_text)
            job_requirements = jd_data["requirements"]
            job_keyphrases = jd_data["keyphrases"]

            total = len(resume_items)
            results: List[CandidateEvaluationArtifact] = []

            for idx, (source, fname) in enumerate(resume_items, start=1):
                artifact = self.evaluate_single_resume(
                    file_source=source,
                    filename=fname,
                    job_description_text=job_description_text,
                    job_requirements=job_requirements,
                    job_keyphrases=job_keyphrases
                )
                results.append(artifact)

                if progress_callback is not None:
                    progress_callback(idx, total, fname)

            results.sort(key=lambda x: x.match_percentage, reverse=True)
            logger.info("Batch matching completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in evaluate_batch: {str(e)}")
            raise CustomException(e, sys)

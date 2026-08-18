import re
import sys
import numpy as np
from typing import List, Tuple, Dict, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.entity.config_entity import ModelConfig, MatchingWeightsConfig
from src.entity.artifact_entity import RequirementEvidence, DynamicRequirementAnalysis
from src.exception.custom_exception import CustomException
from src.logger.logging import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticMatcher:
    """
    Universal, Domain-Agnostic Hybrid Matching Engine.
    Combines Dense Vector Embeddings with Lexical Entity, Content-Token Recall, and Passage-Level
    Evidence Retrieval across all professional domains.
    """

    def __init__(self, model_config: ModelConfig = ModelConfig(), weights_config: MatchingWeightsConfig = MatchingWeightsConfig()):
        self.model_config = model_config
        self.weights_config = weights_config
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading SentenceTransformer [{self.model_config.model_name}] on [{self.model_config.device}]")
                self.model = SentenceTransformer(self.model_config.model_name, device=self.model_config.device)
            except Exception as e:
                logger.warning(f"SentenceTransformer load warning ({str(e)}). Using Lexical & TF-IDF hybrid engine.")
                self.model = None
        else:
            logger.info("Operating in Lexical & TF-IDF hybrid matching mode.")
            self.model = None

    @staticmethod
    def _extract_content_tokens(text: str) -> Set[str]:
        """Extracts content words, abbreviations, and technical symbols."""
        if not text:
            return set()
        raw_tokens = set(re.findall(r'\b[\w\+\#\.\-]{2,}\b', text.lower()))
        stopwords = {
            'the', 'and', 'for', 'with', 'in', 'of', 'at', 'least', 'one', 'e.g', 'eg',
            'or', 'to', 'able', 'being', 'having', 'understanding', 'solid', 'basic',
            'knowledge', 'experience', 'familiarity', 'comfortable', 'such', 'as', 'an',
            'you', 'your', 'our', 'will', 'must', 'have', 'good', 'etc', 'working',
            'plus', 'across', 'using', 'from', 'into', 'without', 'when', 'needed',
            'skills', 'expertise', 'requirements', 'qualifications', 'responsibilities'
        }
        return {t for t in raw_tokens if t not in stopwords and not t.isdigit()}

    def _compute_dense_similarity_matrix(self, list_a: List[str], list_b: List[str]) -> np.ndarray:
        """Computes dense vector cosine similarity matrix if transformer model is active."""
        if not list_a or not list_b:
            return np.zeros((len(list_a), len(list_b)))

        if self.model is not None:
            try:
                emb_a = self.model.encode(list_a, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                emb_b = self.model.encode(list_b, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                return np.dot(emb_a, emb_b.T)
            except Exception as e:
                logger.warning(f"Dense vector encoding error: {str(e)}")

        # Fallback to joint TF-IDF matrix
        try:
            all_texts = list_a + list_b
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, token_pattern=r'(?u)\b[\w\+\#\.\-]{2,}\b')
            matrix = vectorizer.fit_transform(all_texts)
            mat_a = matrix[:len(list_a)]
            mat_b = matrix[len(list_a):]
            return cosine_similarity(mat_a, mat_b)
        except Exception:
            return np.zeros((len(list_a), len(list_b)))

    def _evaluate_single_requirement_evidence(
        self,
        requirement: str,
        resume_text_lower: str,
        passages: List[str],
        sim_row: np.ndarray
    ) -> Tuple[float, str, float]:
        """
        Evaluates evidence for a single requirement across passage similarities and whole-document entity recall.
        """
        req_tokens = self._extract_content_tokens(requirement)
        
        # 1. Identify best passage from vector similarity
        best_passage_idx = int(np.argmax(sim_row)) if len(sim_row) > 0 else 0
        best_raw_sim = float(sim_row[best_passage_idx]) if len(sim_row) > 0 else 0.0
        best_passage = passages[best_passage_idx] if passages else ""

        # 2. Token Recall against best passage and full resume
        matched_tokens_passage = 0
        matched_tokens_full = 0

        passage_lower = best_passage.lower()
        for token in req_tokens:
            if token in passage_lower:
                matched_tokens_passage += 1
            if token in resume_text_lower:
                matched_tokens_full += 1

        total_req_tokens = len(req_tokens) if req_tokens else 1
        passage_recall = matched_tokens_passage / total_req_tokens
        full_recall = matched_tokens_full / total_req_tokens

        # 3. Dense Vector Calibration
        if self.model is not None:
            calibrated_dense = max(0.0, (best_raw_sim - 0.20) / 0.45) if best_raw_sim > 0.20 else 0.0
        else:
            calibrated_dense = min(1.0, best_raw_sim * 1.8)

        # 4. Hybrid Synthesis
        # If strong token recall in document (>60% or >=2 key terms)
        if full_recall >= 0.60 or (matched_tokens_full >= 2 and total_req_tokens <= 4):
            token_score = min(1.0, 0.70 + (full_recall * 0.30))
        elif full_recall >= 0.30 or matched_tokens_full >= 1:
            token_score = 0.50 + (full_recall * 0.35)
        else:
            token_score = full_recall * 0.50

        final_item_score = max(calibrated_dense, token_score, (calibrated_dense * 0.4 + token_score * 0.6))
        final_item_score = max(0.0, min(1.0, float(final_item_score)))

        return final_item_score, best_passage, best_raw_sim

    def evaluate_requirements(
        self,
        job_requirements: List[str],
        resume_passages: List[str],
        extracted_keyphrases: List[str],
        resume_text: str
    ) -> DynamicRequirementAnalysis:
        """
        Performs fine-grained requirement evidence retrieval across candidate resume passages.
        """
        try:
            if not job_requirements:
                job_requirements = ["General professional qualifications and domain competencies."]

            if not resume_passages:
                resume_passages = [resume_text] if resume_text else ["No content available."]

            similarity_matrix = self._compute_dense_similarity_matrix(job_requirements, resume_passages)
            resume_text_lower = resume_text.lower()

            evidence_list: List[RequirementEvidence] = []
            satisfied_requirements: List[str] = []
            unmet_requirements: List[str] = []

            satisfied_count = 0
            partial_count = 0
            unmet_count = 0
            calibrated_score_sum = 0.0

            for i, req in enumerate(job_requirements):
                sim_row = similarity_matrix[i] if i < len(similarity_matrix) else np.zeros(len(resume_passages))
                
                score, best_snippet, raw_sim = self._evaluate_single_requirement_evidence(
                    requirement=req,
                    resume_text_lower=resume_text_lower,
                    passages=resume_passages,
                    sim_row=sim_row
                )
                
                calibrated_score_sum += score

                if score >= 0.65:
                    status = "Satisfied"
                    is_sat = True
                    satisfied_count += 1
                    satisfied_requirements.append(req)
                elif score >= 0.35:
                    status = "Partial Evidence"
                    is_sat = False
                    partial_count += 1
                else:
                    status = "Unmet / Missing"
                    is_sat = False
                    unmet_count += 1
                    unmet_requirements.append(req)

                evidence_list.append(RequirementEvidence(
                    requirement_text=req,
                    matched_evidence_snippet=best_snippet,
                    raw_similarity=raw_sim,
                    calibrated_score=round(score * 100, 1),
                    is_satisfied=is_sat,
                    status_label=status
                ))

            coverage_ratio = calibrated_score_sum / len(job_requirements) if job_requirements else 0.0

            # Keyphrase Overlap
            matched_kp = []
            missing_kp = []
            for kp in extracted_keyphrases:
                kp_clean = kp.lower().strip()
                if kp_clean in resume_text_lower or any(part in resume_text_lower for part in kp_clean.split() if len(part) >= 4):
                    matched_kp.append(kp)
                else:
                    missing_kp.append(kp)

            return DynamicRequirementAnalysis(
                total_requirements=len(job_requirements),
                satisfied_count=satisfied_count,
                partial_count=partial_count,
                unmet_count=unmet_count,
                requirement_evidence_list=evidence_list,
                satisfied_requirements=satisfied_requirements,
                unmet_requirements=unmet_requirements,
                coverage_score=round(coverage_ratio * 100, 1),
                extracted_keyphrases=extracted_keyphrases,
                matched_keyphrases=matched_kp,
                missing_keyphrases=missing_kp
            )
        except Exception as e:
            logger.error(f"Error in evaluate_requirements: {str(e)}")
            raise CustomException(e, sys)

    def compute_macro_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Computes macro document contextual similarity."""
        try:
            if not text_a.strip() or not text_b.strip():
                return 0.0

            sim_matrix = self._compute_dense_similarity_matrix([text_a], [text_b])
            raw_sim = float(sim_matrix[0][0])
            
            tokens_a = self._extract_content_tokens(text_a)
            tokens_b = self._extract_content_tokens(text_b)
            overlap = len(tokens_a.intersection(tokens_b)) / len(tokens_a) if tokens_a else 0.0

            if self.model is not None:
                calibrated = max(0.0, min(1.0, (raw_sim - 0.20) / 0.45))
                combined = max(calibrated, overlap)
            else:
                combined = max(raw_sim * 1.6, overlap * 1.3)

            return round(min(100.0, combined * 100), 1)
        except Exception as e:
            logger.warning(f"Macro semantic similarity error: {str(e)}")
            return 0.0

    def compute_terminology_score(self, matched_kp_count: int, total_kp_count: int) -> float:
        if total_kp_count == 0:
            return 100.0
        return round(min(100.0, (matched_kp_count / total_kp_count) * 100), 1)

    def calculate_unified_ats_score(
        self,
        requirement_coverage: float,
        macro_semantic_score: float,
        terminology_score: float
    ) -> float:
        """
        Calculates unified ATS match score using configured domain-agnostic weights.
        """
        try:
            total_w = (
                self.weights_config.requirement_weight +
                self.weights_config.macro_semantic_weight +
                self.weights_config.terminology_weight
            )

            if total_w > 0:
                w_req = self.weights_config.requirement_weight / total_w
                w_macro = self.weights_config.macro_semantic_weight / total_w
                w_term = self.weights_config.terminology_weight / total_w
            else:
                w_req, w_macro, w_term = 0.55, 0.25, 0.20

            final_score = (
                (w_req * requirement_coverage) +
                (w_macro * macro_semantic_score) +
                (w_term * terminology_score)
            )

            return max(0.0, min(100.0, round(final_score, 1)))
        except Exception as e:
            logger.error(f"Error calculating unified ATS score: {str(e)}")
            raise CustomException(e, sys)

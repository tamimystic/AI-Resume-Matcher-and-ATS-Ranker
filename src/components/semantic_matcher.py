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
    Tiered, Cluster-Aware Universal Semantic Matching Engine.
    Combines Dense Neural Vector Embeddings, Passage Evidence Retrieval,
    and Disjunctive (OR) Entity Fulfillment to evaluate candidate alignment.
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
                logger.warning(f"SentenceTransformer load warning ({str(e)}). Operating in Lexical/TF-IDF hybrid mode.")
                self.model = None
        else:
            logger.info("SentenceTransformer not present. Operating in Lexical/TF-IDF hybrid mode.")
            self.model = None

    @staticmethod
    def _extract_content_tokens(text: str) -> Set[str]:
        """Extracts informative content words, acronyms, and technical symbols."""
        if not text:
            return set()
        raw_tokens = set(re.findall(r'\b[\w\+\#\.\-]{2,}\b', text.lower()))
        stopwords = {
            'the', 'and', 'for', 'with', 'in', 'of', 'at', 'least', 'one', 'e.g', 'eg',
            'or', 'to', 'able', 'being', 'having', 'understanding', 'solid', 'basic',
            'knowledge', 'experience', 'familiarity', 'comfortable', 'such', 'as', 'an',
            'you', 'your', 'our', 'will', 'must', 'have', 'good', 'etc', 'working',
            'plus', 'across', 'using', 'from', 'into', 'without', 'when', 'needed',
            'skills', 'expertise', 'requirements', 'qualifications', 'responsibilities',
            'role', 'job', 'position', 'duties', 'minimum'
        }
        return {t for t in raw_tokens if t not in stopwords and not t.isdigit()}

    def _compute_dense_similarity_matrix(self, list_a: List[str], list_b: List[str]) -> np.ndarray:
        """Computes pairwise cosine similarity matrix using dense vectors or joint TF-IDF."""
        if not list_a or not list_b:
            return np.zeros((len(list_a), len(list_b)))

        if self.model is not None:
            try:
                emb_a = self.model.encode(list_a, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                emb_b = self.model.encode(list_b, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                return np.dot(emb_a, emb_b.T)
            except Exception as e:
                logger.warning(f"Dense vector encoding error: {str(e)}")

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
        category: str,
        resume_text_lower: str,
        passages: List[str],
        sim_row: np.ndarray
    ) -> Tuple[float, str, float]:
        """
        Evaluates evidence for a single requirement across passage similarities and whole-document entity recall.
        Handles cluster (OR) options automatically.
        """
        req_tokens = self._extract_content_tokens(requirement)
        
        best_passage_idx = int(np.argmax(sim_row)) if len(sim_row) > 0 else 0
        best_raw_sim = float(sim_row[best_passage_idx]) if len(sim_row) > 0 else 0.0
        best_passage = passages[best_passage_idx] if passages else ""

        # Check token matches in document
        matched_tokens_full = [t for t in req_tokens if t in resume_text_lower]
        total_req_tokens = len(req_tokens) if req_tokens else 1
        full_recall = len(matched_tokens_full) / total_req_tokens

        # Check for Disjunctive / Alternative options (e.g., Python or Java or PHP)
        # If requirement lists alternatives (commas or 'or'), matching key entities satisfies the requirement
        has_alternatives = "," in requirement or " or " in requirement or "e.g." in requirement
        if has_alternatives and len(matched_tokens_full) >= 2:
            full_recall = max(full_recall, 0.85)

        # Dense model calibration (sigmoid-style mapping)
        if self.model is not None:
            calibrated_dense = max(0.0, (best_raw_sim - 0.20) / 0.45) if best_raw_sim > 0.20 else 0.0
        else:
            calibrated_dense = min(1.0, best_raw_sim * 1.8)

        # Token score synthesis
        if full_recall >= 0.50 or len(matched_tokens_full) >= 2:
            token_score = min(1.0, 0.70 + (full_recall * 0.30))
        elif full_recall >= 0.25 or len(matched_tokens_full) >= 1:
            token_score = 0.50 + (full_recall * 0.35)
        else:
            token_score = full_recall * 0.40

        # For Soft Skills, ensure reasonable baseline so absence of conversational phrases doesn't crush score
        if category == "Soft Skills / General":
            if full_recall > 0 or calibrated_dense > 0.30:
                final_item_score = max(0.75, token_score)
            else:
                final_item_score = 0.50
        else:
            final_item_score = max(calibrated_dense, token_score, (calibrated_dense * 0.4 + token_score * 0.6))

        final_item_score = max(0.0, min(1.0, float(final_item_score)))
        return final_item_score, best_passage, best_raw_sim

    def evaluate_tiered_requirements(
        self,
        categorized_requirements: List[Tuple[str, str]],
        resume_passages: List[str],
        extracted_keyphrases: List[str],
        resume_text: str
    ) -> DynamicRequirementAnalysis:
        """
        Performs tiered requirement evaluation across Core, Preferred, and Soft Skill categories.
        """
        try:
            if not categorized_requirements:
                categorized_requirements = [("General qualifications and professional domain competence.", "Core / Mandatory")]

            if not resume_passages:
                resume_passages = [resume_text] if resume_text else ["No content available."]

            req_texts = [r[0] for r in categorized_requirements]
            similarity_matrix = self._compute_dense_similarity_matrix(req_texts, resume_passages)
            resume_text_lower = resume_text.lower()

            evidence_list: List[RequirementEvidence] = []
            satisfied_requirements: List[str] = []
            unmet_requirements: List[str] = []

            core_scores: List[float] = []
            pref_scores: List[float] = []
            soft_scores: List[float] = []

            satisfied_count = 0
            partial_count = 0
            unmet_count = 0

            for i, (req, cat) in enumerate(categorized_requirements):
                sim_row = similarity_matrix[i] if i < len(similarity_matrix) else np.zeros(len(resume_passages))
                
                score, best_snippet, raw_sim = self._evaluate_single_requirement_evidence(
                    requirement=req,
                    category=cat,
                    resume_text_lower=resume_text_lower,
                    passages=resume_passages,
                    sim_row=sim_row
                )

                if cat == "Core / Mandatory":
                    core_scores.append(score)
                elif cat == "Preferred / Good-to-Have":
                    pref_scores.append(score)
                else:
                    soft_scores.append(score)

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
                    category=cat,
                    matched_evidence_snippet=best_snippet,
                    raw_similarity=raw_sim,
                    calibrated_score=round(score * 100, 1),
                    is_satisfied=is_sat,
                    status_label=status
                ))

            # Tier-Specific Averages
            core_avg = (sum(core_scores) / len(core_scores)) if core_scores else 0.80
            pref_avg = (sum(pref_scores) / len(pref_scores)) if pref_scores else 0.70
            soft_avg = (sum(soft_scores) / len(soft_scores)) if soft_scores else 0.75

            # Dynamic Overall Coverage
            total_reqs = len(categorized_requirements)
            all_scores = core_scores + pref_scores + soft_scores
            overall_cov = (sum(all_scores) / len(all_scores)) if all_scores else 0.0

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
                total_requirements=total_reqs,
                core_requirements_count=len(core_scores),
                preferred_requirements_count=len(pref_scores),
                soft_skills_count=len(soft_scores),
                core_score=round(core_avg * 100, 1),
                preferred_score=round(pref_avg * 100, 1),
                soft_skills_score=round(soft_avg * 100, 1),
                overall_coverage_score=round(overall_cov * 100, 1),
                requirement_evidence_list=evidence_list,
                satisfied_requirements=satisfied_requirements,
                unmet_requirements=unmet_requirements,
                matched_domain_terms=matched_kp,
                missing_domain_terms=missing_kp
            )
        except Exception as e:
            logger.error(f"Error in evaluate_tiered_requirements: {str(e)}")
            raise CustomException(e, sys)

    def calculate_unified_ats_score(
        self,
        core_score: float,
        preferred_score: float,
        soft_skills_score: float
    ) -> float:
        """
        Calculates unified ATS match score using configured tiered weights.
        """
        try:
            total_w = (
                self.weights_config.core_requirements_weight +
                self.weights_config.preferred_qualifications_weight +
                self.weights_config.experiential_evidence_weight
            )

            if total_w > 0:
                w_core = self.weights_config.core_requirements_weight / total_w
                w_pref = self.weights_config.preferred_qualifications_weight / total_w
                w_soft = self.weights_config.experiential_evidence_weight / total_w
            else:
                w_core, w_pref, w_soft = 0.60, 0.25, 0.15

            final_score = (
                (w_core * core_score) +
                (w_pref * preferred_score) +
                (w_soft * soft_skills_score)
            )

            return max(0.0, min(100.0, round(final_score, 1)))
        except Exception as e:
            logger.error(f"Error calculating unified ATS score: {str(e)}")
            raise CustomException(e, sys)

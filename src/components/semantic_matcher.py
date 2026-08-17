import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.entity.config_entity import ModelConfig, MatchingWeightsConfig
from src.exception.custom_exception import CustomException
from src.logger.logging import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticMatcher:
    """
    Component for evaluating contextual semantic similarity, keyword density, and hybrid ATS scoring.
    """

    def __init__(self, model_config: ModelConfig = ModelConfig(), weights_config: MatchingWeightsConfig = MatchingWeightsConfig()):
        self.model_config = model_config
        self.weights_config = weights_config
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading SentenceTransformer model [{self.model_config.model_name}] on device [{self.model_config.device}]")
                self.model = SentenceTransformer(self.model_config.model_name, device=self.model_config.device)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer ({str(e)}). Fallback vectorizer will be used.")
                self.model = None
        else:
            logger.info("SentenceTransformer not available. Using TF-IDF vectorizer.")
            self.model = None

    def compute_semantic_similarity(self, text_a: str, text_b: str) -> float:
        try:
            if not text_a.strip() or not text_b.strip():
                return 0.0

            if self.model is not None:
                embeddings = self.model.encode([text_a, text_b], convert_to_numpy=True, normalize_embeddings=True)
                similarity = float(np.dot(embeddings[0], embeddings[1]))
                return max(0.0, min(1.0, similarity))

            return self._compute_tfidf_similarity(text_a, text_b)
        except Exception as e:
            logger.error(f"Semantic similarity computation error: {str(e)}")
            return self._compute_tfidf_similarity(text_a, text_b)

    def _compute_tfidf_similarity(self, text_a: str, text_b: str) -> float:
        try:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
            tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(max(0.0, min(1.0, sim)))
        except Exception as e:
            logger.warning(f"TF-IDF similarity calculation fallback error: {str(e)}")
            return 0.0

    def compute_keyword_overlap(self, job_text: str, resume_text: str) -> float:
        return self._compute_tfidf_similarity(job_text, resume_text)

    def calculate_hybrid_score(
        self,
        semantic_sim: float,
        skill_coverage_ratio: float,
        keyword_density: float,
        has_job_skills: bool = True
    ) -> float:
        """
        Combines semantic similarity, skill coverage, and keyword density into a unified 0-100 score.
        """
        try:
            total_weight = (
                self.weights_config.semantic_weight +
                self.weights_config.skills_weight +
                self.weights_config.keywords_weight
            )
            
            if total_weight > 0:
                w_sem = self.weights_config.semantic_weight / total_weight
                w_sk = self.weights_config.skills_weight / total_weight
                w_kw = self.weights_config.keywords_weight / total_weight
            else:
                w_sem, w_sk, w_kw = 0.45, 0.40, 0.15

            if not has_job_skills:
                raw_score = (semantic_sim * 0.70) + (keyword_density * 0.30)
            else:
                raw_score = (w_sem * semantic_sim) + (w_sk * skill_coverage_ratio) + (w_kw * keyword_density)

            score_percentage = round(float(raw_score * 100), 1)
            return max(0.0, min(100.0, score_percentage))
        except Exception as e:
            logger.error(f"Error calculating hybrid score: {str(e)}")
            raise CustomException(e, sys)

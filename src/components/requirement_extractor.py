import re
import sys
from typing import List, Set, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from src.exception.custom_exception import CustomException
from src.logger.logging import logger


class RequirementExtractor:
    """
    Universal, Domain-Agnostic Component for Tiered Requirement Categorization
    and Dynamic Keyphrase Extraction across all professional disciplines.
    """

    def __init__(self):
        logger.info("Initializing Tiered RequirementExtractor component")

    @staticmethod
    def chunk_document_into_passages(text: str) -> List[str]:
        """
        Splits text into meaningful semantic passages (sentences, bullet points, lines)
        for fine-grained evidence retrieval.
        """
        if not text:
            return []

        raw_lines = re.split(r'[\r\n]+', text)
        passages = []

        for line in raw_lines:
            line_clean = line.strip()
            line_clean = re.sub(r'^[•\-\*\d+\.\)]+\s*', '', line_clean).strip()
            
            if len(line_clean.split()) >= 3:
                sub_sentences = re.split(r'(?<=[.!?])\s+', line_clean)
                for sent in sub_sentences:
                    sent_strip = sent.strip()
                    if len(sent_strip.split()) >= 3:
                        passages.append(sent_strip)
            elif len(line_clean) > 0:
                passages.append(line_clean)

        return passages if passages else [text]

    @staticmethod
    def _classify_requirement_tier(line_text: str, current_section_header: str) -> str:
        """
        Categorizes a requirement line into:
        - 'Core / Mandatory'
        - 'Preferred / Good-to-Have'
        - 'Soft Skills / General'
        """
        lower = line_text.lower()
        header_lower = current_section_header.lower()

        # Check section header context first
        if "good to have" in header_lower or "preferred" in header_lower or "nice to have" in header_lower or "bonus" in header_lower:
            return "Preferred / Good-to-Have"
        
        # Soft skills / behavioral indicators
        soft_keywords = [
            "communication", "ownership", "mindset", "proactively", "self-directed",
            "team player", "interpersonal", "punctual", "dependable", "document your work",
            "collaborate", "problem-solve", "adaptable", "work independently", "motivated"
        ]
        if any(sk in lower for sk in soft_keywords) and not any(hard in lower for hard in ["degree", "bachelor", "bsc", "license", "years", "sql", "python", "cdl", "pharm"]):
            return "Soft Skills / General"

        # Explicit preferred markers within text
        if "good to have" in lower or "preferred" in lower or "plus" in lower or "nice to have" in lower:
            return "Preferred / Good-to-Have"

        # Default to Core / Mandatory
        return "Core / Mandatory"

    def extract_categorized_job_requirements(self, job_description_text: str) -> List[Tuple[str, str]]:
        """
        Extracts individual role requirements and categorizes each as (Requirement_Text, Tier_Category).
        Systematically filters administrative headers and company boilerplates.
        """
        try:
            if not job_description_text.strip():
                return []

            lines = re.split(r'[\r\n]+', job_description_text)
            extracted: List[Tuple[str, str]] = []
            
            boilerplate_patterns = [
                r'^(about us|company description|about the company|about the role|responsibilities & context|skills & expertise|other relevant skills|additional requirements|job details)\b',
                r'^(we are|we\'re looking|our company is|who we are|what you\'ll do at)\b',
                r'^(location|duration|job type|job mode|salary|experience level|deadline|education)\b',
                r'^(equal opportunity employer|benefits|what we offer|perks)\b'
            ]

            current_section_header = "General Requirements"

            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue

                cleaned_line = re.sub(r'^[•\-\*\d+\.\)]+\s*', '', line_str).strip()
                cleaned_lower = cleaned_line.lower().rstrip(":")

                # Detect Section Headers
                if cleaned_lower in ["must have", "mandatory requirements", "requirements", "minimum qualifications", "key responsibilities", "responsibilities"]:
                    current_section_header = "Must Have"
                    continue
                elif cleaned_lower in ["good to have", "preferred qualifications", "nice to have", "bonus points", "preferred skills"]:
                    current_section_header = "Good to Have"
                    continue

                # Filter out pure boilerplate and header strings
                if any(re.search(pat, cleaned_lower) for pat in boilerplate_patterns):
                    continue
                if "is a forward-thinking" in cleaned_lower or "company that helps" in cleaned_lower:
                    continue

                words = cleaned_line.split()
                if 3 <= len(words) <= 45:
                    tier = self._classify_requirement_tier(cleaned_line, current_section_header)
                    extracted.append((cleaned_line, tier))

            if not extracted:
                raw_sentences = re.split(r'(?<=[.!?])\s+', job_description_text)
                for s in raw_sentences:
                    s_clean = s.strip()
                    if len(s_clean.split()) >= 4:
                        extracted.append((s_clean, "Core / Mandatory"))

            return extracted[:30]
        except Exception as e:
            logger.error(f"Error in extract_categorized_job_requirements: {str(e)}")
            raise CustomException(e, sys)

    def extract_domain_keyphrases(self, text: str, max_keyphrases: int = 25) -> List[str]:
        """
        Extracts salient domain keyphrases dynamically using TF-IDF n-grams across any discipline.
        """
        try:
            if not text.strip():
                return []

            clean = re.sub(r'[^\w\s\+\#\.\-]', ' ', text)
            
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 3),
                max_features=150,
                token_pattern=r'(?u)\b[\w\+\#\.\-]{2,}\b'
            )
            
            tfidf_matrix = vectorizer.fit_transform([clean])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            ranked_indices = scores.argsort()[::-1]
            extracted = []
            seen: Set[str] = set()

            for idx in ranked_indices:
                phrase = feature_names[idx].strip()
                if phrase.isdigit() or len(phrase) <= 2:
                    continue
                if phrase.lower() not in seen:
                    seen.add(phrase.lower())
                    extracted.append(phrase.title())
                if len(extracted) >= max_keyphrases:
                    break

            return extracted
        except Exception as e:
            logger.warning(f"Domain keyphrase extraction fallback: {str(e)}")
            return []

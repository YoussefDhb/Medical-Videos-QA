"""
Sparse Search Module (BM25)

BM25-based lexical retrieval for keyword matching and medical terminology.
"""
import numpy as np
import re
from rank_bm25 import BM25Okapi
from typing import List, Dict


class BM25Search:
    """
    BM25 sparse retrieval engine for keyword matching.

    Args:
        segments_data: List of segment dictionaries with 'text' and 'segment_id' fields
        k1: BM25 term frequency saturation parameter (default: 1.5)
        b: BM25 length normalization parameter (default: 0.75)
    """

    def __init__(self, segments_data: List[Dict], k1: float = 1.5, b: float = 0.75):
        self.segments_data = segments_data
        self.k1 = k1
        self.b = b

        # Build BM25 index
        print(f"Building BM25 index for {len(segments_data)} segments...")
        self.corpus = [seg.get('text', '') for seg in segments_data]
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)

        # Create segment_id to index mapping
        self.segment_id_to_idx = {
            seg.get('segment_id', seg.get('video_id', f"seg_{i}")): i
            for i, seg in enumerate(segments_data)
        }

        print(f"BM25 index built with {len(self.corpus)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        Preserves medical terms and handles hyphenation.
        """
        if not text:
            return []

        # Convert to lowercase
        text = text.lower()

        # Split on whitespace and punctuation, but preserve hyphenated terms
        tokens = re.findall(r'\b[\w-]+\b', text)

        return tokens

    def search(self, query: str, top_k: int = 100, expand_query: bool = False) -> List[Dict]:
        """
        Search using BM25 sparse retrieval.

        Args:
            query: Search query
            top_k: Number of results to return
            expand_query: Whether to expand query with medical synonyms (requires QueryExpander)

        Returns:
            List of results with segment_id and bm25_score
        """
        query_to_use = query

        # Tokenize query
        tokenized_query = self._tokenize(query_to_use)

        # Get BM25 scores for all documents
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k results
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:  # Only include documents with non-zero scores
                results.append({
                    'segment_id': self.segments_data[idx].get('segment_id',
                                                               self.segments_data[idx].get('video_id', f"seg_{idx}")),
                    'bm25_score': float(bm25_scores[idx]),
                    'segment_idx': int(idx),
                    'metadata': self.segments_data[idx]
                })

        return results


class MedicalQueryExpander:
    """
    Medical query expansion using UMLS, SNOMED CT, and common medical terminology.
    """

    # Comprehensive medical term expansion dictionary
    MEDICAL_EXPANSIONS = {
        # Cancer and Oncology
        "mouth cancer": ["oral cancer", "oral cavity cancer", "oropharyngeal cancer", "mouth tumor"],
        "cancer": ["tumor", "malignancy", "carcinoma", "neoplasm", "oncology", "malignant growth"],
        "breast cancer": ["mammary carcinoma", "breast tumor", "breast malignancy"],
        "lung cancer": ["pulmonary carcinoma", "bronchogenic carcinoma", "lung tumor"],
        "skin cancer": ["melanoma", "basal cell carcinoma", "squamous cell carcinoma", "dermatologic cancer"],
        "tumor": ["neoplasm", "growth", "mass", "lesion", "malignancy"],

        # Cardiovascular
        "heart attack": ["myocardial infarction", "MI", "cardiac infarction", "coronary thrombosis"],
        "heart disease": ["cardiovascular disease", "cardiac disease", "coronary artery disease", "CAD"],
        "stroke": ["cerebrovascular accident", "CVA", "brain attack", "cerebral infarction"],
        "blood pressure": ["hypertension", "BP", "arterial pressure", "blood pressure level"],
        "high blood pressure": ["hypertension", "HTN", "elevated blood pressure", "arterial hypertension"],
        "heart failure": ["cardiac failure", "congestive heart failure", "CHF", "cardiac insufficiency"],
        "arrhythmia": ["irregular heartbeat", "cardiac dysrhythmia", "abnormal heart rhythm"],

        # Diabetes and Endocrine
        "diabetes": ["diabetic", "blood sugar", "glucose", "diabetes mellitus", "hyperglycemia"],
        "type 1 diabetes": ["insulin dependent diabetes", "IDDM", "juvenile diabetes"],
        "type 2 diabetes": ["non-insulin dependent diabetes", "NIDDM", "adult onset diabetes"],
        "thyroid": ["thyroid gland", "thyroid disorder", "thyroid disease"],

        # Respiratory
        "asthma": ["bronchial asthma", "reactive airway disease", "asthmatic condition"],
        "COPD": ["chronic obstructive pulmonary disease", "emphysema", "chronic bronchitis"],
        "pneumonia": ["lung infection", "pulmonary infection", "pneumonitis"],
        "breathing": ["respiration", "respiratory", "ventilation", "breath"],

        # Infectious Diseases
        "infection": ["inflammatory", "sepsis", "contamination", "infectious disease", "pathogen"],
        "bacterial infection": ["bacterial disease", "bacteremia", "bacterial contamination"],
        "viral infection": ["virus", "viral disease", "viremia"],
        "flu": ["influenza", "viral flu", "influenza virus"],
        "COVID": ["coronavirus", "SARS-CoV-2", "COVID-19", "coronavirus disease"],

        # Neurological
        "alzheimer": ["dementia", "cognitive decline", "alzheimer disease", "memory loss"],
        "parkinson": ["parkinson disease", "PD", "parkinsonian syndrome"],
        "seizure": ["convulsion", "epileptic seizure", "fit", "epilepsy"],
        "migraine": ["headache", "severe headache", "migrainous headache"],
        "headache": ["cephalgia", "head pain", "cranial pain"],

        # Gastrointestinal
        "stomach": ["gastric", "abdominal", "belly", "gastrointestinal"],
        "diarrhea": ["loose stools", "loose bowel movements", "gastroenteritis"],
        "constipation": ["irregular bowels", "difficulty passing stool", "obstipation"],
        "ulcer": ["peptic ulcer", "gastric ulcer", "duodenal ulcer", "stomach ulcer"],

        # Musculoskeletal
        "arthritis": ["joint inflammation", "arthritic", "osteoarthritis", "rheumatoid arthritis"],
        "fracture": ["broken bone", "bone break", "bone fracture"],
        "osteoporosis": ["bone loss", "decreased bone density", "brittle bones"],
        "back pain": ["lumbar pain", "spinal pain", "dorsalgia", "backache"],

        # General Symptoms
        "symptom": ["sign", "indication", "manifestation", "clinical feature"],
        "pain": ["discomfort", "ache", "soreness", "hurt", "painful", "hurting"],
        "fever": ["pyrexia", "high temperature", "febrile", "elevated temperature"],
        "fatigue": ["tiredness", "exhaustion", "weakness", "lethargy"],
        "nausea": ["sick to stomach", "queasiness", "feeling sick", "nauseated"],
        "dizziness": ["vertigo", "lightheaded", "giddiness", "dizzy"],
        "swelling": ["edema", "inflammation", "swollen", "puffiness"],

        # Medical Procedures
        "surgery": ["procedure", "operation", "surgical intervention", "operative procedure"],
        "check": ["examination", "screening", "inspection", "assessment", "evaluation"],
        "home": ["self-examination", "self-check", "self-screening", "at-home"],
        "diagnosis": ["diagnostic", "identification", "detection", "diagnose"],
        "treatment": ["therapy", "intervention", "management", "care", "therapeutic"],
        "prescription": ["medication", "drug", "medicine", "pharmaceutical"],
        "vaccine": ["vaccination", "immunization", "shot", "inoculation"],
        "test": ["screening", "diagnostic test", "lab test", "examination", "assay"],
        "scan": ["imaging", "radiograph", "X-ray", "CT", "MRI"],
        "biopsy": ["tissue sample", "tissue examination", "histology"],

        # Mental Health
        "depression": ["depressive disorder", "major depression", "clinical depression"],
        "anxiety": ["anxiety disorder", "anxious", "nervousness", "worry"],
        "stress": ["psychological stress", "mental stress", "tension"],

        # Kidney and Urinary
        "kidney": ["renal", "kidney function", "nephrology"],
        "kidney disease": ["renal disease", "nephropathy", "kidney disorder"],
        "urinary": ["urine", "urination", "urologic", "bladder"],

        # Reproductive Health
        "pregnancy": ["pregnant", "gestation", "prenatal", "expecting"],
        "menstruation": ["period", "menstrual cycle", "menses", "monthly cycle"],

        # Skin Conditions
        "rash": ["skin rash", "dermatitis", "skin irritation", "eruption"],
        "eczema": ["atopic dermatitis", "dermatitis", "skin inflammation"],
        "psoriasis": ["psoriatic condition", "skin lesion", "plaque psoriasis"],

        # Blood Disorders
        "anemia": ["low blood count", "iron deficiency", "anemic", "low hemoglobin"],
        "bleeding": ["hemorrhage", "blood loss", "hemorrhaging"],

        # Vision and Hearing
        "vision": ["sight", "visual", "eyesight", "eye health"],
        "blind": ["blindness", "vision loss", "visual impairment"],
        "hearing": ["auditory", "audition", "hearing ability"],
        "deaf": ["deafness", "hearing loss", "hearing impairment"],

        # Nutrition and Lifestyle
        "diet": ["nutrition", "dietary", "eating habits", "nutritional"],
        "exercise": ["physical activity", "workout", "fitness", "activity"],
        "obesity": ["overweight", "excess weight", "obese", "weight problem"],
        "weight loss": ["losing weight", "weight reduction", "slimming"],

        # Emergency and Acute Care
        "emergency": ["urgent", "acute", "critical", "urgent care"],
        "trauma": ["injury", "traumatic injury", "wound", "physical trauma"],
        "poisoning": ["toxic", "toxicity", "intoxication", "poison"],
    }

    @classmethod
    def expand_query(cls, query: str) -> str:
        """
        Expand query with common medical synonyms.

        Args:
            query: Original search query

        Returns:
            Expanded query string with synonyms
        """
        query_lower = query.lower()
        expanded_terms = [query]

        for term, synonyms in cls.MEDICAL_EXPANSIONS.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)

        return " ".join(expanded_terms)

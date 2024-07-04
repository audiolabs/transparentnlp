from dataclasses import dataclass

@dataclass
class BleuMetric:
    reference: str
    hypothesis: str

@dataclass
class RougeMetric:
    reference: str
    hypothesis: str

@dataclass
class ReadabilityMetric:
    hypothesis: str

@dataclass
class LexicalDiversityMetric:
    hypothesis: str
    threshold: float = 0.72  # Default threshold for MTLD

@dataclass
class AvgSentenceLengthMetric:
    hypothesis: str

@dataclass
class BertScoreMetric:
    reference: str
    hypothesis: str

@dataclass
class SemanticSimilarityMetric:
    reference: str
    hypothesis: str
    model: str = "all-MiniLM-L6-v2"

@dataclass
class FaithfulnessMetric:
    query: str
    context: str
    hypothesis: str

@dataclass
class AnswerRelevancyMetric:
    query: str
    context: str
    hypothesis: str

@dataclass
class AnswerSimilarityMetric:
    query: str
    reference: str
    hypothesis: str

@dataclass
class AnswerCorrectnessMetric:
    query: str
    reference: str
    hypothesis: str

@dataclass
class RqugeScoreMetric:
    query: str
    context: str
    hypothesis: str
    model_name: str = "alirezamsh/rquge"

@dataclass
class GevalMetric:
    query: str
    context: str
    hypothesis: str

@dataclass
class SelfCheckCorrectnessMetric:
    query: str
    context: str
    hypothesis: str
    num_samples: int = 5  # Default number of samples for self-check

@dataclass
class ZeroshotPromptsMetric:
    query: str
    context: str
    hypothesis: str

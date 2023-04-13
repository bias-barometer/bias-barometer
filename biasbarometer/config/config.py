from simple_parsing import choice
from dataclasses import dataclass


@dataclass
class BarometerConfig:
    pass


@dataclass
class BiasDirectionConfig(BarometerConfig):
    # Path to target word list used for computing the bias
    target: str
    # Path to word pair list used for estimating the bias direction
    wordpairs: str
    method: str = "direction"
    bias_direction: str = "classification-normal"


@dataclass
class STSBConfig(BarometerConfig):
    # Path to target word list used for computing the bias
    target: str
    # Path to word pair list used for estimating the bias direction
    templates: str
    method: str = "sts-b"


@dataclass
class HyperParameters:
    seed: int = 42


@dataclass
class ModelConfig:
    # Filepath or HuggingFace name of the model
    path: str
    # Device for the model
    device: str = choice(("cpu", "cuda", "mps"), default="cuda")
    # Whether words (e.g., in the wordlists) should be lowercased first
    lowercase: bool = False


@dataclass
class BERTConfig(ModelConfig):
    # Architecture of the model
    architecture: str = "bert"


@dataclass
class WordEmbeddingsConfig(ModelConfig):
    # Architecture of the model
    architecture: str = "word embeddings"
    # Whether the model name is a filepath
    name_is_filepath: bool = False

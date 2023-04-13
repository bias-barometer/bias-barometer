from abc import ABC, abstractmethod

import gensim
import gensim.downloader
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from transformers import logging as tf_logging
import logging

from biasbarometer.models import (
    Embedding,
    SentenceEmbedding,
)

from biasbarometer.config import ModelConfig

# Turn off "Some weights of the model checkpoint at XX were not used when initializing" warning
tf_logging.set_verbosity_error()


class Model(ABC):
    def __init__(
        self,
        vocab=None,
        device="cuda",
        **kwargs,
    ):
        self.name = None
        torch.set_grad_enabled(False)
        self._device = device
        if device == "cpu":
            self.device = torch.device("cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                logging.warning("Device cuda is not available; Falling back to cpu.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "mps":
            if not torch.backends.mps.is_available():
                logging.warning("Device mps is not available; Falling back to cpu.")
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")
        self.representations = None
        self.sentence_embedding = None
        self.input_embedding = None

    @classmethod
    def from_spec(cls, model, architecture, **kwargs):
        if architecture == "bert":
            return BERTModel(model, **kwargs)
        elif architecture == "word embeddings":
            return WordEmbeddingsModel(model, **kwargs)
        else:
            raise ValueError(f"{architecture} is not (yet) supported as architecture.")

    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs):
        architecture = config.architecture
        if architecture == "bert":
            return BERTModel(config.path, **vars(config), **kwargs)
        elif architecture == "word embeddings":
            return WordEmbeddingsModel(config.path, **vars(config), **kwargs)
        else:
            raise ValueError(f"{architecture} is not (yet) supported as architecture.")

    def load(self):
        raise NotImplementedError

    def get_representation(self, representation):
        raise NotImplementedError


class BERTModel(Model):
    def __init__(self, model_fp, **kwargs):
        super().__init__(**kwargs)
        self.architecture = "bert"
        self.load_model(model_fp)
        self.representations = ("embedding", "sentence embedding")

    @staticmethod
    def get_bert(model_fp, tokenizer_fp=None, config_fp=None):
        """Possible options are, for example:
        - pdelobelle/robbert-v2-dutch-base
        - GroNLP/bert-base-dutch-cased
        """
        config_fp = config_fp or model_fp
        config = AutoConfig.from_pretrained(config_fp)
        config.output_scores = True
        tokenizer_fp = tokenizer_fp or model_fp
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_fp)
        bert = AutoModelForMaskedLM.from_pretrained(model_fp, config=config)
        bert.eval()
        return bert, tokenizer

    def load_model(self, model_fp):
        self.model, self.tokenizer = self.get_bert(model_fp)

        # Get model without language modeling head on top
        if hasattr(self.model, "bert"):
            self.model_ = self.model.bert
        elif hasattr(self.model, "distilbert"):
            self.model_ = self.model.distilbert
        else:
            self.model_ = self.model.roberta

        self.model = self.model.to(self.device)

    def get_intput_embedding(self):
        if not self.input_embedding:
            self.input_embedding = Embedding(
                self.model_.get_input_embeddings(),
                self.tokenizer.vocab,
                device=self.device,
            )
        return self.input_embedding

    def get_sentence_embedding(self):
        if not self.sentence_embedding:
            self.sentence_embedding = SentenceEmbedding.from_model(
                self.architecture,
                model=self.model,
                tokenizer=self.tokenizer,
                return_representation="cls",
            )
        #            self.sentence_embedding = MaskedSentenceEmbedding(self.model, self.tokenizer, device=self.device, return_representation="cls")
        return self.sentence_embedding

    def get_representation(self, representation):
        if representation == "embedding":
            return self.get_intput_embedding()
        elif representation == "sentence embedding":
            return self.get_sentence_embedding()
        else:
            raise ValueError(
                f"{representation} is not a supported representation for this model."
            )


class WordEmbeddingsModel(Model):
    def __init__(self, model_fp, name_is_filepath=False, **kwargs):
        super().__init__(**kwargs)
        self.model_ = None
        self.embedding = None
        self.load_model(model_fp, name_is_filepath=name_is_filepath)
        self.representations = "embedding"

    def load_model(self, model_fp, name_is_filepath=False):
        # TODO: maybe there is a more straightforward way to load a static word embedding in gensim?
        if name_is_filepath:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(
                model_fp, binary=True
            )
        else:
            self.model = gensim.downloader.load(model_fp)
        weights = torch.FloatTensor(self.model.vectors)
        self.model_ = torch.nn.Embedding.from_pretrained(weights)
        self.w2i = self.model.key_to_index

    def get_representation(self, representation):
        if representation == "embedding":
            return self.get_embedding()
        else:
            raise ValueError(
                f"{representation} is not a supported representation for this model."
            )

    def get_embedding(self):
        if not self.embedding:
            self.embedding = Embedding(self.model_, self.w2i)
        return self.embedding

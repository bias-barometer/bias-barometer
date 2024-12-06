from abc import ABC, abstractmethod

import gensim
import gensim.downloader
import fasttext
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, GPTNeoXForCausalLM
from transformers import logging as tf_logging
import logging

from biasbarometer.models import (
    WordEmbeddings,
    SentenceEmbeddings,
    CharacterEmbeddings,
)

from biasbarometer.config import ModelConfig

# Turn off "Some weights of the model checkpoint at XX were not used when initializing" warning
tf_logging.set_verbosity_error()

class Model(ABC):
    """
    Abstract class as the basis for classes that wrap embeddings and language models.
    """    
    def __init__(
        self,
        device: str = "cuda",
        **kwargs,
    ):
        """Initializes the model's device (cpu/cuda/mps) and representations (as None).

        Args:
            device (str, optional): String representation of the torch device. Defaults to "cuda".
        """    
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

        # Initialize possible representations as None
        self._sentence_embeddings = None
        self._embeddings = None 
        #self._representations = [self.embeddings, self.sentence_embeddings]

    @property
    def embeddings(self):
        raise ValueError(
                f"(Static) embeddings is not a supported representation for this model."
            )

    @property
    def sentence_embeddings(self):
        raise ValueError(
                f"Sentence embeddings is not a supported representation for this model."
            )

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

class BERTModel(Model):
    """Wrapper class for BERT models from the 🤗 Transformers library"""
    def __init__(self, model_fp: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.architecture = "bert"
        self.load_model(model_fp)

    @staticmethod
    def get_bert(model_fp: str, tokenizer_fp: str = None, config_fp: str = None, device: str = "cpu"):
        """Helper function for loading a 🤗 Transformers BERT model from disk or the Hugging Face hub.

        Examples for model_fp are bert-base-uncased, pdelobelle/robbert-v2-dutch-base, and GroNLP/bert-base-dutch-cased

        Args:
            model_fp (str): File path or name for the transformer model.
            tokenizer_fp (str, optional): File path for the BERT tokenizer. Defaults to None.
            config_fp (str, optional): File path for the (optional) configuration file. Defaults to None.

        Returns:
            BERT transformer model and its tokenizer.
        """
        config_fp = config_fp or model_fp
        config = AutoConfig.from_pretrained(config_fp)
        config.output_scores = True
        tokenizer_fp = tokenizer_fp or model_fp
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_fp)
        bert = AutoModelForMaskedLM.from_pretrained(
            model_fp, 
            config=config,
            low_cpu_mem_usage=True, # https://huggingface.co/docs/transformers/main_classes/model#large-model-loading
            device_map=device,
        )
        bert.eval()
        return bert, tokenizer

    def load_model(self, model_fp: str) -> None:
        """Loads the BERT model and finds the base model without the language modeling head.

        Args:
            model_fp (str): File path or name of model on the hugging face hub.

        Raises:
            NotImplementedError: Raised if the base model without the language modeling head cannot be found.
        """        
        self.model, self.tokenizer = self.get_bert(model_fp, device = self._device)

        # Get model without language modeling head on top
        if hasattr(self.model, "bert"):
            self.model_ = self.model.bert
        elif hasattr(self.model, "distilbert"):
            self.model_ = self.model.distilbert
        elif hasattr(self.model, "roberta"):
            self.model_ = self.model.roberta
        else:
            raise NotImplementedError("Cannot find base model without language modeling head for this model.")

        #self.model = self.model.to(self.device)

    @property
    def sentence_embeddings(self) -> SentenceEmbeddings:
        """Property defining the sentence embeddings for this model.

        Returns:
            SentenceEmbeddings: Sentence embeddings derived from this model.
        """
        if not self._sentence_embeddings:
            self._sentence_embeddings = SentenceEmbeddings.from_model(
                self.architecture,
                model=self.model,
                tokenizer=self.tokenizer,
                return_representation="cls",
            )
        #            self.sentence_embedding = MaskedSentenceEmbedding(self.model, self.tokenizer, device=self.device, return_representation="cls")
        return self._sentence_embeddings

    @property
    def embeddings(self) -> WordEmbeddings:
        """Property defining the static embeddings for this model.

        Returns:
            WordEmbeddings: Static word embeddings derived from this model, which are the input embeddings.
        """
        if not self._embeddings:
            self._embeddings = WordEmbeddings(
                self.model_.get_input_embeddings(),
                self.tokenizer.vocab,
                device=self.device,
            )
        return self._embeddings

class GloVeEmbeddingsModel(Model):
    """Wrapper class for GloVe word embeddings models from the Gensim library."""
    def __init__(self, model_fp: str, **kwargs) -> None:
        """Initializes the GloVe model.

        Args:
            model_fp (str): Filepath of the model or the name used by the gensim downloader.
        """        
        super().__init__(**kwargs)
        self.model_ = None
        self.load_model(model_fp)

    def load_model(self, model_fp: str) -> None:
        """Load model from filepath or using the gensim downloader.

        Args:
            model_fp (str): Filepath or gensim name of the model.
        """        
        try:
            self.model = gensim.downloader.load(model_fp)
        except ValueError:
            self.model = gensim.models.KeyedVectors.load(
                    model_fp
                )
        weights = torch.FloatTensor(self.model.vectors)
        self.model_ = torch.nn.Embedding.from_pretrained(weights)
        self.w2i = self.model.key_to_index

    @property
    def embeddings(self) -> WordEmbeddings:
        """Property defining the static embeddings for this model.

        Returns:
            WordEmbeddings: Static word embeddings derived from this model.
        """        
        if not self._embeddings:
            self._embeddings = WordEmbeddings(self.model_, self.w2i)
        return self._embeddings

class Word2VecEmbeddingsModel(Model):
    """Wrapper class for word2vec word embeddings models from the Gensim library."""
    def __init__(self, model_fp: str, **kwargs) -> None:
        """Initializes the word2vec model.

        Args:
            model_fp (str): Filepath to the model or name used by the gensim downloader.
        """        
        super().__init__(**kwargs)
        self.model_ = None
        self.load_model(model_fp)

    def load_model(self, model_fp: str) -> None:
        """Load model from filepath or using the gensim downloader.

        Args:
            model_fp (str): Filepath or gensim name of the model.
        """
        try:
            self.model = gensim.downloader.load(model_fp)
        except ValueError:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(
                model_fp, binary=True
            )
        weights = torch.FloatTensor(self.model.vectors)
        self.model_ = torch.nn.Embedding.from_pretrained(weights)
        self.w2i = self.model.key_to_index

    @property
    def embeddings(self) -> WordEmbeddings:
        """Property defining the static embeddings for this model.

        Returns:
            WordEmbeddings: Static word embeddings derived from this model.
        """
        if not self._embeddings:
            self._embeddings = WordEmbeddings(self.model_, self.w2i)
        return self._embeddings

# class FastTextEmbeddingsModel(Model):
#     """Wrapper class for FastText embeddings models."""
#     def __init__(self, model_fp: str, **kwargs) -> None:
#         """Initializes the FastText model.

#         Args:
#             model_fp (str): File path to saved the FastText model.
#         """        
#         super().__init__(**kwargs)
#         self.load_model(model_fp)

#     def load_model(self, model_fp: str) -> None:
#         """Load FastText model from filepath.

#         Args:
#             model_fp (str): Filepath of the model.
#         """
#         self.model = fasttext.load_model(model_fp)

#     @property
#     def embeddings(self) -> CharacterEmbeddings:
#         """Property defining the static embeddings for this model.

#         Returns:
#             CharacterEmbeddings: Static word embeddings derived from this model.
#         """
#         if not self._embeddings:
#             self._embeddings = CharacterEmbeddings(self.model)
#         return self._embeddings

class PythiaModel(Model):
    def __init__(self, model_fp, step=3000, **kwargs):
        #self.name = f"pythia_70M_{step}"
        #self.unk = "[UNK]" #TODO
        #self.pad_token = "[PAD]" #TODO

        super().__init__(**kwargs)
        self.step = step
        self.architecture = "pythia"
        self.load_model(model_fp)

    @staticmethod
    def get_pythia(model_fp, step, config_fp=None):
        """
        TODO
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_fp, #"EleutherAI/pythia-70m-deduped",
            revision=f"step{step}",
            cache_dir=f"./{model_fp.split('/')[1]}/step{step}",
            #cache_dir=f"./pythia-70m-deduped/step{step}",
            )
        # TODO
        model = GPTNeoXForCausalLM.from_pretrained(
                    model_fp, #"EleutherAI/pythia-70m-deduped",
                    revision=f"step{step}",
                    cache_dir=f"./{model_fp.split('/')[1]}/step{step}",
                    low_cpu_mem_usage=True,
                    #cache_dir=f"./pythia-70m-deduped/step{step}",
                    )
        model.eval()
        return model, tokenizer

    def load_model(self, model):
        self.model, self.tokenizer = self.get_pythia(model, self.step)
        self.model_ = self.model.gpt_neox
        self.w2i = self.tokenizer.vocab
        self.vocab = self.w2i # Necessary for attribution-bias
        self.model = self.model.to(self.device)

    @property
    def sentence_embeddings(self) -> SentenceEmbeddings:
        """Property defining the sentence embeddings for this model.

        Returns:
            SentenceEmbeddings: Sentence embeddings derived from this model.
        """
        if not self._sentence_embeddings:
            self._sentence_embeddings = SentenceEmbeddings.from_model(
                self.architecture,
                model=self.model,
                tokenizer=self.tokenizer,
                return_representation="cls",
            )
        return self._sentence_embeddings

    @property
    def embeddings(self) -> WordEmbeddings:
        """Property defining the static embeddings for this model.

        Returns:
            WordEmbeddings: Static word embeddings derived from this model, which are the input embeddings.
        """
        if not self._embeddings:
            self._embeddings = WordEmbeddings(
                self.model_.get_input_embeddings(),
                self.tokenizer.vocab,
                device=self.device,
            )
        return self._embeddings

class FastTextEmbeddingsModel(Model):
    """Wrapper class for FastText embeddings models."""
    def __init__(self, model_fp: str, **kwargs) -> None:
        """Initializes the FastText model.

        Args:
            model_fp (str): File path to saved the FastText model.
        """        
        super().__init__(**kwargs)
        self.load_model(model_fp)

    def load_model(self, model_fp: str) -> None:
        """Load FastText model from filepath.

        Args:
            model_fp (str): Filepath of the model.
        """
        self.model = fasttext.load_model(model_fp)

    @property
    def embeddings(self) -> CharacterEmbeddings:
        """Property defining the static embeddings for this model.

        Returns:
            CharacterEmbeddings: Static word embeddings derived from this model.
        """
        if not self._embeddings:
            self._embeddings = CharacterEmbeddings(self.model)
        return self._embeddings
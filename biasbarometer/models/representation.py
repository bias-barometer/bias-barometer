from abc import ABC, abstractmethod
import torch
import numpy as np
from fasttext import _FastText
from transformers import PretrainedModel, PreTrainedTokenizer, BertModel, BertTokenizer

from collections.abs import Iterable

class Representation(ABC):
    """This is an abstract class for a model representation, which is a possible location for measuring the bias using a barometer."""
    def __init__(
        self,
        device: torch.device = None,
        lowercase: bool = False,
        **kwargs,
    ):
        """Initialize the device of the representation and some other properties.

        Args:
            device (torch.device, optional): Torch device used by the model; Uses cpu if None. Defaults to None.
            lowercase (bool, optional): Flag for indicating whether all inputs should be lowercased. Defaults to False.
        """    
        self.device = device or torch.device("cpu")
        self.lowercase = lowercase

    @abstractmethod
    def __call__(self):
        pass

class StaticEmbeddings(Representation):
    """Super class for word embeddings and character embeddings."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class WordEmbeddings(StaticEmbeddings):
    """The WordEmbeddings Representation returns a vector representation for a given word when called."""

    def __init__(self, model: torch.nn.Embedding, vocab: dict, **kwargs):
        """Initializes the WordEmbeddings object.

        Args:
            model (torch.nn.Embedding): The embeddings model used for mapping the token to its vector representation.
            vocab (dict): Dictionary mapping a word to its token index.
        """
        super().__init__(**kwargs)
        self.vocab = vocab
        self.model = model

    def __call__(self, words: Iterable[str] | str):
        """Returns the vector representation for the word(s) if possible.

        Args:
            words (Iterable[str] | str): Word(s) for which to find the vector representation(s).

        Raises:
            ValueError: If the word is not part of the vocabulary.

        Returns:
            np.ndarray: Array representing the word vectors.
        """        
        # If words is a string, make it a list first
        words = [words] if isinstance(words, str) else words

        # We first check if the words are part of the vocabulary
        tokens = []
        for word in words:
            if self.lowercase:
                word = word.lower()
            if word in self.vocab:
                token = self.vocab[word]
            elif "##" + word in self.vocab:
                token = self.vocab["##" + word]
            elif "Ġ" + word in self.vocab:
                token = self.vocab["Ġ" + word]
            else:
                raise ValueError(f"{word} is not part of the vocabulary.")
            tokens.append(token)

        w_tok = torch.LongTensor([tokens]).to(self.device)
        word_vectors = self.model(w_tok).detach().squeeze().cpu().numpy()
        return word_vectors

class CharacterEmbeddings(StaticEmbeddings):
    """The Character Embedding Representation returns a vector representation for a given word when called."""

    def __init__(self, model: _FastText, **kwargs):
        """Initializes the character embeddings representation for a FastText model.

        Args:
            model (_FastText): FastText model.
        """        
        super().__init__(**kwargs)
        self.model = model

    def __call__(self, words: Iterable[str] | str):
        """Returns the vector representation(s) for the word(s) if available.

        Args:
            words (Iterable[str] | str): A word or list of words to encode.

        Raises:
            ValueError: Raised if the word is not part of the model's vocabulary.

        Returns:
            np.ndarray: Array representing the word vectors.
        """        
        # If words is a string, make it a list
        words = [words] if isinstance(words, str) else words

        # We first check if the words are part of the vocabulary
        word_vectors = []
        for word in words:
            if self.lowercase:
                word = word.lower()
            try:
                word_vectors.append(self.model[word])
            except:
                raise ValueError(f"{word} is not part of the vocabulary.")
        return np.stack(word_vectors).squeeze()


class SentenceEmbeddings(Representation):
    """Super class for representing sentence embeddings."""
    def __init__(
        self,
        model: PretrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs,
    ) -> None:
        """Initializes the super class for representing sentence embeddings.

        Args:
            model (PretrainedModel): The transformer model used for finding the sentence embeddings.
            tokenizer (PreTrainedTokenizer): The tokenizer belonging to the model.
        """    
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        assert self.model
        assert self.tokenizer
        self.device = model.device

    @classmethod
    def from_model(cls, model_architecture: str, **kwargs):

        if model_architecture == "bert":
            return MaskedSentenceEmbeddings(**kwargs)
        else:
            raise ValueError(
                f"SentenceEmbedding not implemented for the {model_architecture} model type."
            )

    def to_word_embeddings(self, words: Iterable[str], templates: Iterable[str]):
        """Transforms the sentence embeddings representation to a static word embeddings representation.

        Args:
            words (Iterable[str]): List of words for which you want to find embeddings.
            templates (Iterable[str]): List of templates used for sampling the embeddings.

        Raises:
            NotImplementedError: _description_
        """        
        # TODO: function to transform sentence embeddings to static word embeddings
        raise NotImplementedError

    @abstractmethod
    def __call__(self, sentences: Iterable[str]):
        """Returns the sentence embeddings for the sentences."""
        pass


class MaskedSentenceEmbeddings(SentenceEmbeddings):
    """Sentence embeddings representation for masked language models (e.g., BERT)"""
    def __init__(
        self,
        *,
        model: BertModel,
        tokenizer: BertTokenizer,
        return_representation: str = "cls",
        **kwargs,
    ):
        """Initializes the masked sentence embeddings representation.

        Args:
            model (BertModel): BERT model used for encoding the sentences into vector representations.
            tokenizer (BertTokenizer): BERT tokenizer.
            return_representation (str, optional): How the sentence embedding is found. Defaults to "cls".
        """    
        super().__init__(model, tokenizer, **kwargs)
        self.return_representation = return_representation

    def __call__(self, sentences: Iterable[str]):
        """Creates sentence embeddings using the BERT model for the list of sentences passed to it.
        Either uses the "CLS" token or the pooler output, depending on the initialization of this representation.

        Args:
            sentences (Iterable[str]): List of sentences to create sentence embeddings for using the model.

        Returns:
            torch.Tensor: The sentence embeddings.
        """        
        # Borrowed code from https://github.com/pdrm83/sent2vec/blob/master/sent2vec/vectorizer.py
        # Add [CLS] token to use as proxy for sentence embedding
        cls_tok = self.tokenizer.cls_token + " "
        sep_tok = " " + self.tokenizer.sep_token
        pad_tok = self.tokenizer.pad_token
        tokenized = list(
            map(
                lambda x: self.tokenizer.encode(
                    cls_tok + x + sep_tok, add_special_tokens=True
                ),
                sentences,
            )
        )

        max_len = 0
        for i in tokenized:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        input_ids = torch.tensor(np.array(padded)).type(torch.LongTensor)
        input_ids = input_ids.to(self.device)
        # attention_mask = torch.tensor(np.where(padded != 0, 1, 0)).type(torch.LongTensor)

        with torch.no_grad():
            output = self.model(input_ids, labels=input_ids, output_hidden_states=True)

        # Could also use pooler output: https://github.com/huggingface/transformers/issues/7540
        if self.return_representation == "cls":
            output = output.hidden_states[-1][:, 0, :]
        elif self.return_representation == "pooled":
            output = output.pooler_output[:, :]
        else:
            raise ValueError("return_representation should be either cls or pooled.")

        return output

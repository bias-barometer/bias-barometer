from abc import ABC, abstractmethod
import torch
import numpy as np


class Representation(ABC):
    def __init__(
        self,
        device=None,
        lowercase=False,
        **kwargs,
    ):
        self.device = device or torch.device("cpu")
        self.lowercase = lowercase


class Embedding(Representation):
    """The Embedding Representation returns a vector representation for a given word when called."""

    def __init__(self, model, vocab, **kwargs):
        """
        Args:
            vocab:
                Dictionary mapping a word to its token index.
            model:
                torch.nn.Embedding that maps the token to its vector representation.
        """
        super().__init__(**kwargs)
        self.vocab = vocab
        self.model = model

    def __call__(self, words):
        """Returns the vector representation for the words if available."""
        # If words is a string, make it a list
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


class SentenceEmbedding(ABC):
    def __init__(
        self,
        model,
        tokenizer,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        assert self.model
        assert self.tokenizer
        self.device = model.device

    @classmethod
    def from_model(cls, model_architecture, **kwargs):

        if model_architecture == "bert":
            return MaskedSentenceEmbedding(**kwargs)
        else:
            raise ValueError(
                f"SentenceEmbedding not implemented for the {model_architecture} model type."
            )

    @abstractmethod
    def __call__(self, sentences):
        """Returns the sentence embeddings for the sentences."""
        pass


class MaskedSentenceEmbedding(SentenceEmbedding):
    def __init__(
        self,
        return_representation="cls",
        model=None,
        tokenizer=None,
        **kwargs,
    ):
        super().__init__(model, tokenizer, **kwargs)
        self.return_representation = return_representation

    def __call__(self, sentences):
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

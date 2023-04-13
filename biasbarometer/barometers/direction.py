import numpy as np
import pandas as pd
import sklearn.svm
import logging

from biasbarometer.barometers import AutoBarometer
from biasbarometer.data import WordList
from biasbarometer.config import BiasDirectionConfig
from biasbarometer.models import Embedding


class ClassificationNormalBiasDirection:
    """
    Adapted from (the now deprecated) https://github.com/allenai/allennlp

    Classification normal bias direction. Computes one-dimensional subspace that is the span
    of a specific concept (e.g. gender) as the direction perpendicular to the classification
    boundary of a linear support vector machine fit to classify seed word embedding sets.
    """

    def __init__(self):
        pass

    def __call__(self, seed_embeddings1, seed_embeddings2):

        X = np.vstack([seed_embeddings1, seed_embeddings2])
        Y = np.concatenate(
            [[0] * seed_embeddings1.shape[0], [1] * seed_embeddings2.shape[0]]
        )

        classifier = sklearn.svm.SVC(kernel="linear").fit(X, Y)
        bias_direction = classifier.coef_[0]
        return bias_direction / np.linalg.norm(bias_direction)


def get_relation_vector(pairs, embedding, bias_direction="classification-normal"):
    """Returns relation vector b that can be used to measure the bias of an embedding"""
    pair1_array = embedding(pairs[0])
    pair2_array = embedding(pairs[1])

    if bias_direction == "classification-normal":
        bias_dim = ClassificationNormalBiasDirection()
        b = bias_dim(pair1_array, pair2_array)
    else:
        raise ValueError(f"Unknown bias direction implementation: {bias_direction}")
    return b


def get_bias_score(w, b):
    """Get bias score given word embedding w and bias direction b."""
    return np.dot(w, b)


def correct_sign_df(df, pairs):
    """Corrects the wrong sign of the relation vector"""
    dict_ = {w: "positive" for w in pairs[0]}
    dict_.update({w: "negative" for w in pairs[1]})
    df_ = df[df["category"] != "target"].copy()
    df_.loc[:, ("direction")] = df_["word"].map(dict_)

    if (
        df_[df_["direction"] == "positive"]["score"].sum()
        > df_[df_["direction"] == "negative"]["score"].sum()
    ):
        return df
    else:
        df["score"] = -1 * df["score"]
        return df


class BiasDirection(AutoBarometer):
    @staticmethod
    def get_display_name() -> str:
        return "direction"

    @staticmethod
    def get_config_class():
        return BiasDirectionConfig

    def evaluate(self, embedding: Embedding) -> None:
        targetlist = WordList.from_file(self.config["target"], pairs=False)
        pairlist = WordList.from_file(self.config["wordpairs"], pairs=True)

        logging.info(f"Finding relation vector.")
        b = get_relation_vector(
            pairlist.get_pairlist(),
            embedding,
            **{k: self.config[k] for k in ("bias_direction") if k in self.config},
        )

        logging.info(f"Scoring bias for words in target and pairlist.")
        bias_df_words = []
        for word in targetlist:
            try:
                bias_score = get_bias_score(embedding(word), b)

                bias_df_words.append(
                    {
                        "word": word,
                        "score": bias_score,
                        "category": targetlist.get_category_word(word),
                    }
                )
            except ValueError:
                logging.warning(
                    f"Target word {word} is not part of the embedding vocabulary and is therefore skipped."
                )

        for word in pairlist:
            try:
                bias_score = get_bias_score(embedding(word), b)

                bias_df_words.append(
                    {
                        "word": word,
                        "score": bias_score,
                        "category": pairlist.get_category_word(word),
                    }
                )
            except ValueError:
                logging.warning(
                    f"Word {word} is not part of the embedding vocabulary and is therefore skipped."
                )

        # Bias score per word
        # Convert list of dicts to dataframe
        bias_df_words = pd.DataFrame(bias_df_words).sort_values(by=["score"])
        bias_df_words = correct_sign_df(bias_df_words, pairlist.get_pairlist())

        # Mean absolute bias score for target words
        score = (
            bias_df_words[bias_df_words["category"] == "target"]["score"].abs().mean()
        )
        # bias_df = pd.DataFrame.from_dict({"score": [abs_mean_score]})

        logging.info(f"Found an average bias of {score}.")
        self.results = {
            "score": score,  # Mean absolute score for target words
            "bias_df": bias_df_words.reset_index(drop=True),  # Bias score per word
        }

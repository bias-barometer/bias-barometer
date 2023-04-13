import argparse
import logging

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from biasbarometer.barometers import AutoBarometer
from biasbarometer.data import STSBDataset
from biasbarometer.config import STSBConfig
from biasbarometer.models import SentenceEmbedding


class STS_B(AutoBarometer):
    @staticmethod
    def get_display_name() -> str:
        return "sts-b"

    @staticmethod
    def get_config_class():
        return STSBConfig

    @staticmethod
    def get_similarity(target, group):
        return torch.nn.functional.cosine_similarity(target, group).cpu()

    def evaluate(self, sentence_embedding: SentenceEmbedding) -> None:
        templates = STSBDataset(
            template_fp=self.config["templates"], target_fp=self.config["target"]
        )
        dataloader = torch.utils.data.DataLoader(
            templates, batch_size=32, shuffle=False
        )
        df = []
        for template_ids, target_labels, sentences in tqdm(dataloader, disable=False):
            target = sentence_embedding(sentences[0])
            group1 = sentence_embedding(sentences[1])
            group2 = sentence_embedding(sentences[2])
            sim_target_group1 = self.get_similarity(target, group1)
            sim_target_group2 = self.get_similarity(target, group2)
            df.append(
                pd.DataFrame.from_dict(
                    {
                        "word": target_labels,
                        "template_id": template_ids,
                        "sim_target_group1": sim_target_group1,
                        "sim_target_group2": sim_target_group2,
                        "score": sim_target_group1 - sim_target_group2,
                    }
                )
            )

        df = pd.concat(df)
        bias_df_words = (
            df.groupby("word").mean()[["score"]].reset_index().sort_values(by=["score"])
        )
        score = df["score"].abs().mean()

        logging.info(f"Found an average bias of {score}.")
        self.results = {
            "score": score,
            "bias_df": bias_df_words.reset_index(drop=True),
            "results_raw": df,
        }

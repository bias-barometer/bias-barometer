from abc import ABC, abstractmethod
import torch
from typing import List
import pandas as pd

from biasbarometer.data import WordList, TargetList


class TemplateList:
    def __init__(self, templates: List[str], mask: str = "[MASK]", **kwargs):
        self.mask = mask
        self.templates = templates

    @classmethod
    def from_file(cls, filepath: str, **kwargs):
        with open(filepath, "r") as f:
            templates = f.read().splitlines()
        return TemplateList(templates, **kwargs)

    def __iter__(self):
        for t in self.templates:
            yield t


class TemplateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name: str,
        templates: TemplateList,
        target: TargetList,
        mask="[MASK]",
        **kwargs
    ):
        self.name = name
        self.templates = templates
        self.target = target
        self.mask = mask
        self.df = None
        self._prepare_sentences()
        assert self.df is not None

    @abstractmethod
    def _prepare_sentences(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.df)

    @abstractmethod
    def __getitem__(self, index):
        return self.df.iloc[index]


class STSBDataset(TemplateDataset):
    def __init__(
        self,
        template_fp: str = "data/templates/sts-b.txt",
        target_fp: str = "data/wordlists/occupations.txt",
        mask: str = "[MASK]",
        **kwargs
    ):
        templates = TemplateList.from_file(template_fp, mask=mask)
        target = WordList.from_file(target_fp)
        self.group1 = "man"
        self.group2 = "woman"
        super().__init__("STS-B", templates, target, **kwargs)

    def _prepare_sentences(self):
        df = []
        for template_id, template in enumerate(self.templates):
            for t in self.target:
                df.append(
                    {
                        "target": t,
                        "template_id": template_id,
                        "sentence_target": template.replace(self.mask, t),
                        "sentence_group1": template.replace(self.mask, self.group1),
                        "sentence_group2": template.replace(self.mask, self.group2),
                    }
                )
        self.df = pd.DataFrame(df)

    @abstractmethod
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return (
            row["template_id"],
            row["target"],
            (row["sentence_target"], row["sentence_group1"], row["sentence_group2"]),
        )

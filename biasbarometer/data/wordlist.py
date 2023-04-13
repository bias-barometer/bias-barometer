from abc import ABC, abstractmethod
import pandas as pd


class WordList(ABC):
    def __init__(self):
        self.wordlist = None

    @classmethod
    def from_file(cls, filepath, pairs=False, **kwargs):
        if pairs:
            wordlist = PairList(**kwargs)
        else:
            wordlist = TargetList(**kwargs)
        wordlist.load(filepath)
        assert wordlist.wordlist
        return wordlist

    @abstractmethod
    def load(self, filepath):
        pass

    @abstractmethod
    def get_category_word(self, word):
        pass

    def get_wordlist(self):
        return self.wordlist

    def __iter__(self):
        for w in self.wordlist:
            yield w


class PairList(WordList):
    """WordList consisting of pairs of words."""

    def __init__(self, sep=",", **kwargs):
        super().__init__(**kwargs)
        self.sep_token = sep
        self.pairlist = None

    def load(self, filepath):
        # Assumes the first row to contain column descriptions
        df = pd.read_csv(filepath, sep=self.sep_token)
        self.categories = list(df.columns)
        assert len(self.categories) == 2, "currently only supports two categories"

        # Used for mapping back the word to its category
        self.word2category = {
            v: k for k, l in df.to_dict(orient="list").items() for v in l
        }

        # Load the wordlist (flattened) and pairlist
        self.pairlist = []
        for cat in self.categories:
            wl = df[cat].tolist()
            assert len(wl) > 0, f"{Wordlist} for category {cat} cannot be empty!"
            self.pairlist.append(wl)
        self.wordlist = [w for sublist in self.pairlist for w in sublist]
        # self.wordlist = list(zip(*self.pairlist))

    def get_pairlist(self):
        return self.pairlist

    def get_category_word(self, word):
        return self.word2category[word]


class TargetList(WordList):
    """WordList consisting of a list of target words."""

    def __init__(self, comment_token="#", **kwargs):
        super().__init__(**kwargs)
        self.comment_token = comment_token

    def load(self, filepath):
        with open(filepath, "r") as f:
            self.wordlist = f.read().splitlines()

        # This allows for commenting out certain words from the wordlist
        # E.g., if self.comment_token == '#', words starting with # are skipped
        if self.comment_token:
            wordlist = []
            for w in self.wordlist:
                if w[0] != self.comment_token:
                    wordlist.append(w)
            self.wordlist = wordlist

    def get_category_word(self, word):
        return "target"

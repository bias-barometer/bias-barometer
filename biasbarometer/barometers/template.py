from dataclasses import dataclass

from biasbarometer.barometers import AutoBarometer
from biasbarometer.data import WordList
from biasbarometer.config import BarometerConfig
from biasbarometer.models import Embedding, SentenceEmbedding


@dataclass
class TemplateBarometerConfig(BarometerConfig):
    """Template config file that list the required and optional arguments for operationalizing this barometer."""

    # Path to target word list used for computing the bias
    target: str
    # Path to word pair list used for estimating the bias direction
    wordpairs: str
    # TODO: Name of the method
    method: str = "template"


class TemplateBarometer(AutoBarometer):
    @staticmethod
    def get_display_name() -> str:
        # TODO: change the name
        return "template"

    @staticmethod
    def get_config_class():
        return TemplateBarometerConfig

    def evaluate(self, representation) -> None:
        """Implements the bias evaluation on a representation and writes the results to self.results.
        The specifics for operationalizing this barometer can be found in self.config.
        """
        # TODO: Operationalize the barometer using a wordlist or dataset
        # (Optional) Get target list
        targetlist = WordList.from_file(self.config["target"], pairs=False)
        # (Optional) Get wordpairs list
        pairlist = WordList.from_file(self.config["wordpairs"], pairs=True)

        # TODO: Run evaluation and add results to self.results
        raise NotImplementedError

        self.results = {
            "score": score,  # Score
        }

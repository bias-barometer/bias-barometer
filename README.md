# bias-barometer
> Tools for measuring social biases in language models and word embeddings.

⚠ Please be aware that `bias-barometer` is intended for educational and research purposes only! It is crucial that the validity and reliability of a bias measure is tested before it can be used to make any claims about the bias of an NLP system. In fact, many bias measures have shown poor validity and reliability (in certain use-cases).

## Install bias-barometer
Clone `bias-barometer` and install using `uv` or `pip`. 
Make sure you have installed at least python3.9.

<details>
  <summary>Using uv</summary>

  ```bash
    cd bias-barometer
    uv venv --python 3.10 # or another version
    uv sync
  ```
</details>

<details>
  <summary>Using pip and venv</summary>

  ```bash
    cd bias-barometer
    # Create and load a virtual environment (optional)
    python -m venv .venv # or use e.g. python3.10
    source .venv/bin/activate
    # Install using pip
    pip install .
  ```
</details>

## Quick start
<details>
  <summary>Evaluate GloVe static word embeddings</summary>

```python
from biasbarometer.barometers import AutoBarometer
from biasbarometer.models import GloVeEmbeddingsModel

# Initialize the embedding representation from a GloVe model
embedding = GloVeEmbeddingsModel("glove-twitter-25").embeddings

# Operationalize the Bias Direction using two wordlists
barometer = AutoBarometer.from_spec("direction", wordpairs="../data/wordlists/man_vs_woman.csv", target="../data/wordlists/occupations.txt")

# Run the bias evaluation
barometer.evaluate(embedding)

barometer.results["bias_df"]
```

`glove-twitter-25` is only one of the models made available by Gensim that can be loaded using `GloveEmbeddingsModel`; See for a complete list the [Gensim-data repository](https://github.com/RaRe-Technologies/gensim-data#models).
</details>

<details>
  <summary>Evaluate distilBERT using STS-B</summary>

```python
from biasbarometer.barometers import AutoBarometer
from biasbarometer.models import BERTModel

# Initialize the sentence embedding representation from a GloVe model
sentence_embeddings = BERTModel("distilbert-base-uncased").sentence_embeddings

# Operationalize the STS-B bias measure using occupation target list and the template list
barometer = AutoBarometer.from_spec("sts-b", target="../data/wordlists/occupations.txt", templates="../data/templates/sts-b.txt")

# Run the bias evaluation
barometer.evaluate(sentence_embeddings)

barometer.results["bias_df"]
```
</details>

Check `notebooks/examples.ipynb` for more.

## Overview
At the center of `bias-barometer` are the `barometers`, which are implementations of bias measures. Each `barometer` is designed for a model `representation`: For example, the **Bias Direction** (e.g., Bolukbasi et al., 2016) measures the bias in *word embeddings*, which can be obtained from e.g. word2vec or the input embeddings of a BERT model.

Each `barometer` requires some dataset or wordlist to operationalize the bias: For example, to measure gender bias for occupations, the **Bias Direction** needs a set of masculine vs. feminine words (*wordpairs*) and a list of occupation terms (*target*) in the language of interest. See [Orgad & Belinkov (2022)](https://aclanthology.org/2022.gebnlp-1.17/) for why it is a good idea to separate the dataset from the metric/task.

### Terminology
- A `barometer` measures some bias in a certain representation; generally it needs a dataset or wordlist to operationalize this bias.
- A `model` is a 'wrapper' for either a static embedding model (e.g., word2vec provided by gensim) or a contextual language model (e.g., BERT provided by 🤗transformers).
- A `representation` is part of a model where we can measure the bias (e.g., word embeddings or sentence embeddings).

## Implement your own barometer
**TODO**

## Possible roadmap
- [ ] Add documentation on how to use and implement new barometers.
- [ ] Incorporate [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluating models on bias benchmarks.
- [ ] Implement more `barometers` for different model types as well as interventions (e.g. "model debiasing" or doing an ablation study with wordlists).
- [ ] Implement `barometers` for textual data.
- [ ] Help researchers to assess the reliability and validity of different bias measures.
- [ ] Accomodate the analysis of model suites (e.g., to compare models across different sizes) and checkpoints (e.g., to study the training dynamics).

## Similar projects
- [AllenNLP](https://github.com/allenai/allennlp) (now deprecated) implements various fairness related metrics, including the [bias direction](https://docs.allennlp.org/main/api/fairness/bias_direction/), and has been an important inspiration for this code; However, this library has been archived.
- [WEFE](https://github.com/dccuchile/wefe): The Word Embedding Fairness Evaluation Framework aspires to provide a unified framework and focuses on measuring and mitigating bias in word embedding models.
- [🤗 Evaluate](https://github.com/huggingface/evaluate) implements some bias metrics.
- [LM-Eval-Harness](https://github.com/EleutherAI/lm-evaluation-harness) includes some bias benchmarks for autoregressive language models.
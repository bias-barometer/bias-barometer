# bias-barometer
> Tools for measuring social biases in language models and word embeddings.

⚠ Please be aware that `bias-barometer` is intended for educational and research purposes only! It is crucial that the validity and reliability of a bias measure is tested before it can be used to make any claims about the bias of an NLP system. In fact, many bias measures have shown poor validity and reliability (in certain use-cases).

The current version of `bias-barometer` is in an early stage of development. But the eventual goal of `bias-barometer` is to provide students and researchers with a unified toolbox for developing and evaluating bias measures for natural language processing (NLP).

## Install bias-barometer
Clone `bias-barometer` and install using `poetry` or `pip`. 
Make sure you have installed at least python3.9.

<details>
  <summary>Using poetry</summary>
  ```
    cd bias-barometer
    poetry install
  ```
</details>

<details>
  <summary>Using pip and venv</summary>
  ```
    cd bias-barometer
    # Create a virtual environment
    python3.9 -m venv .venv
    # Load the environment
    source .venv/bin/activate
    # Install using pip
    pip install .
  ```
</details>

## Quick start
Check `notebooks/examples.ipynb`.

## Overview
At the center of `bias-barometer` are the `barometers`, which are implementations of bias measures. Each `barometer` is designed for a model `representation`: For example, the **Bias Direction** (e.g., Bolukbasi et al., 2016) measures the bias in a *word embedding*, which can be obtained from e.g. word2vec or the input embeddings of a BERT model. 

Each `barometer` requires some dataset or wordlist to operationalize the bias: For example, to measure gender bias for occupations, the **Bias Direction** needs a set of masculine vs. feminine words (*wordpairs*) and a list of occupation terms (*target*) in the language of interest. See [Orgad & Belinkov (2022)](https://aclanthology.org/2022.gebnlp-1.17/) for why it is a good idea to separate the dataset from the metric/task.

### Terminology
- A `barometer` measures some bias in a certain representation; generally it needs a dataset or wordlist to operationalize this bias.
- A `model` is a 'wrapper' for either a word embedding (e.g., word2vec provided by gensim) or a language model (e.g., BERT provided by transformers).
- A `representation` is part of a model where we can measure the bias (e.g., word embeddings or sentence embeddings).

## Implement your own barometer
**TODO**

## Roadmap
- [ ] Add documentation on how to use and implement new barometers.
- [ ] Implement more `barometers` for different model types as well as interventions (e.g. "model debiasing" or doing an ablation study with wordlists).
- [ ] Help researchers to assess the reliability and validity of different bias measures.
- [ ] Accomodate the analysis of model suites (e.g., to compare models across different sizes) and checkpoints (e.g., to study the training dynamics).

## Similar projects
- [AllenNLP](https://github.com/allenai/allennlp) had implemented various fairness related metrics, including the [bias direction](https://docs.allennlp.org/main/api/fairness/bias_direction/), and has been an important inspiration to this code; However, this library has been archived.
- [WEFE](https://github.com/dccuchile/wefe): The Word Embedding Fairness Evaluation Framework also aspires to provide a unified framework and focuses on measuring and mitigating bias in word embedding models.
- [🤗 Evaluate](https://github.com/huggingface/evaluate) also implements some bias metrics.
- [LM-Eval-Harness](https://github.com/EleutherAI/lm-evaluation-harness) also implemented some bias metrics; Leverages prompt-source for prompt-based (bias) evaluations.
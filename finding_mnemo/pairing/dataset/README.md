# Pairing Dataset

This module establishes the datasets for the pairing model to train on.

This dataset is formed of pairs of words:
- *Good pairs*: pairs of words that sound similar. 
- *Bad pairs*: pairs of words that sound different. 

> How different pairs sound is regulated by a parameter called `margin` which states how worse (in terms of distance) should the bad pair be compared to its good pair alternative.

Pairs can be English-English, Mandarin-Mandarin and Mandarin-English.

## Processing

To create such a dataset, we are starting with corpora of common words in both languages.

### Corpora

|  English corpus | Mandarin corpus  |
|---|---|
|  This corpus lists common words in the English language and their frequency. It comes from the [English Word Frequency](https://www.kaggle.com/datasets/rtatman/english-word-frequency) dataset. |  We use HSK words for the mandarin corpus from 1 to 6. |

### Phonemes

We then proceed to transform each word into an phonetic alphabet.

This requires two steps:
1. Translation to the International Phonetic Alphabet (IPA)
2. Transformation to a custom simplified alphabet

> This custom alphabet is inspired by the 15 most stable lexemes (Dolgopolsky, 1986). Using features from each IPA phoneme, we cluster them and use each cluster to create a new letter. That way we reduce the alphabet's cardinality.

### Pair making

Creating pairs is delicate since it will serve as ground truth for our model training.

We are using the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) distance as the heuristic we aim to replicate in our weakly-supervised setting.

Pairs are created 2 by 2. One good pair and one bad pair for the same *anchor* word (that is our target word or query word).

A bad pair must be worst in distance by a specified `margin`, inspired by the [triplet loss](https://en.wikipedia.org/wiki/Triplet_loss) and [contrastive loss](https://lilianweng.github.io/posts/2021-05-31-contrastive/) we use during learning.

## Loading

Loader are then made available to produce samples for a PyTorch training. 

Two loader are currently available: 
- Triplet-Loss loader: `finding_mnemo\pairing\dataset\phonetic_triplet_dataset.py`
- Contrastive-Loss loader: `finding_mnemo\pairing\dataset\phonetic_pair_dataset.py`

## User Guide

The dataset can be generated using the `finding_mnemo\pairing\dataset\processing\main.py` script.

Use example: 
```bash
python main.py --margin 0.2
```

It will perform all processing steps and generate pairs into a `csv` file. 

This process takes time, and can be stopped and restarted using the same file if needed.
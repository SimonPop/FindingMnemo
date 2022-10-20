import numpy as np

ipa_chars = [
    "s",
    "l",
    "ɤ",
    "d",
    "t",
    "ʃ",
    "ŋ",
    "z",
    "ʐ",
    "i",
    "n",
    "ɛ",
    "m",
    "r",
    "k",
    "ɻ",
    "θ",
    "ʈ",
    "ɨ",
    "ʊ",
    "e",
    "ɑ",
    "v",
    "a",
    "ɡ",
    "h",
    "ɥ",
    "p",
    "œ",
    "ɔ",
    "ʃ",
    "t",
    "æ",
    "d",
    "ʒ",
    "ɯ",
    "ɪ",
    "w",
    "o",
    "u",
    "ə",
    "j",
    "y",
    "b",
    "ɕ",
    "ɹ",
    "x",
    "ʒ",
    "ð",
    "ʂ",
    "f",
]

non_ipas = ["ʰ",
    "P",
    "I",
    "ʧ",
    "A"]

import panphon
import json
from sklearn.cluster import KMeans

ft = panphon.FeatureTable()
X = np.array([ft.word_to_vector_list(ipa_char, numeric=True)[0] for ipa_char in ipa_chars])
n_clusters = 10
model = KMeans(n_clusters=n_clusters)
classes = model.fit_predict(X)
output = dict(sorted(list(zip(ipa_chars, classes)), key=lambda tup: tup[1]))
for i, non_ipa in enumerate(non_ipas):
    output[non_ipa] = i + n_clusters
for k, v in output.items():
    output[k] = int(v)
json_output = json.dumps(output, indent=4)
with open("phonemes_{}.json".format(n_clusters), "w", encoding='utf-8') as outfile:
    outfile.write(json_output)
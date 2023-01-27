import optuna
import pandas as pd
import numpy as np
import panphon.distance
from pathlib import Path

"""
This script is designed to find the best sub-distance combiations in order to best characterize english <> chinese pronunciation difference.
"""

# print(balanced_df.sample(30))

# def objective(trial):
#     a = trial.suggest_float('a', 0, 10)
#     b = trial.suggest_float('b', 0, 10)
#     c = trial.suggest_float('c', 0, 10)

#     def distance_func(zh_w, en_w):
#         dolgo = dst.dolgo_prime_distance_div_maxlen(zh_w, en_w)
#         leven = dst.fast_levenshtein_distance_div_maxlen(zh_w, en_w)
#         c_hamming = dst.hamming_feature_edit_distance_div_maxlen(zh_w, en_w)
#         return (a*dolgo + b*leven + c*c_hamming) / (a+b+c)

#     distances = np.array([
#         distance_func(zh_w, en_w) for zh_w, en_w in zip(zh, en)
#     ])
#     return np.mean(abs(validity - distances))

# study = optuna.create_study()
# study.optimize(objective, n_trials=100)

# print(study.best_params)
# import numpy as np
# from ipapy import UNICODE_TO_IPA

# from weighted_levenshtein import lev, osa, dam_lev
# substitute_costs = np.ones((128, 128), dtype=np.float64)  # make a 2D array of 1's
# substitute_costs[ord('H'), ord('B')] = 1.25  # make substituting 'H' for 'B' cost 1.25
# print(substitute_costs)
# print(lev('HANANA', 'BANANA', substitute_costs=substitute_costs))  # prints '1.25'
# print(len({w: i for i, w in enumerate(UNICODE_TO_IPA.keys())}))


# TODO: Idea -> optimize mapping IPA to letter

from dragonmapper import transcriptions
import numpy as np

pair_df = pd.read_csv(Path(__file__).parent / "pairs_v2.csv")
dst = panphon.distance.Distance()

invalid_df = pair_df[pair_df["valid"] == 0]
valid_df = pair_df[pair_df["valid"] == 1]
balanced_df = pd.concat([valid_df, invalid_df.sample(len(valid_df))])

zh = balanced_df["chinese_ipa"].to_list()
en = balanced_df["english_ipa"].to_list()
validity = balanced_df["valid"].to_numpy()


def get_ipa_chars():
    # TODO: create a function that converts an IPA to an arbitrary alphabet so that we can weight distances.
    eng_ipa_chars = [
        "æ",
        "ʧ",
        "r",
        "p",
        "b",
        "m",
        "t",
        "d",
        "n",
        "k",
        "ɡ",
        "ŋ",
        "f",
        "v",
        "s",
        "z",
        "θ",
        "ð",
        "ʃ",
        "ʒ",
        "tʃ",
        "dʒ",
        "l",
        "ɹ",
        "j",
        "w",
        "h",
    ]
    zh_ipa_chars = [x for x in transcriptions._IPA_CHARACTERS]
    ipa_chars = eng_ipa_chars + zh_ipa_chars
    return ipa_chars

ipa_chars = get_ipa_chars()

def objective(trial: optuna.Trial):
    n_cat = len(set(ipa_chars))
    ipa2cat = {
        x: trial.suggest_categorical("category_{}".format(x), [i for i in range(n_cat)])
        for x in ipa_chars
    }

    def distance_func(zh_w, en_w):
        zh_w = "".join([chr(ipa2cat[x]) for x in zh_w])
        en_w = "".join([chr(ipa2cat[x]) for x in en_w])
        leven = dst.fast_levenshtein_distance_div_maxlen(zh_w, en_w)
        return leven

    distances = np.array([distance_func(zh_w, en_w) for zh_w, en_w in zip(zh, en)])
    return np.mean(abs(validity - distances))


study = optuna.create_study()
study.optimize(objective, n_trials=100)

print()
print(study.best_value)
print(study.best_params)

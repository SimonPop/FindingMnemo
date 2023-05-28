from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class PhoneticTripletDataset(Dataset):
    def __init__(self, best_pairs_path: str, worst_pairs_path: str):
        self.best_pairs = pd.read_csv(best_pairs_path).dropna(
            subset=["ipa_a", "ipa_b", "distance"]
        )
        self.worst_pairs = pd.read_csv(worst_pairs_path).dropna(
            subset=["ipa_a", "ipa_b", "distance"]
        )

        self.filter_same_pairs()

        self.longest_word = max(
            [
                self.best_pairs["ipa_a"].str.len().max(),
                self.best_pairs["ipa_b"].str.len().max(),
                self.worst_pairs["ipa_a"].str.len().max(),
                self.worst_pairs["ipa_b"].str.len().max(),
            ]
        )

    def filter_same_pairs(self):
        self.best_pairs["same"] = self.best_pairs["word_a"] == self.best_pairs["word_b"]
        to_drop_indexes = self.best_pairs[self.best_pairs["same"]].index

        self.best_pairs = self.best_pairs.drop(to_drop_indexes)
        self.worst_pairs = self.worst_pairs.drop(to_drop_indexes)

    def __getitem__(self, index):
        best_pair = self.best_pairs.iloc[index]
        worst_pair = self.worst_pairs.iloc[index]
        return {
            "anchor_phonetic": best_pair["ipa_a"],
            "similar_phonetic": best_pair["ipa_b"],
            "similar_distance": best_pair["distance"],
            "distant_phonetic": worst_pair["ipa_b"],
            "distant_distance": worst_pair["distance"],
        }

    def __len__(self):
        if len(self.best_pairs) != len(self.worst_pairs):
            raise ValueError(
                "Best & Worst pairs have different lengths. Cannot apply a triplet loss: verify alignment."
            )
        return len(self.best_pairs)

    def filter_data(self):
        # Keep only the best data (threhsold on best_pairs?)
        pass

    def padding(self):
        pass

from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path


class PhoneticTripletDataset(Dataset):
    def __init__(self, best_pairs_path: str, worst_pairs_path: str):
        self.path = Path(__file__).parent
        self.best_pairs = pd.read_csv(best_pairs_path).dropna(
            subset=["chinese_ipa", "english_ipa"]
        )
        self.worst_pairs = pd.read_csv(worst_pairs_path).dropna(
            subset=["chinese_ipa", "english_ipa"]
        )
        self.longest_word = max(
            [
                self.best_pairs["chinese_ipa"].str.len().max(),
                self.best_pairs["english_ipa"].str.len().max(),
                self.worst_pairs["english_ipa"].str.len().max(),
                self.worst_pairs["chinese_ipa"].str.len().max(),
            ]
        )

    def __getitem__(self, index):
        best_pair = self.best_pairs.iloc[index]
        worst_pair = self.worst_pairs.iloc[index]
        return {
            "anchor_phonetic": best_pair["chinese_ipa"],
            "similar_phonetic": best_pair["english_ipa"],
            "distant_phonetic": worst_pair["english_ipa"],
        }

    def __len__(self):
        if  len(self.best_pairs) != len(self.worst_pairs):
            raise ValueError('Best & Worst pairs have different lengths. Cannot apply a triplet loss: verify alignment.')
        return len(self.best_pairs)

    def filter_data(self):
        # Keep only the best data (threhsold on best_pairs?)
        pass

    def padding(self):
        pass

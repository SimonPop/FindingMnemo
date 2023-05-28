from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class PhoneticPairDataset(Dataset):
    def __init__(self, best_pairs_path: str, worst_pairs_path: str):
        self.path = Path(__file__).parent
        self.best_pairs = pd.read_csv(best_pairs_path).dropna(
            subset=["chinese_ipa", "english_ipa"]
        )
        self.worst_pairs = pd.read_csv(worst_pairs_path).dropna(
            subset=["chinese_ipa", "english_ipa"]
        )

    def __getitem__(self, index):
        is_negative = index % 2
        if is_negative == 0:
            pair = self.best_pairs.iloc[index // 2]
        else:
            pair = self.worst_pairs.iloc[index // 2]
        return {
            "chinese_phonetic": pair["chinese_ipa"],
            "english_phonetic": pair["english_ipa"],
            "distance": pair["distance"],
            "is_negative": is_negative,
        }

    def __len__(self):
        return len(self.best_pairs) + len(self.worst_pairs)

    def filter_data(self):
        # Keep only the best data (threhsold on best_pairs?)
        pass

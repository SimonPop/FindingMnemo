from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

class PhoneticPairDataset(Dataset):
    def __init__(self):
        self.path = Path(__file__).parent
        self.best_pairs = pd.read_csv(self.path / '../../data/best_pairs.csv')
        self.worst_pairs = pd.read_csv(self.path / '../../data/worst_pairs.csv')

    def __getitem__(self, index):
        best_pair = self.best_pairs.iloc[index]
        worst_pair = self.worst_pairs.iloc[index]
        return {
            'chinese_phonetic': best_pair['chinese_ipa'],
            'best_english_phonetic': best_pair['english_ipa'],
            'worst_english_phonetic': worst_pair['english_ipa'],
            'best_distance': best_pair['distance'],
            'worst_distance': worst_pair['distance']
        }

    def filter_data(self):
        # Keep only the best data (threhsold on best_pairs?)
        pass
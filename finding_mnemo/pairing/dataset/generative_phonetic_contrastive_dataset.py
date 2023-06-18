import pandas as pd
from torch.utils.data import Dataset
from finding_mnemo.pairing.dataset.generation.pair_generator import PairGenerator


class GenerativePhoneticContrastiveDataset(Dataset):
    def __init__(self, english_data_path: str, mandarin_data_path: str, size: str):
        self.size = size

        columns = ["ipa", "valid_ipa"]
        english_data = pd.read_csv(english_data_path, usecols=columns).dropna()
        mandarin_data = pd.read_csv(mandarin_data_path, usecols=columns).dropna()

        dataset = pd.concat((english_data, mandarin_data))
        self.dataset = dataset[dataset["valid_ipa"] == True]
        self.dataset = self.dataset[self.dataset["ipa"].str.len() > 2]
        # Shuffling:
        self.dataset = self.dataset.sample(frac=1)

        self.longest_word = self.dataset["ipa"].str.len().max()

        self.pair_generator = PairGenerator()

    def __getitem__(self, index):

        target_row = self.dataset.iloc[index % len(self.dataset)]
        target_word = target_row["ipa"]

        generation = self.pair_generator.generate_tiplet_pair(target_word)

        return {
            "phonetic_a": target_word,
            "phonetic_b": generation["positive_word"] if index % 2 == 0 else generation["negative_word"],
            "distance": generation["positive_distance"] if index % 2 ==0 else generation["negative_distance"],
            "is_negative": index % 2 == 1,
        }

    def __len__(self):
        return self.size
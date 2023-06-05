import pandas as pd
from torch.utils.data import Dataset
from finding_mnemo.pairing.dataset.generation.pair_generator import PairGenerator

class GenerativePhoneticTripletDataset(Dataset):
    def __init__(self, english_data_path: str, mandarin_data_path: str, size: str):
        self.size = size

        columns = ["ipa", "valid_ipa"]
        english_data = pd.read_csv(english_data_path, usecols=columns).dropna()
        mandarin_data = pd.read_csv(mandarin_data_path, usecols=columns).dropna()

        dataset = pd.concat((english_data, mandarin_data))
        self.dataset = dataset[dataset["valid_ipa"]==True]
        self.dataset = self.dataset[self.dataset["ipa"].str.len() > 2]
        # Shuffling:
        self.dataset = self.dataset.sample(frac=1)

        self.longest_word = self.dataset["ipa"].str.len().max()

        self.pair_generator = PairGenerator()

    def __getitem__(self, index):
        
        target_row = self.dataset.iloc[index % len(self.dataset)]
        target_word = target_row['ipa']

        generation = self.pair_generator.generate_pair(target_word)

        return {
            "anchor_phonetic": target_word,
            "similar_phonetic": generation["positive_word"],
            "similar_distance": generation["positive_distance"],
            "distant_phonetic": generation["negative_word"],
            "distant_distance": generation["negative_distance"],
        }

    def __len__(self):
        return self.size

if __name__ == "__main__":
    from pathlib import Path
    dataset = GenerativePhoneticTripletDataset(
        Path(__file__).parent / "data/english.csv",
        Path(__file__).parent / "data/chinese.csv",
        10
        )
    print(dataset[0])
    print(dataset[1])
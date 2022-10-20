import panphon.distance
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
from pathlib import Path


class PairingMaker:
    def __init__(self):
        self.path = Path(__file__).parent
        self.dst = panphon.distance.Distance()
        self.columns = [
            "chinese_hanzi",
            "chinese_pinyin",
            "chinese_phonetic",
            "english_word",
            "english_phonetic",
            "distance",
        ]
        self.best_pairs_path = self.path / "best_pairs.csv"
        self.worst_pairs_path = self.path / "worst_pairs.csv"

    def custom_distance(self, chinese_row, english_row):
        # FIXME: Use a better custom distance using distance between each type of IPA letter directly in levenshtein distance.
        # dolgo_dst = self.dst.fast_levenshtein_distance_div_maxlen(
        #     chinese_row["dolgo"], english_row["dolgo"]
        # )
        # ipa_dst = self.dst.fast_levenshtein_distance_div_maxlen(
        #     chinese_row["ipa"], english_row["ipa"]
        # )
        custom_dst_5 = self.dst.fast_levenshtein_distance_div_maxlen(
            chinese_row["custom_alphabet_5"], english_row["custom_alphabet_5"]
        )
        custom_dst_10 = self.dst.fast_levenshtein_distance_div_maxlen(
            chinese_row["custom_alphabet_10"], english_row["custom_alphabet_10"]
        )
        custom_dst_15 = self.dst.fast_levenshtein_distance_div_maxlen(
            chinese_row["custom_alphabet_15"], english_row["custom_alphabet_15"]
        )
        custom_dst_20 = self.dst.fast_levenshtein_distance_div_maxlen(
            chinese_row["custom_alphabet_20"], english_row["custom_alphabet_20"]
        )
        return (custom_dst_5 + custom_dst_10 + custom_dst_15 + custom_dst_20) / 4

    def make_pairs(self, saving_frequency: int = 10):
        best_pairs, worst_pairs = self.load_pairs()

        english_df = self.load_english_data()
        chinese_df = self.load_chinese_data(best_pairs)

        for i, (_, chinese_row) in enumerate(chinese_df.iterrows()):
            best_pair, worst_pair = self.find_pair_from_chinese(chinese_row, english_df)

            best_pairs = pd.concat((best_pairs, pd.DataFrame([best_pair])))
            worst_pairs = pd.concat((worst_pairs, pd.DataFrame([worst_pair])))

            if i % saving_frequency == 0:
                self.save_data(best_pairs, worst_pairs)

    def find_pair_from_chinese(self, chinese_row, english_df: pd.DataFrame):
        desc = "Finding closest word for {}"
        distances = [
            self.custom_distance(chinese_row, english_row)
            for _, english_row in tqdm(
                english_df.iterrows(), desc=desc.format(chinese_row["pinyin"])
            )
        ]
        best_row = english_df.iloc[np.argmin(distances)]
        worst_row = english_df.iloc[np.argmax(distances)]

        best_pair = {
            "chinese_hanzi": chinese_row["hanzi"],
            "chinese_pinyin": chinese_row["pinyin"],
            "chinese_ipa": chinese_row["ipa"],
            "english_word": best_row["word"],
            "english_ipa": best_row["ipa"],
            "distance": min(distances),
        }

        worst_pair = {
            "chinese_hanzi": chinese_row["hanzi"],
            "chinese_pinyin": chinese_row["pinyin"],
            "chinese_ipa": chinese_row["ipa"],
            "english_word": worst_row["word"],
            "english_ipa": worst_row["ipa"],
            "distance": max(distances),
        }

        return best_pair, worst_pair

    def save_data(self, best_pairs: List[dict], worst_pairs: List[dict]) -> None:
        pd.DataFrame(best_pairs).to_csv(self.best_pairs_path, index=False)
        pd.DataFrame(worst_pairs).to_csv(self.worst_pairs_path, index=False)

    def load_pairs(self) -> List[pd.DataFrame]:
        print("Loading chinese corpus.")
        try:
            best_pairs = pd.read_csv(self.best_pairs_path)
            worst_pairs = pd.read_csv(self.worst_pairs_path)
        except:
            col_names = [
                "chinese_hanzi",
                "chinese_pinyin",
                "chinese_phonetic",
                "english_word",
                "english_phonetic",
                "distance",
            ]
            best_pairs = pd.DataFrame(columns=col_names)
            worst_pairs = pd.DataFrame(columns=col_names)
        return best_pairs, worst_pairs

    def load_english_data(self):
        print("Loading english corpus.")
        english_corpus = pd.read_csv(self.path / "english.csv")
        return english_corpus[english_corpus["valid_ipa"] == True]

    def load_chinese_data(self, pairs):
        print("Loading chinese corpus.")
        hsk_df = pd.read_csv(self.path / "chinese.csv")
        hsk_df = hsk_df[hsk_df["valid_ipa"] == True]
        if not pairs is None:
            hsk_df = hsk_df[~hsk_df["hanzi"].isin(pairs["chinese_hanzi"])]
        return hsk_df


if __name__ == "__main__":
    PairingMaker().make_pairs()

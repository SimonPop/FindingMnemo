import panphon.distance
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
from pathlib import Path
from src.pairing.dataset.pairing import Pair

class PairingMaker:
    """
    PairingMaker can be used to create pairs of chinese <> english terms that either sounds very similar (best pairs) or does not (worst pairs).
    """
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

    def custom_distance(self, chinese_row, english_row) -> float:
        """Computes a custom phonetic distance between an english and chinese term.

        Args:
            chinese_row (_type_): Row containing chinese phonetic information.
            english_row (_type_): Row containing english phonetic information.

        Returns:
            float: Distance between english and chinese pronunciation.
        """
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

    def make_pairs(self, saving_frequency: int = 10) -> None:
        """Creates new pairs (best and worst) and saves them.

        Args:
            saving_frequency (int, optional): Frequency for saving newly created pairs (every x new pairs, stores them). Defaults to 10.
        """
        best_pairs, worst_pairs = self.load_pairs()

        english_df = self.load_english_data()
        chinese_df = self.load_chinese_data(best_pairs)

        for i, (_, chinese_row) in enumerate(chinese_df.iterrows()):
            best_pair, worst_pair = self.find_pair_from_chinese(chinese_row, english_df)

            best_pairs = pd.concat((best_pairs, pd.DataFrame([best_pair.dict()])))
            worst_pairs = pd.concat((worst_pairs, pd.DataFrame([worst_pair.dict()])))

            if i % saving_frequency == 0:
                self.save_data(best_pairs, worst_pairs)

    def find_pair_from_chinese(self, chinese_row, english_df: pd.DataFrame) -> List[Pair]:
        """Find the closest and farthest sounding english word from the given chinese word. 

        Args:
            chinese_row (_type_): Dataframe row containing chinese phonetic information.
            english_df (pd.DataFrame): English dataframe containing the whole of available english vocabulary.

        Returns:
            List[Pair]: Best and Worst pair found for the given chinese term.
        """
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

        return Pair(**best_pair), Pair(**worst_pair)

    def save_data(self, best_pairs: List[dict], worst_pairs: List[dict]) -> None:
        """Saves best and worst pairs in a csv file.

        TODO: Use SQL instead.

        Args:
            best_pairs (List[dict]): Best pairs dataframe.
            worst_pairs (List[dict]): Worst pairs dataframe.
        """
        pd.DataFrame(best_pairs).to_csv(self.best_pairs_path, index=False)
        pd.DataFrame(worst_pairs).to_csv(self.worst_pairs_path, index=False)

    def load_pairs(self) -> List[pd.DataFrame]:
        """Load, if existing, already computed pairs of english <> chinese terms.

        Returns:
            List[pd.DataFrame]: Best and Worst pairs dataframes.
        """
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

    def load_english_data(self) -> pd.DataFrame:
        """Loads the chinese vocabulary dataframe.

        Returns:
            pd.DataFrame: English vocabulary dataframe.
        """
        print("Loading english corpus.")
        english_corpus = pd.read_csv(self.path / "english.csv")
        return english_corpus[english_corpus["valid_ipa"] == True]

    def load_chinese_data(self, pair_df: pd.DataFrame) -> pd.DataFrame:
        """Loads the chinese vocabulary dataframe.

        Args:
            pairs (pd.DataFrame): Existing pair dataframe to filter vocabulary with (in order not to create duplicates).

        Returns:
            pd.DataFrame: Chinese vocabulary dataframe.
        """
        print("Loading chinese corpus.")
        hsk_df = pd.read_csv(self.path / "chinese.csv")
        hsk_df = hsk_df[hsk_df["valid_ipa"] == True]
        if not pair_df is None:
            hsk_df = hsk_df[~hsk_df["hanzi"].isin(pair_df["chinese_hanzi"])]
        return hsk_df


if __name__ == "__main__":
    PairingMaker().make_pairs()

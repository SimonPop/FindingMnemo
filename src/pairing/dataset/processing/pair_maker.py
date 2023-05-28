import panphon.distance
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
from pathlib import Path
from src.pairing.dataset.processing.pair import Pair, InterlanguagePair

class PairMaker:
    """
    PairingMaker can be used to create pairs of chinese <> english terms that either sounds very similar (best pairs) or does not (worst pairs).
    """
    def __init__(self, margin: float = 0.2):
        """

        Args:
            margin (float): Margin that best & worst pairs should have between themselves (so that triplet loss makes sense in terms of distance)
            e.g. margin=3, a --> b is 1, a --> c should be >4 since the triplet loss will ask for "dist(a,b) - distance(a,c) + m" to be <0 
        """
        self.dst = panphon.distance.Distance()
        self.columns = [
            "chinese_hanzi",
            "chinese_pinyin",
            "chinese_phonetic",
            "english_word",
            "english_phonetic",
            "distance",
        ]
        self.best_pairs_suffix = "best_pairs.csv"
        self.worst_pairs_suffix = "worst_pairs.csv"
        self.margin = margin

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

    def make_pairs(self, english_dataset_path: Path, mandarin_dataset_path: Path, output_directory: Path, saving_frequency: int = 10) -> None:
        """Creates new pairs (best and worst) and saves them.

        Args:
            saving_frequency (int, optional): Frequency for saving newly created pairs (every x new pairs, stores them). Defaults to 10.
        """
        best_pairs, worst_pairs = self.load_pairs(output_directory)

        english_df = self.load_english_data(english_dataset_path, best_pairs).dropna()[["word", "ipa", "custom_alphabet_5", "custom_alphabet_10", "custom_alphabet_15", "custom_alphabet_20"]]
        chinese_df = self.load_chinese_data(mandarin_dataset_path, best_pairs).dropna().rename(columns={"pinyin": "word"})[["word", "ipa", "custom_alphabet_5", "custom_alphabet_10", "custom_alphabet_15", "custom_alphabet_20"]]

        all_df = pd.concat((english_df, chinese_df)).sample(frac=1)

        for i, (_, row) in enumerate(all_df.iterrows()):
            bests, worsts = self.find_pair(row, english_df)
            bests = pd.DataFrame([pair.dict() for pair in bests])
            worsts = pd.DataFrame([pair.dict() for pair in worsts])

            best_pairs = pd.concat((best_pairs, pd.DataFrame(bests)))
            worst_pairs = pd.concat((worst_pairs, pd.DataFrame(worsts)))

            bests, worsts = self.find_pair(row, chinese_df)
            bests = pd.DataFrame([pair.dict() for pair in bests])
            worsts = pd.DataFrame([pair.dict() for pair in worsts])

            best_pairs = pd.concat((best_pairs, pd.DataFrame(bests)))
            worst_pairs = pd.concat((worst_pairs, pd.DataFrame(worsts)))

            if i % saving_frequency == 0:
                self.save_data(best_pairs, worst_pairs, output_directory)

    def find_pair(self, row, df: pd.DataFrame) -> List[Pair]:

        desc = "Finding closest word for {}"
        distances = [
            self.custom_distance(row, df_row)
            for _, df_row in tqdm(
                df.iterrows() # , desc=desc.format(str(row))
            )
        ]

        sorted_indexes = np.argsort(distances)
        limit_worst_index = sorted_indexes[-11]
        limit_best_index = sorted_indexes[10]
        distance_threshold = min(distances[limit_worst_index], distances[limit_best_index] + self.margin)
        offset = len([d for d in distances if d < distance_threshold])
        best_indexes = sorted_indexes[range(10)]
        worst_indexes = sorted_indexes[range(offset, offset+10)]
        # TODO: could make some combinations. 

        best_pairs = []
        worst_pairs = []

        for best_index, worst_index in zip(best_indexes, worst_indexes):
            best_row = df.iloc[best_index]
            worst_row = df.iloc[worst_index]
            best_dst = distances[best_index]
            worst_dst = distances[worst_index]

            best_pairs.append({
                "word_a": row["word"],
                "word_b": best_row["word"],
                "ipa_a": row["ipa"],
                "ipa_b": best_row["ipa"],
                "distance": best_dst,
            })

            worst_pairs.append({
                "word_a": row["word"],
                "word_b": worst_row["word"],
                "ipa_a": row["ipa"],
                "ipa_b": worst_row["ipa"],
                "distance": worst_dst,
            })

        return [Pair(**best_pair) for best_pair in best_pairs], [Pair(**worst_pair) for worst_pair in worst_pairs]

    def save_data(self, best_pairs: List[dict], worst_pairs: List[dict], output_directory: Path) -> None:
        """Saves best and worst pairs in a csv file.

        TODO: Use SQL instead.

        Args:
            best_pairs (List[dict]): Best pairs dataframe.
            worst_pairs (List[dict]): Worst pairs dataframe.
        """
        pd.DataFrame(best_pairs).to_csv(output_directory / self.best_pairs_suffix, index=False)
        pd.DataFrame(worst_pairs).to_csv(output_directory / self.worst_pairs_suffix, index=False)

    def load_pairs(self, pair_directory: Path) -> List[pd.DataFrame]:
        """Load, if existing, already computed pairs of english <> chinese terms.

        Returns:
            List[pd.DataFrame]: Best and Worst pairs dataframes.
        """
        print("Loading chinese corpus.")
        try:
            best_pairs = pd.read_csv(pair_directory / self.best_pairs_suffix)
            worst_pairs = pd.read_csv(pair_directory / self.worst_pairs_suffix)
        except:
            col_names = [
                "word_a",
                "word_b",
                "ipa_a",
                "ipa_b",
                "distance"
            ]
            best_pairs = pd.DataFrame(columns=col_names)
            worst_pairs = pd.DataFrame(columns=col_names)
        return best_pairs, worst_pairs

    def load_english_data(self, dataset_path: Path, pair_df: pd.DataFrame) -> pd.DataFrame:
        """Loads the chinese vocabulary dataframe.

        Returns:
            pd.DataFrame: English vocabulary dataframe.
        """
        print("Loading english corpus.")
        english_corpus = pd.read_csv(dataset_path)
        if not pair_df is None:
            english_corpus = english_corpus[~english_corpus["word"].isin(pair_df["word_a"])]
        return english_corpus[english_corpus["valid_ipa"] == True]

    def load_chinese_data(self, dataset_path: Path, pair_df: pd.DataFrame) -> pd.DataFrame:
        """Loads the chinese vocabulary dataframe.

        Args:
            pairs (pd.DataFrame): Existing pair dataframe to filter vocabulary with (in order not to create duplicates).

        Returns:
            pd.DataFrame: Chinese vocabulary dataframe.
        """
        print("Loading chinese corpus.")
        hsk_df = pd.read_csv(dataset_path)
        hsk_df = hsk_df[hsk_df["valid_ipa"] == True]
        if not pair_df is None:
            hsk_df = hsk_df[~hsk_df["hanzi"].isin(pair_df["word_a"])]
        return hsk_df

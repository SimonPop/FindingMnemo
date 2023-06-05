from typing import Tuple
from finding_mnemo.pairing.dataset.processing.custom_phonemes import (
    CustomPhonemes,
    IPA_CHARS,
)
import numpy as np
import panphon.distance

class PairGenerator:
    margin: float

    def __init__(self):
        self.cp = CustomPhonemes()
        self.dst = panphon.distance.Distance()
        self.ipa_features = {
            ipa: feature
            for ipa, feature in zip(IPA_CHARS, self.cp._get_ipa_features())
        }
        self.percentage_change_positive = 0.3
        self.percentage_change_negative = 0.75

    def generate_pair(self, word: str) -> dict:
        # Random select a number of edits.
        percentage_change_negative = np.random.random() * 0.5 + self.percentage_change_negative
        edit_number_positive = round(len(word)*self.percentage_change_positive)
        edit_number_negative = round(len(word)*percentage_change_negative)

        positive_word, positive_distance = self.generate_positive(word, edit_number_positive)
        negative_word, negative_distance = self.generate(word, edit_number_negative)

        positive_distance = self.dst.fast_levenshtein_distance_div_maxlen(word, positive_word)
        negative_distance = self.dst.fast_levenshtein_distance_div_maxlen(word, negative_word)

        if positive_distance > negative_distance:
            positive_word, positive_distance, negative_word, negative_distance = (
                negative_word,
                negative_distance,
                positive_word,
                positive_distance,
            )

        return {
            "positive_word": positive_word,
            "negative_word": negative_word,
            "positive_distance": positive_distance,
            "negative_distance": negative_distance,
        }

    def generate(self, word: str, edit_number: int) -> Tuple[str, str]:
        """Generates a positive and negative pair separated by margin distance."""

        # Distribute edit to each actions.
        actions = np.random.random(4)
        actions = (np.round(actions / sum(actions) * edit_number)).astype(int)

        total_distance = 0

        possible_switches = list(
            [(i, j) for i in range(len(word)) for j in range(i + 1, len(word))]
        )
        possible_replacements = list(
            range(len(word))
        )  # FIXME: Should not interefer with switches.
        np.random.shuffle(possible_switches)
        np.random.shuffle(possible_replacements)

        for action in range(actions[0]):
            # Select a switch from remaining
            if len(possible_switches) > 0:
                i1, i2 = possible_switches.pop()
                word, distance = self.switch(word, i1, i2)
                total_distance += distance

        for action in range(actions[1]):
            if len(possible_replacements) > 0:
                index = possible_replacements.pop()
                word, distance = self.replace(word, index, IPA_CHARS)
                total_distance += distance

        for action in range(min(len(word), actions[2])):
            if len(word) > 3:
                index = np.random.randint(0, len(word))
                word, distance = self.remove(word, index)
                total_distance += distance

        for action in range(actions[3]):
            index = np.random.randint(0, len(word))
            word, distance = self.add(word, index)
            total_distance += distance

        return word, total_distance


    def generate_positive(self, word: str, edit_number: int) -> Tuple[str, float]:

        # Distribute edit to each actions.
        actions = np.random.random(3)
        actions = (np.round(actions / sum(actions) * edit_number)).astype(int)

        total_distance = 0

        possible_replacements = list(
            range(len(word))
        )
        np.random.shuffle(possible_replacements)

        for action in range(actions[0]):
            if len(possible_replacements) > 0:
                index = possible_replacements.pop()
                letter = word[index]
                cluster_pool = self.cp.get_cluster(letter, level=5)
                word, distance = self.replace(word, index, cluster_pool)
                total_distance += distance

        for action in range(min(len(word), actions[1])):
            if len(word) > 3:
                index = np.random.choice([0, len(word)-1])
                word, distance = self.remove(word, index)
                total_distance += distance

        for action in range(actions[2]):
            index = np.random.choice([0, len(word)])
            word, distance = self.add(word, index)
            total_distance += distance

        return word, total_distance


        # Repalcements
        # Prefix: choose addition / removal / ignore
        # Suffix: choose addition / removal / ignore
        pass

    def switch(self, word: str, index_1: int, index_2: int) -> Tuple[str, float]:
        l1, l2 = word[index_1], word[index_2]
        distance = self.letter_distance(l1, l2)
        word = word[:index_1] + l2 + word[index_1 + 1 :]
        word = word[:index_2] + l1 + word[index_2 + 1 :]
        return word, distance

    def replace(self, word: str, index: int, pool: list) -> Tuple[str, float]:
        letter = word[index]
        new_letter, distance = self.select_new_letter(word[index], pool)
        word = word[:index] + new_letter + word[index + 1 :]
        return word, distance

    def add(self, word: str, index: int) -> Tuple[str, float]:
        new_letter, distance = self.select_new_letter(None)
        return word + new_letter, distance

    def remove(self, word: str, index: int) -> Tuple[str, float]:
        distance = self.letter_distance(word[index])
        word = word[:index] + word[index + 1 :]
        return word, distance

    def select_new_letter(self, target_letter: str, pool: list = IPA_CHARS) -> Tuple[str, float]:
        if target_letter is not None and target_letter not in pool:
            target_letter = None

        pool = [x for x in pool if x != target_letter]
        if len(pool) == 0:
            pool = [target_letter]
            
        new_letter = np.random.choice(pool)

        distance = self.letter_distance(new_letter, target_letter)

        return (new_letter, distance)

    def letter_distance(self, letter_1: str, letter_2: str = None) -> float:
        # TODO: normalize by worst distance in pairs.
        pool = self.ipa_features.keys()

        if letter_1 not in pool:
            return np.nan
        elif letter_2 is None or letter_2 not in pool:
            return np.linalg.norm(self.ipa_features[letter_1])
        distance = np.linalg.norm(
            self.ipa_features[letter_1] - self.ipa_features[letter_2]
        )
        return distance


if __name__ == "__main__":
    pair_gen = PairGenerator()
    print(pair_gen.generate_pair("tatjɛnxwa"))

    # For poisitive: only replace by similar role, only append or prepend

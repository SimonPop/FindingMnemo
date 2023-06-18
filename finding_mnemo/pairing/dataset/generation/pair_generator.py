from typing import Tuple
from finding_mnemo.pairing.utils.ipa import IPA_CHARS
from finding_mnemo.pairing.utils.distance import panphon_dtw, levenshtein_distance
from finding_mnemo.pairing.dataset.processing.custom_phonemes import CustomPhonemes
import numpy as np


class PairGenerator:
    margin: float

    def __init__(self, distance: str = "levenshtein"):
        self.cp = CustomPhonemes()

        if distance == 'dtw':
            self.metric = panphon_dtw
        elif distance == "levenshtein":
            self.metric = levenshtein_distance
        else:
            raise ValueError(f'Unknown distance metric {distance}. Chosse among: dtw, levenshtein.')

    def generate_tiplet_pair(self, word: str) -> dict:
        """Generates a negative and positive pair of ipas for triplet loss."""
        pos_edit_nb, neg_edit_nb = self.sample_edit_distance(word)

        positive_word = self.generate_positive(
            word, pos_edit_nb
        )
        negative_word = self.generate(word, neg_edit_nb)

        positive_distance = self.metric(
            word, positive_word
        )
        negative_distance = self.metric(
            word, negative_word
        )

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
    
    def sample_edit_distance(self, word: str):
        pos = np.random.random()
        neg = pos + np.random.random()*(1-pos)
        return round(pos*len(word)), round(neg*len(word))
    
    def generate(self, word: str, edit_number: int) -> Tuple[str, str]:
        """Generates a positive and negative pair separated by margin distance."""
        actions = self.distribute_actions(edit_number, 4)

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
                word = self.switch(word, i1, i2)

        for _ in range(actions[1]):
            if len(possible_replacements) > 0:
                index = possible_replacements.pop()
                word = self.replace(word, index, IPA_CHARS)

        for _ in range(min(len(word), actions[2])):
            if len(word) > 3:
                index = np.random.randint(0, len(word))
                word = self.remove(word, index)

        for _ in range(actions[3]):
            index = np.random.randint(0, len(word))
            word = self.add(word, index)

        return word

    def generate_positive(self, word: str, edit_number: int) -> str:
        actions = self.distribute_actions(edit_number, 3)

        possible_replacements = list(range(len(word)))
        np.random.shuffle(possible_replacements)

        for _ in range(actions[0]):
            if len(possible_replacements) > 0:
                index = possible_replacements.pop()
                letter = word[index]
                cluster_pool = self.cp.get_cluster(letter, level=5)
                word = self.replace(word, index, cluster_pool)

        for _ in range(min(len(word), actions[1])):
            if len(word) > 3:
                index = np.random.choice([0, len(word) - 1])
                word = self.remove(word, index)

        for _ in range(actions[2]):
            index = np.random.choice([0, len(word)])
            word = self.add(word, index)

        return word
    
    def distribute_actions(self, edit_number: int, action_cardinal: int):
        action_distribution = np.random.random(action_cardinal)
        action_distribution = (np.round(action_distribution / sum(action_distribution) * edit_number)).astype(int)
        return action_distribution

    def switch(self, word: str, index_1: int, index_2: int) -> str:
        """Switches two letters of place."""
        l1, l2 = word[index_1], word[index_2]
        word = word[:index_1] + l2 + word[index_1 + 1 :]
        word = word[:index_2] + l1 + word[index_2 + 1 :]
        return word

    def replace(self, word: str, index: int, pool: list) -> str:
        """Replaces letter at given index in the  given word chosing in the given pool."""
        new_letter = self.sample_letter(word[index], pool)
        word = word[:index] + new_letter + word[index + 1 :]
        return word

    def add(self, word: str, index: int) -> str:
        """Adds a letter at the given index in the word."""
        new_letter = self.sample_letter(None)
        return word[:index] + new_letter + word[index:]

    def remove(self, word: str, index: int) -> str:
        """Removes a letter at the given index in the word."""
        word = word[:index] + word[index + 1 :]
        return word

    def sample_letter(
        self, target_letter: str, pool: list = IPA_CHARS
    ) -> str:
        if target_letter is not None and target_letter not in pool:
            target_letter = None

        pool = [x for x in pool if x != target_letter]
        if len(pool) == 0:
            pool = [target_letter]

        new_letter = np.random.choice(pool)

        return new_letter

if __name__ == "__main__":
    pair_gen = PairGenerator(distance="dtw")
    print(pair_gen.generate_tiplet_pair("ˈɡʌvnmənt"))

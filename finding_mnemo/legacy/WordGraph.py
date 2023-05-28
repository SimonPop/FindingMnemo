from typing import List

import eng_to_ipa as ipa
import markovify
import networkx as nx
import numpy as np
import panphon.distance
from dragonmapper import hanzi, transcriptions
from tqdm import tqdm


class WordGraph:
    def __init__(
        self,
        corpus: str,
        state_size: int = 5,
        distance: str = "dolgo_prime_distance_div_maxlen",
    ):
        self.corpus = corpus
        self.cutoff = 20
        self.chain = None
        self.graph = nx.DiGraph()
        self.vocab = set()
        self.state_size = state_size
        self.create_markov_chain()
        self.convert_to_graph()
        self.phonetic_vocab = [ipa.convert(w) for w in self.vocab]
        self.dst = panphon.distance.Distance()
        self.dst_function = self.get_distance_function(distance)

    def create_markov_chain(self) -> None:
        self.chain = markovify.Text(self.corpus, state_size=self.state_size)
        # self.chain.compile(inplace = True)

    def convert_to_graph(self) -> None:
        chain_dict = markovify.utils.get_model_dict(self.chain)
        for start, ends in chain_dict.items():
            for end in ends:
                self.vocab.add(end)
                self.graph.add_edge(start, (start[-1], end))

        end_nodes = [x for x in self.graph.nodes() if "." in x[-1]]
        for n in end_nodes:
            self.graph.add_edge(n, "__END__")

    def find_closest_word(self, word: str) -> str:
        phonetic_word = transcriptions.pinyin_to_ipa(word)
        vocab = list(self.vocab)
        distances = [
            self.dst_function(phonetic_word, w2) for w2 in tqdm(self.phonetic_vocab)
        ]
        return vocab[np.argmin(distances)]

    def get_distance_function(self, dst: str):
        if dst == "dolgo_prime_distance":
            return self.dst.dolgo_prime_distance
        elif dst == "dolgo_prime_distance_div_maxlen":
            return self.dst.dolgo_prime_distance_div_maxlen
        elif dst == "weighted_feature_edit_distance_div_maxlen":
            return self.dst.weighted_feature_edit_distance_div_maxlen
        elif dst == "jt_weighted_feature_edit_distance":
            return self.dst.jt_weighted_feature_edit_distance

    def find_path(self, words: List[str]) -> List[str]:
        from itertools import combinations

        all_A = self.get_all_tuples(words[0])
        all_B = self.get_all_tuples(words[1])

        for a in all_A:
            for b in all_B:
                path_to_A = nx.shortest_path(
                    self.graph, source=("___BEGIN__", "___BEGIN__"), target=a
                )
                path_A_to_B = nx.shortest_path(self.graph, source=a, target=b)
                path_to_E = nx.shortest_path(self.graph, source=b, target="__END__")

                return path_to_A, path_A_to_B, path_to_E

        return None

    def get_all_tuples(self, word: str):
        return [n for n in self.graph.nodes if word in n]

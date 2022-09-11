import markovify
import networkx as nx
from typing import List
import panphon.distance
import numpy as np
from tqdm import tqdm
from dragonmapper import hanzi
from dragonmapper import transcriptions
import eng_to_ipa as ipa

class WordGraph():

    def __init__(self, corpus: str, state_size: int = 5):
        self.corpus = corpus
        self.chain = None
        self.graph = nx.DiGraph()
        self.vocab = set()
        self.state_size = state_size
        self.create_markov_chain()
        self.convert_to_graph()
        self.phonetic_vocab = [ipa.convert(w) for w in self.vocab]
        self.dst = panphon.distance.Distance()

    def create_markov_chain(self) -> None:
        self.chain = markovify.Text(self.corpus, state_size=self.state_size)
        # self.chain.compile(inplace = True)

    def convert_to_graph(self) -> None:
        chain_dict = markovify.utils.get_model_dict(self.chain)
        for start, ends in chain_dict.items():
            for end in ends:
                self.vocab.add(end)
                self.graph.add_edge(start, (start[-1], end))

        end_nodes = [x for x in self.graph.nodes() if '.' in x[-1]]
        for n in end_nodes:
            self.graph.add_edge(n, '__END__')

    def find_closest_word(self, word:str) -> str:
        phonetic_word = transcriptions.pinyin_to_ipa(word)
        vocab = list(self.vocab)
        distances = [self.dst.dolgo_prime_distance(phonetic_word, w2) for w2 in tqdm(self.phonetic_vocab)]
        return vocab[np.argmin(distances)]

    def find_path(self, words: List[str]) -> List[str]:
        paths = nx.all_simple_paths(self.graph, source=('___BEGIN__', '___BEGIN__'), target='__END__')
        for path in tqdm(paths):
            path_0 = [n[0] for n in path]
            path_1 = [n[1] for n in path]
            if all([word in path_0 or word in path_1 for word in words]):
                return path
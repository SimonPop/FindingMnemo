from torch.utils.data import Dataset
from dragonmapper import transcriptions
import eng_to_ipa as ipa
from typing import List
import yaml
from tqdm import tqdm
import panphon.distance
import numpy as np
from random import sample
import os

class SoundDataset(Dataset):
    def __init__(self, size: int = 10):

        self.size = size
        self.dst = panphon.distance.Distance()

        chinese_ipas = self.load_chinese_words()
        english_ipas = self.load_english_words()

        self.pair_dict = self.find_best_pairs(english_ipas, chinese_ipas)
        self.english_ipas = english_ipas
        self.chinese_ipas = chinese_ipas

    def __len__(self):
        return 2*self.size

    def find_best_pairs(self, english_vocabulary: List[str], chinese_vocabulary: List[str]):
        
        best_matches = []
        worst_matches = []

        for phonetic_word in english_vocabulary:
            distances = [self.dst.dolgo_prime_distance(phonetic_word, w2) for w2 in tqdm(chinese_vocabulary, desc='Finding closest word for {}'.format(phonetic_word))]
            best_matches.append((phonetic_word, chinese_vocabulary[np.argmin(distances)], np.argmin(distances)))
            worst_matches.append((phonetic_word, chinese_vocabulary[np.argmax(distances)], np.argmax(distances)))
        
        return {
            'positive': best_matches,
            'negative': worst_matches
        }

    def load_chinese_words(self) -> List[str]:
        script_dir = os.path.dirname(__file__)
        path = os.path.join(script_dir, './chinese_vocab_list.yaml')
        chinese_ipas = []
        with open(path, "r", encoding='utf-8') as stream:
            chinese_vocabulary = yaml.safe_load(stream)
            chinese_vocabulary = [c['pinyin'] for c in chinese_vocabulary]
            for word in chinese_vocabulary:
                try:
                    chinese_ipas.append(transcriptions.pinyin_to_ipa(word))
                except:
                    print('Impossible to transcribe: {}'.format(word))
            return chinese_ipas

    def load_english_words(self) -> List[str]:
        script_dir = os.path.dirname(__file__)
        path = os.path.join(script_dir, './common_english_words.txt')
        with open(path, "r") as stream:
            english_words = stream.readlines()
        # Select words of more than 5 letters
        english_words = [w for w in english_words if len(w) > 5]
        return [ipa.convert(w) for w in english_words]

    def __getitem__(self, index: int):
        if index % 2 == 0:
            english_match, chinese_match, distance = self.get_positive_pair(index//2)
        else:
            english_match, chinese_match, distance = self.get_negative_pair(index//2)
        return {
            'chinese_match': chinese_match, 
            'english_match': english_match, 
            'distance': distance,
            'negative': index % 2 != 0
            }

    def get_positive_pair(self, index: int):
        return self.pair_dict['positive'][index]

    def get_negative_pair(self, index: int):
        return self.pair_dict['negative'][index]

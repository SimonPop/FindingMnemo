from jina import Document, DocumentArray, Executor
from pathlib import Path
import pandas as pd 
import torch

from src.model.sound_siamese_v2 import SoundSiamese

class Indexer(Executor):
    model: SoundSiamese

    def index(self) -> DocumentArray:
        model = self.load_model()
        da = self.load_documents()
        # self.embed(da)
        return da

    def load_model(self) -> SoundSiamese:
        model = SoundSiamese()
        model.load_state_dict(torch.load(Path(__file__).parent.parent / "model" / "model_dict"))
        self.model = model
        return model

    def load_documents(self) -> DocumentArray:
        # TODO: Use batches to encode faster.
        dataframe = pd.read_csv(Path(__file__).parent.parent / 'dataset' / 'pairing' / 'english.csv')
        words = dataframe['word'].astype(str)
        # embedding = self.model.encode(words)
        with DocumentArray(
            storage='redis',
            config={
                'n_dim': 128,
                'index_name': 'idx',
            },
        ) as da:
            da.extend([Document(text=w, embedding=self.model.encode(w)) for w in words])
        return da

    def embed(self, da: DocumentArray):
        da.embed(self.model.encode)


i = Indexer().index()
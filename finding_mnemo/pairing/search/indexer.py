from pathlib import Path

import pandas as pd
import torch
from jina import Document, DocumentArray

from finding_mnemo.pairing.model.phonetic_siamese import PhoneticSiamese


class Indexer():
    model: PhoneticSiamese

    def index(self, **kwargs) -> DocumentArray:
        model = self.load_model()
        da = self.load_documents()
        return da

    def load_model(self) -> PhoneticSiamese:
        model = PhoneticSiamese()
        model.load_state_dict(
            torch.load(Path(__file__).parent.parent / "model" / "model_dict")
        )
        self.model = model
        model.eval()
        return model

    def load_documents(self) -> DocumentArray:
        dataframe = pd.read_csv(
            Path(__file__).parent.parent / "dataset" / "pairing" / "english.csv"
        )
        words = dataframe[["word", "ipa"]].astype(str)

        local_da = DocumentArray(
            [Document(text=w["word"], ipa=w["ipa"]) for _, w in words.iterrows()]
        )

        def embed(da: DocumentArray) -> DocumentArray:
            x = da[:, "tags__ipa"]
            da.embeddings = self.model.encode(x).detach()
            return da

        local_da.apply_batch(embed, batch_size=32)

        with DocumentArray(
            storage="redis",
            config={
                "n_dim": self.model.embedding_dim,
                "index_name": "english_words",
                "distance": "COSINE",
                "host": "redis",
                "port": "6379",
            },
        ) as da:
            da += local_da
        return da

    def embed(self, da: DocumentArray):
        da.embed(self.model.encode)
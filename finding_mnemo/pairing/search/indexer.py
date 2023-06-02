from pathlib import Path

import pandas as pd
import torch
import json
from docarray import Document, DocumentArray

from finding_mnemo.pairing.model.phonetic_siamese import PhoneticSiamese


class Indexer():
    model: PhoneticSiamese

    def index(self, **kwargs) -> DocumentArray:
        model = self.load_model()
        da = self.load_documents()
        return da

    def load_model(self) -> PhoneticSiamese:
        with open(Path(__file__).parent.parent / "model" / "model_config.json", "r") as f:
            model_config = json.load(f)
        model = PhoneticSiamese(
            embedding_dim=model_config["embedding_dim"],
            dim_feedforward=model_config["dim_feedforward"],
            nhead=model_config["nhead"],
            dropout=model_config["dropout"],
            batch_size=model_config["batch_size"],
            weight_decay=model_config["weight_decay"],
            lr=model_config["lr"],
            margin=model_config["margin"],
            lambda_triplet=model_config["lambda_triplet"],
            lambda_pos=model_config["lambda_pos"],
            lambda_neg=model_config["lambda_neg"],
        )
        model.load_state_dict(
            torch.load(Path(__file__).parent.parent / "model" / "model_dict")
        )
        self.model = model
        model.eval()
        return model

    def load_documents(self) -> DocumentArray:
        dataframe = pd.read_csv(
            Path(__file__).parent.parent / "dataset" / "data" / "english.csv"
        )
        words = dataframe[["word", "ipa"]].astype(str)

        local_da = DocumentArray(
            [Document(text=w["word"], ipa=w["ipa"]) for _, w in words.iterrows()]
        )

        local_da = DocumentArray([Document(text=w['word'], ipa=w['ipa']) for _, w in words.iterrows()])
        def embed(da: DocumentArray) -> DocumentArray:
                x = da[:,'tags__ipa']
                da.embeddings = self.model.encode(x).detach()
                return da
        local_da.apply_batch(embed, batch_size=32)

        with DocumentArray() as da:
            da += local_da
        return da

    def embed(self, da: DocumentArray):
        da.embed(self.model.encode)

if __name__ == "__main__":
    indexer = Indexer()
    indexer.load_model()
    indexer.load_documents()
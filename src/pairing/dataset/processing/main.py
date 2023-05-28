from pathlib import Path

import click

from src.pairing.dataset.processing.alphabet_converter import AlphabetConverter
from src.pairing.dataset.processing.pair_maker import PairMaker


@click.command()
@click.option(
    "--pair-margin",
    default=0.2,
    type=float,
    help='Minimum distance between a "good" and "bad" pair for the same word.',
)
@click.option(
    "--pair-saving-frequency",
    default=10,
    type=int,
    help="Saves new pair every x iterations into the file.",
)
def process(pair_margin: float, pair_saving_frequency: int):

    data_path = Path(__file__).parent.parent / "data"

    english_corpus_path = data_path / "english_corpus.txt"
    mandarin_corpus_path = data_path / "hsk"

    english_dataset_path = data_path / "english.csv"
    mandarin_dataset_path = data_path / "chinese.csv"

    # 1. Add alphabet information to each corpuses.
    converter = AlphabetConverter()

    english_dataset = converter.process_english_corpus(english_corpus_path)
    mandarin_dataset = converter.process_mandarin_corpus(mandarin_corpus_path)

    # Saving processed datasets.

    english_dataset.to_csv(english_dataset_path)
    mandarin_dataset.to_csv(mandarin_dataset_path)

    pair_maker = PairMaker(margin=pair_margin)

    # 2. Create pairs (good and bad).
    pair_maker.make_pairs(
        english_dataset_path,
        mandarin_dataset_path,
        output_directory=data_path,
        saving_frequency=pair_saving_frequency,
    )


if __name__ == "__main__":
    process()

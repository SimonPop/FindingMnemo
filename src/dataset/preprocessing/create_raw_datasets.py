from pathlib import Path
import pandas as pd
from dragonmapper import hanzi
from eng_to_ipa import convert
import panphon.distance
from src.dataset.phonemes.custom_phonemes import CustomPhonemes
from tqdm import tqdm
tqdm.pandas()

dst = panphon.distance.Distance()
file_path = Path(__file__).parent
cp = CustomPhonemes()

def convert_eng_to_ipa(word: str):
    return (
        convert(word, keep_punct=False)
        .replace("ˈ", "")
        .replace("ˌ", "")
        .replace(" ", "")
    )

def convert_to_custom(word: str, valid: bool, level: int): 
    if valid:
        return cp.convert(word, level)
    else:
        return None

def create_english_dataset():
    # Load raw data
    raw_df = pd.read_csv(
        file_path / "english_corpus.txt", sep=" ", header=None
    ).rename(columns={0: "word", 1: "occurence"})
    # Filter out values too low
    filter_occ = 5000
    raw_df = raw_df[raw_df["occurence"] > filter_occ]
    # Filter out words too short
    # raw_df = raw_df[raw_df['word'].str.len() > 2]
    # Create IPA
    raw_df["ipa"] = raw_df["word"].astype(str).progress_apply(convert_eng_to_ipa)
    raw_df["valid_ipa"] = ~raw_df["ipa"].str.contains("*", regex=False)
    raw_df["dolgo"] = raw_df["ipa"].apply(dst.map_to_dolgo_prime)
    raw_df["custom_alphabet_5"] = raw_df.progress_apply(lambda x: convert_to_custom(x["ipa"], x["valid_ipa"], level=5), axis=1)
    raw_df["custom_alphabet_10"] = raw_df.progress_apply(lambda x: convert_to_custom(x["ipa"], x["valid_ipa"], level=10), axis=1)
    raw_df["custom_alphabet_15"] = raw_df.progress_apply(lambda x: convert_to_custom(x["ipa"], x["valid_ipa"], level=15), axis=1)
    raw_df["custom_alphabet_20"] = raw_df.progress_apply(lambda x: convert_to_custom(x["ipa"], x["valid_ipa"], level=20), axis=1)
    return raw_df


def filter_chinese_ipa(ipa: str):
    from dragonmapper import transcriptions

    _IPA_CHARACTERS = transcriptions._IPA_CHARACTERS
    # Remove spaces, tones etc.
    return "".join([x for x in ipa if x in _IPA_CHARACTERS])


def convert_mandarin_to_ipa(h: str):
    try:
        return filter_chinese_ipa(hanzi.to_ipa(h))
    except:
        return "*"


def create_chinese_dataset():
    # Load raw data
    hsk_1 = pd.read_csv(file_path / "hsk/hsk1.csv", header=None)
    hsk_2 = pd.read_csv(file_path / "hsk/hsk2.csv", header=None)
    hsk_3 = pd.read_csv(file_path / "hsk/hsk3.csv", header=None)
    hsk_4 = pd.read_csv(file_path / "hsk/hsk4.csv", header=None)
    hsk_5 = pd.read_csv(file_path / "hsk/hsk5.csv", header=None)
    hsk_6 = pd.read_csv(file_path / "hsk/hsk6.csv", header=None)
    raw_df = pd.concat(
        [hsk_1, hsk_2, hsk_3, hsk_4, hsk_5, hsk_6], ignore_index=True
    ).rename(columns={0: "hanzi", 1: "pinyin", 2: "translation"})
    # Create IPA
    raw_df["ipa"] = raw_df["hanzi"].apply(convert_mandarin_to_ipa)
    raw_df["valid_ipa"] = ~raw_df["ipa"].str.contains("*", regex=False)
    raw_df["dolgo"] = raw_df["ipa"].apply(dst.map_to_dolgo_prime)
    raw_df["custom_alphabet_5"] = raw_df.progress_apply(lambda x: convert_to_custom(x["ipa"], x["valid_ipa"], level=5), axis=1)
    raw_df["custom_alphabet_10"] = raw_df.progress_apply(lambda x: convert_to_custom(x["ipa"], x["valid_ipa"], level=10), axis=1)
    raw_df["custom_alphabet_15"] = raw_df.progress_apply(lambda x: convert_to_custom(x["ipa"], x["valid_ipa"], level=15), axis=1)
    raw_df["custom_alphabet_20"] = raw_df.progress_apply(lambda x: convert_to_custom(x["ipa"], x["valid_ipa"], level=20), axis=1)
    return raw_df


create_english_dataset().to_csv("english.csv")
create_chinese_dataset().to_csv("chinese.csv")

from tslearn.metrics import dtw_path_from_metric
from finding_mnemo.pairing.utils.ipa import IPA_FEATURE_DICT
import panphon.distance

dst = panphon.distance.Distance()

def panphon_dtw(s1: str, s2: str) -> float:
    """Computes a distance between two IPA strings based on DTW and panphon features."""
    s1_b = [list(IPA_FEATURE_DICT[c]) if c in IPA_FEATURE_DICT else [0]*24 for c in s1]
    s2_b = [list(IPA_FEATURE_DICT[c]) if c in IPA_FEATURE_DICT else [0]*24 for c in s2]
    MAX_DIST = 6.557438524302 # Computed  on each pair distance in possible phonemes.
    return dtw_path_from_metric(s1_b, s2_b, metric="euclidean")[-1] / MAX_DIST

def levenshtein_distance(s1: str, s2: str) -> float:
    """Levenshtein distance between two IPA strings."""
    return dst.fast_levenshtein_distance_div_maxlen(s1, s2)
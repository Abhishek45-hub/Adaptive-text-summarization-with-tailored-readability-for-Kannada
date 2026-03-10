import re
from typing import List

SENT_SPLIT = re.compile(r"[.!?]|[।]|[|]")
PUNC = re.compile(r"[^\w\s\u0C80-\u0CFF]")
SPACE = re.compile(r"\s+")

KANNADA_INDEP_VOWELS = set("ಅಆಇಈಉಊಋೠಎಏಐಒಓಔಌೡ")
KANNADA_VOWEL_SIGNS = set("ಾಿೀುೂೃೄೆೇೈೊೋೌ")

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]

def words(text: str) -> List[str]:
    clean = PUNC.sub(" ", text)
    return [w for w in SPACE.split(clean) if w]

def count_syllables(word: str) -> int:
    return sum(1 for ch in word if ch in KANNADA_INDEP_VOWELS or ch in KANNADA_VOWEL_SIGNS)

def readability_kre(text: str) -> float:
    sents = split_sentences(text)
    wlist = words(text)
    if not sents or not wlist:
        return 50.0

    asl = len(wlist) / len(sents)
    syll = sum(count_syllables(w) for w in wlist)
    awl = syll / len(wlist)

    kre = 206.835 - 1.015 * asl - 84.6 * awl
    return max(0.0, min(100.0, kre))

def kre_category(kre: float) -> str:
    if kre >= 80: return "ಪ್ರಾಥಮಿಕ"
    if kre >= 60: return "ಮಧ್ಯಮಿಕ"
    if kre >= 40: return "ಹೈಸ್ಕೂಲ್"
    return "ಕಾಲೇಜು"

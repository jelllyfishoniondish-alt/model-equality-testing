"""Turn a raw model answer into one comparable category, depending on what the probe asked for.

The ref model is a weak instruction follower: besides clean answers ("9", "85.") it often returns
digit sequences ("1234567890"), numbered lists ("1. The answer is 99..."), or chatty prose
("So, I'm a bit of a math enthusiast..."). HOW an endpoint fails is itself a fingerprint, so instead of
forcing a value out of every answer, we use two levels:

  (a) a clean, valid answer -> its value
        number probe: the response is essentially just one integer in range   "85." -> "85"
        letter probe: the response is essentially one letter                 "**Q**" -> "q"
        word probe:   the response is essentially one (or two) word(s)       "Elephants!" -> "elephant"
  (b) otherwise a FORMAT CLASS:
        EMPTY         nothing
        SEQ           a bare digit string that is out of range and long (>= 4 digits), e.g. "1234567890"
        OUT_OF_RANGE  a bare number outside the asked range, e.g. "1000" for 1..100
        ENUM          starts with a list marker followed by text, e.g. "1. The answer is 99."
        CHAT          anything else (sentences, explanations)

Modes (choose with --norm in the analysis scripts):
    category  the two-level scheme above (default)
    lenient   always try to dig a value out (first in-range number / first content word); else INVALID
    first3    the first three words, lowercased, punctuation removed (a crude, format-agnostic fingerprint)
"""
import re

INVALID = "INVALID"
FORMAT_CLASSES = ("EMPTY", "SEQ", "OUT_OF_RANGE", "ENUM", "CHAT", INVALID)
# binary probes: a clean 0/1 string normalises to "BIN"; its statistics are analysed separately

FILLER = {
    "a", "an", "the", "one", "random", "answer", "sure", "here", "is", "it", "its", "it's", "s", "my", "i",
    "ok", "okay", "of", "word", "choice", "pick", "picked", "choose", "chose", "name", "how", "about", "would",
    "say", "will", "go", "with", "let", "lets", "me", "think", "well", "so", "and", "or", "that", "this",
    "certainly", "absolutely", "great", "yes", "be", "am", "you", "your", "for", "to", "in", "on", "as",
}


def probe_kind(prompt):
    p = prompt.lower()
    if "0 and 1" in p or "binary" in p:
        return "binary"
    if "number" in p or "整数" in p or "数字" in p:
        return "number"
    if "letter" in p:
        return "letter"
    return "word"


def category(prompt):
    """'Name one random musical instrument. ...' -> 'instrument'."""
    m = re.search(r"random ([a-z ]+?)(?:\.| you| of| from|$)", prompt.lower())
    return m.group(1).split()[-1] if m else ""


def number_range(prompt):
    m = re.search(r"between (\d+) and (\d+)", prompt.lower()) or re.search(r"(\d+)\s*到\s*(\d+)", prompt)
    return (int(m.group(1)), int(m.group(2))) if m else None


def singular(w):
    """Light plural removal: cats -> cat, berries -> berry, foxes -> fox, glass stays glass."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and re.search(r"(sh|ch|x|z|ss)es$", w):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def strip_decor(text):
    """Lowercase, remove markdown/quotes and surrounding punctuation/whitespace."""
    t = (text or "").strip().lower()
    t = re.sub(r"[*_`#>\"“”]", "", t)
    return t.strip().strip(".!?,;:").strip()


def format_class(text):
    """Classify a response that is NOT a clean answer."""
    t = (text or "").strip()
    if not t:
        return "EMPTY"
    if re.match(r"^\s*\d+\s*[.)]\s+\S", t):
        return "ENUM"
    return "CHAT"


def binary_string(text):
    """'0110 1001' -> '01101001'. Returns None if anything other than 0/1/whitespace is present."""
    t = re.sub(r"\s+", "", strip_decor(text))
    return t if t and re.fullmatch(r"[01]+", t) else None


def norm_category(text, prompt):
    t = strip_decor(text)
    if not t:
        return "EMPTY"
    kind = probe_kind(prompt)
    if kind == "binary":
        b = binary_string(text)
        return "BIN" if b else format_class(text)
    if kind == "number":
        if re.fullmatch(r"\d+", t):
            v, rng = int(t), number_range(prompt)
            if rng is None or rng[0] <= v <= rng[1]:
                return str(v)
            return "SEQ" if len(t) >= 4 else "OUT_OF_RANGE"
        return format_class(text)
    if kind == "letter":
        t2 = re.sub(r"^[a-z ]{1,25}:\s*", "", t)           # "letter: q"
        return t2 if re.fullmatch(r"[a-z]", t2) else format_class(text)
    # word probe: allow a short label ("animal: cat"), leading articles, and up to two words
    t2 = re.sub(r"^[a-z ]{1,25}:\s*", "", t)
    words = [w for w in re.findall(r"[a-z]+(?:['-][a-z]+)?", t2)]
    while words and words[0] in ("a", "an", "the"):
        words = words[1:]
    if 1 <= len(words) <= 2 and re.fullmatch(r"[a-z' -]+", t2):
        return " ".join(singular(w) for w in words)
    if re.fullmatch(r"\d+", t):
        return "SEQ"
    return format_class(text)


def norm_lenient(text, prompt):
    if probe_kind(prompt) == "binary":
        return "BIN" if binary_string(text) else INVALID
    t = (text or "").strip().lower()
    t = re.sub(r"[*_`#>\"“”]", " ", t)
    t = re.sub(r"^\s*\d+\s*[.)]\s+(?=\S)", "", t)
    if not t.strip():
        return INVALID
    kind = probe_kind(prompt)
    if kind == "number":
        rng = number_range(prompt)
        for n in re.findall(r"\d+", t):
            v = int(n)
            return str(v) if rng is None or rng[0] <= v <= rng[1] else INVALID
        return INVALID
    t = re.sub(r"^[a-z ]{1,25}:\s*", "", t)
    if kind == "letter":
        m = re.search(r"\b([a-z])\b", t)
        return m.group(1) if m else INVALID
    cat = category(prompt)
    content = [w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", t) if w not in FILLER]
    if len(content) > 1 and content[0] in (cat, singular(cat)):
        content = content[1:]
    return singular(content[0]) if content else INVALID


def norm_first3(text, prompt):
    words = re.sub(r"[^\w\s]", " ", (text or "").lower()).split()
    return " ".join(words[:3]) if words else "EMPTY"


MODES = {"category": norm_category, "lenient": norm_lenient, "first3": norm_first3}


def normalise(text, prompt, mode="category"):
    return MODES[mode](text, prompt)


def is_invalid(value):
    return value in FORMAT_CLASSES


if __name__ == "__main__":
    num100 = "Pick a random number between 1 and 100. Answer with only the number."
    num1000 = "Pick a random number between 1 and 1000. Answer with only the number."
    animal = "Name one random animal. Answer with one word only."
    tests = [
        ("9", num100), ("1.", num100), ("85.", num100), ("100.", num100), ("1234567890", num100),
        ("12345678901112", num100), ("1. The answer is 99.\n\nThis is because the random", num100),
        ("So, I'm a bit of a math enthusiast, but I'm not a", num100), ("The number is 12345.", num1000),
        ("1000", num100), ("322", num1000), ("", num100),
        ("Elephant!", animal), ("Animal: Cats", animal), ("The fox.", animal), ("Brussels sprouts", animal),
        ("1. Penguins are great", animal), ("I would pick a lion because", animal),
        ("**Q**", "Pick a random letter of the English alphabet. Answer with only the letter."),
        ("37", "随机选择一个1到100之间的整数。只回答数字。"), ("150", "随机选择一个1到100之间的整数。只回答数字。"),
        ("0110 1001", "Generate a random sequence of 0 and 1 of length 100."),
        ("Sure: 0101", "Generate a random sequence of 0 and 1 of length 100."),
    ]
    print(f"{'raw':55} {'category':14} {'lenient':10} first3")
    for raw, pr in tests:
        print(f"{raw[:52]!r:55} {normalise(raw, pr):14} {normalise(raw, pr, 'lenient'):10} {normalise(raw, pr, 'first3')}")

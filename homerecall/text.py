"""Text handling for captions and questions: prompts, parsing a model's reply, keyword
search (BM25), and reading a time range out of a question.

Pure Python, unit-tested without a model. The object-name helpers come from LiveScene's
`describe.py`.
"""

import datetime as dt
import math
import re

# ---------------------------------------------------------------- prompts

# Small models: a plain question. Listing actions to look for ("open, pick up, carry") made
# the 500M model report those actions in a clip of a dog running across a square.
CAPTION_PROMPT = (
    "These frames are from a home security camera, in time order. Describe in two sentences "
    "what the person is doing and what they are holding or using. "
    "Only mention things you can clearly see."
)
CAPTION_PROMPT_NOBODY = (
    "These frames are from a home security camera, in time order. In two sentences, say what "
    "is happening. Only mention things you can clearly see."
)
LIST_PROMPT = "List the main objects visible in the image as a short comma-separated list of singular nouns."
# Capable models return both fields in one reply.
PROPOSE_PROMPT = (
    "These frames are from a home security camera, in time order.\n"
    "Use exactly this format:\n"
    "OBJECTS: comma-separated singular nouns for the objects the person touches or uses and the "
    "furniture or appliances involved, at most 10\n"
    "DESCRIPTION: two or three sentences saying what the person does, step by step: where they "
    "are and what they open, pick up, carry or put down. Only mention things you can clearly see."
)
VERIFY_PROMPT = "These frames are from a home security camera, in time order. {question}"

# ---------------------------------------------------------------- object names

_ARTICLES = {"a", "an", "the", "some", "several", "many", "two", "three", "four", "five", "one", "of", "and"}
_NOT_OBJECTS = {"none", "nothing", "n/a", "object", "objects", "thing", "things", "scene", "image", "photo",
                "picture", "background", "foreground", "frame", "camera"}
_PERSON = {"person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children", "kid",
           "guy", "lady", "adult", "human", "persons", "someone", "somebody", "anyone", "anybody"}
_KEEP_PLURAL = {"glasses", "sunglasses", "scissors", "pants", "shorts", "jeans", "trousers", "headphones",
                "earphones", "stairs", "clothes", "goggles", "pliers", "tongs", "bus", "grass", "keys"}


def singular(word):
    if word in _KEEP_PLURAL or len(word) < 4 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    return word[:-1] if word.endswith("s") else word


def clean_object(name):
    """Normalise one object name; '' if it is not usable as a detector prompt."""
    words = [w for w in re.findall(r"[a-z][a-z\-]*", name.lower()) if w not in _ARTICLES]
    if not words or len(words) > 3:
        return ""
    words[-1] = singular(words[-1])
    if words[-1] in _PERSON:
        return "person"
    out = " ".join(words)
    return "" if out in _NOT_OBJECTS or len(out) < 2 else out


def parse_list(text, max_objects=10):
    """Objects from a free-form list reply (commas, new lines or bullets)."""
    out, seen = [], set()
    for part in re.split(r"[,;\n]|(?:^|\s)[-*•]\s|\band\b", re.sub(r"^[^:\n]{0,40}:", "", text.strip())):
        name = clean_object(part)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out[:max_objects]


def parse_proposal(text, max_objects=10):
    """Split an `OBJECTS: ... DESCRIPTION: ...` reply into (objects, description). With no
    OBJECTS line the whole reply is the description."""
    m_obj = re.search(r"objects\s*:\s*(.*?)(?=\n\s*description\s*:|$)", text, re.I | re.S)
    m_desc = re.search(r"description\s*:\s*(.*)", text, re.I | re.S)
    objects = parse_list(m_obj.group(1), max_objects) if m_obj else []
    if m_desc:
        description = m_desc.group(1)
    elif m_obj:
        description = text[: m_obj.start()] + text[m_obj.end():]
    else:
        description = text
    description = re.sub(r"\s+", " ", re.sub(r"objects\s*:.*", "", description, flags=re.I)).strip()
    return objects, description


def complete_sentences(text):
    """Drop a trailing fragment left when generation hit its token limit mid-sentence."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text[-1] in ".!?":
        return text
    whole = [s for s in re.split(r"(?<=[.!?])\s+", text) if s][:-1]
    return " ".join(whole) if whole else text


# ---------------------------------------------------------------- keywords

_STOP = _ARTICLES | {
    "did", "does", "do", "is", "are", "was", "were", "has", "have", "had", "be", "been", "any", "anyone",
    "anybody", "someone", "somebody", "who", "what", "when", "where", "which", "why", "how", "there", "here",
    "this", "that", "these", "those", "it", "its", "they", "them", "their", "he", "she", "his", "her", "i",
    "me", "my", "we", "our", "you", "your", "to", "from", "in", "on", "at", "by", "for", "with", "into", "out",
    "up", "down", "off", "over", "under", "or", "not", "no", "yes", "can", "could", "would", "should", "will",
    "show", "find", "clip", "clips", "video", "time", "ever", "just", "then", "while", "as", "if", "so",
    "person", "people", "man", "woman", "frame", "frames", "camera", "home", "security", "see", "seen",
    "visible", "clearly", "something", "anything", "thing", "being", "happen", "happening",
}
_IRREGULAR = {
    "took": "take", "taken": "take", "taking": "take", "left": "leave", "leaving": "leave", "came": "come",
    "coming": "come", "went": "go", "gone": "go", "ate": "eat", "eaten": "eat", "eating": "eat",
    "drank": "drink", "drunk": "drink", "brought": "bring", "put": "put", "putting": "put", "sat": "sit",
    "sitting": "sit", "stood": "stand", "ran": "run", "running": "run", "got": "get", "getting": "get",
    "gave": "give", "given": "give", "held": "hold", "made": "make", "making": "make", "broke": "break",
    "broken": "break", "fell": "fall", "fallen": "fall", "threw": "throw", "thrown": "throw", "fed": "feed",
    "wore": "wear", "worn": "wear", "stole": "steal", "stolen": "steal", "children": "child", "men": "man",
    "women": "woman", "knives": "knife", "shelves": "shelf", "leaves": "leave", "grabbed": "grab",
    "grabbing": "grab", "removed": "remove", "removing": "remove", "closing": "close", "closed": "close",
    "using": "use", "used": "use",
}
# Words that mean the same thing in a home are mapped to one token.
_SYNONYM = {
    "fridge": "refrigerator", "freezer": "refrigerator", "couch": "sofa", "television": "tv",
    "telly": "tv", "cellphone": "phone", "smartphone": "phone", "mobile": "phone", "parcel": "package",
    "delivery": "package", "bin": "trash", "garbage": "trash", "rubbish": "trash", "cupboard": "cabinet",
    "grab": "take", "remove": "take", "pick": "take", "retrieve": "take", "get": "take", "carry": "take",
    "kitty": "cat", "kitten": "cat", "puppy": "dog", "notebook": "laptop", "computer": "laptop",
    "kid": "child", "boy": "child", "girl": "child", "baby": "child", "enter": "come", "arrive": "come",
    "exit": "leave", "depart": "leave", "consume": "eat",
}


# Action words: useful as keywords, useless as detector prompts.
_VERBS = {v for v in _IRREGULAR.values() if v not in ("child", "man", "woman", "knife", "shelf")} | {
    "open", "walk", "look", "touch", "move", "drop", "cook", "wash", "clean", "sleep", "play", "turn", "watch",
    "carry", "enter", "exit", "arrive", "knock", "ring", "deliver", "climb", "jump", "pour", "cut", "lock", "unlock",
    "pick", "retrieve", "consume", "depart", "wear", "visit", "pass", "reach", "search", "check", "happen",
}


def stem(word):
    """A deliberately small stemmer: irregular forms, then -ing / -ed / plural."""
    w = word.lower()
    if w in _IRREGULAR:
        w = _IRREGULAR[w]
    elif len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
        if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "ls":  # running -> run, but fill, pass
            w = w[:-1]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
        if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "ls":
            w = w[:-1]
    else:
        w = singular(w)
    return _SYNONYM.get(w, w)


def tokens(text):
    """Content words of a caption or question, stemmed, stop words removed."""
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in _STOP:
            continue
        s = stem(w)
        if s not in _STOP and len(s) > 1:
            out.append(s)
    return out


def query_objects(question):
    """Words of the question worth giving to the detector as prompts: its content words
    that are not known action words, singular and with home synonyms resolved. An unknown
    verb can slip through; the detector simply finds nothing for it."""
    out = []
    for w in re.findall(r"[a-z]+", question.lower()):
        if w in _STOP or stem(w) in _VERBS or (w.endswith("ing") and len(w) > 5):
            continue
        s = singular(w)
        s = _SYNONYM.get(s, s)
        if s not in _STOP and len(s) > 2 and s not in out:
            out.append(s)
    return out[:8]


class BM25:
    """Okapi BM25 over tokenised documents."""

    def __init__(self, docs, k1=1.4, b=0.6):
        self.docs = [tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query):
        q = set(tokens(query))
        out = []
        for d in self.docs:
            s = 0.0
            for t in q:
                f = d.count(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / (self.avg or 1)))
            out.append(s)
        return out


def fuse(rankings, k=60):
    """Reciprocal rank fusion. `rankings` is a list of {id: score}; in each one only ids
    with a positive score take part. Returns {id: fused score}."""
    fused = {}
    for scores in rankings:
        ordered = sorted((i for i, s in scores.items() if s > 0), key=lambda i: -scores[i])
        for rank, i in enumerate(ordered):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank + 1)
    return fused


# ---------------------------------------------------------------- time ranges

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"])}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_PARTS = {"morning": (5, 12), "afternoon": (12, 17), "evening": (17, 22), "night": (21, 29), "tonight": (18, 29)}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))


def parse_when(question, now=None):
    """Read a time range out of a question.

    Returns (start, end, rest): `start` and `end` are datetimes or None, and `rest` is the
    question without the time words, which is what the language model is asked (it cannot
    know what "yesterday" means from the frames)."""
    now = now or dt.datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    text = question
    day = None
    span_days = 1

    def cut(m):
        nonlocal text
        text = text[: m.start()] + " " + text[m.end():]

    m = re.search(r"\b(?:on\s+)?(20\d{2})-(\d{1,2})-(\d{1,2})\b", text)
    if m:
        try:
            day = dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            cut(m)
        except ValueError:
            pass
    if day is None:
        m = (re.search(rf"\b(?:on\s+)?(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\b\.?", text, re.I)
             or re.search(rf"\b(?:on\s+)?({_MONTH_RE})\b\.?\s+(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\b", text, re.I))
        if m:
            a, b = m.group(1), m.group(2)
            d, mon = (int(a), _MONTHS[b.lower()]) if a.isdigit() else (int(b), _MONTHS[a.lower()])
            try:
                day = dt.datetime(now.year, mon, d)
                if day > today:  # "december 30" asked in January means last year
                    day = day.replace(year=now.year - 1)
                cut(m)
            except ValueError:
                day = None
    if day is None:
        m = re.search(r"\b(last night|yesterday|today|tonight|this (?=morning|afternoon|evening))\b", text, re.I)
        if m:
            word = m.group(1).lower()
            if word == "last night":
                start = today - dt.timedelta(days=1) + dt.timedelta(hours=_PARTS["night"][0])
                end = today - dt.timedelta(days=1) + dt.timedelta(hours=_PARTS["night"][1])
                cut(m)
                return start, end, _tidy(text)
            if word == "tonight":
                cut(m)
                return (today + dt.timedelta(hours=_PARTS["tonight"][0]),
                        today + dt.timedelta(hours=_PARTS["tonight"][1]), _tidy(text))
            day = today - dt.timedelta(days=1) if word == "yesterday" else today
            cut(m)
    if day is None:
        m = re.search(r"\b(?:in the |over the )?(?:last|past) (\d+|few|couple of)? ?(day|week|month)s?\b", text, re.I)
        if m:
            n = {"few": 3, "couple of": 2, None: 1}.get(m.group(1), None) or int(m.group(1))
            days = n * {"day": 1, "week": 7, "month": 30}[m.group(2).lower()]
            cut(m)
            return today - dt.timedelta(days=days), now, _tidy(text)
        m = re.search(r"\bthis week\b", text, re.I)
        if m:
            cut(m)
            return today - dt.timedelta(days=today.weekday()), now, _tidy(text)
    if day is None:
        m = re.search(rf"\b(?:on |last )?({'|'.join(_WEEKDAYS)})\b", text, re.I)
        if m:
            back = (today.weekday() - _WEEKDAYS.index(m.group(1).lower())) % 7
            day = today - dt.timedelta(days=back or (7 if m.group(0).lower().startswith("last") else 0))
            cut(m)

    part = re.search(r"\b(?:in the |at |during the )?(morning|afternoon|evening|night)\b", text, re.I)
    if part and (day is not None):
        a, b = _PARTS[part.group(1).lower()]
        cut(part)
        return day + dt.timedelta(hours=a), day + dt.timedelta(hours=b), _tidy(text)
    if day is not None:
        return day, day + dt.timedelta(days=span_days), _tidy(text)
    return None, None, _tidy(question)


def _tidy(text):
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([?.!,])", r"\1", text)

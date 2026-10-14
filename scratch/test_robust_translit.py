"""Test script for robust transliteration & translation of English and Marathlish queries."""
import re
import json
import urllib.parse
import urllib.request

_COMMON_WORDS = {
    # Variations of सामसूम (Saamsoom)
    "samsoom": "सामसूम",
    "saamsoom": "सामसूम",
    "ssamsoom": "सामसूम",
    "saamsuum": "सामसूम",
    "ssamsuum": "सामसूम",
    "samsuum": "सामसूम",
    "saamsum": "सामसूम",
    "samsum": "सामसूम",
    "samsoomche": "सामसूमचे",
    # Words related to शब्द (Shabda) / वाक्प्रचार (Vakprachar)
    "shabd": "शब्द",
    "shabda": "शब्द",
    "shabdache": "शब्दाचा",
    "shabdacha": "शब्दाचा",
    "shabdanchi": "शब्दांची",
    "shabdanche": "शब्दांचे",
    "shabdarth": "शब्दार्थ",
    "shabdararth": "शब्दार्थ",
    "vakprachar": "वाक्प्रचार",
    "vakpracharacha": "वाक्प्रचाराचा",
    "vakya": "वाक्य",
    "vakyacha": "वाक्याचा",
    # Demonstratives
    "ha": "या",
    "he": "हे",
    "hi": "ही",
    "ya": "या",
    "hya": "ह्या",
    "tya": "त्या",
    "to": "तो",
    "ti": "ती",
    "te": "ते",
    # Question words (English & Marathlish)
    "kay": "काय",
    "kai": "काय",
    "what": "काय",
    "whats": "काय",
    "what's": "काय",
    "kasa": "कसा",
    "kashi": "कशी",
    "kase": "कसे",
    "how": "कसे",
    "kon": "कोण",
    "koni": "कोणी",
    "kona": "कोणा",
    "who": "कोण",
    "whos": "कोण",
    "who's": "कोण",
    "whose": "कोणाचे",
    "whom": "कोणाला",
    "konata": "कोणता",
    "konti": "कोणती",
    "konte": "कोणते",
    "which": "कोणता",
    "kothe": "कुठे",
    "kuthe": "कुठे",
    "where": "कुठे",
    "kadhi": "कधी",
    "when": "कधी",
    "ka": "का",
    "why": "का",
    # Authors, Poets, Writers & Typos
    "author": "लेखक",
    "auther": "लेखक",
    "authur": "लेखक",
    "autor": "लेखक",
    "authors": "लेखक",
    "writer": "लेखक",
    "writter": "लेखक",
    "writers": "लेखक",
    "write": "लिहा",
    "wrote": "लिहिले",
    "written": "लिहिले",
    "lekhak": "लेखक",
    "lekakh": "लेखक",
    "likhak": "लेखक",
    "lekhika": "लेखिका",
    "poet": "कवी",
    "poets": "कवी",
    "poetess": "कवयित्री",
    "kavi": "कवी",
    "kaviyitri": "कवयित्री",
    "kavayitri": "कवयित्री",
    # Verbs / Actions
    "sang": "सांग",
    "sanga": "सांगा",
    "sangto": "सांगतो",
    "sangte": "सांगते",
    "tell": "सांगा",
    "dya": "द्या",
    "de": "दे",
    "give": "द्या",
    "hone": "होणे",
    "honar": "होणार",
    "zale": "झाले",
    "jhale": "झाले",
    "ahe": "आहे",
    "aahe": "आहे",
    "is": "आहे",
    "are": "आहेत",
    "was": "होते",
    "were": "होते",
    "nahi": "नाही",
    "not": "नाही",
    "pahije": "पाहिजे",
    "hava": "हवा",
    "havi": "हवी",
    "want": "हवे",
    # Meanings & Vocabulary
    "artha": "अर्थ",
    "arth": "अर्थ",
    "artho": "अर्थ",
    "meaning": "अर्थ",
    "mean": "अर्थ",
    "means": "अर्थ",
    "meanning": "अर्थ",
    "matlab": "अर्थ",
    "definition": "व्याख्या",
    "summary": "सारांश",
    "sumary": "सारांश",
    "saransh": "सारांश",
    "sarvansh": "सारांश",
    "moral": "बोध",
    "message": "शिकवण",
    "bodh": "बोध",
    "shikvan": "शिकवण",
    "character": "पात्र",
    "characters": "पात्रे",
    "patra": "पात्र",
    "patre": "पात्रे",
    "theme": "मध्यवर्ती कल्पना",
    "central idea": "मध्यवर्ती कल्पना",
    "madhyavarti": "मध्यवर्ती",
    "kalpana": "कल्पना",
    # Textbook structure
    "bhumigat": "भूमिगत",
    "madhya": "मध्य",
    "ratra": "रात्र",
    "ratri": "रात्री",
    "aai": "आई",
    "aais": "आईस",
    "baba": "बाबा",
    "ajoba": "आजोबा",
    "ajobanche": "आजोबांचे",
    "kavita": "कविता",
    "kavitet": "कवितेत",
    "kaviteche": "कवितेचे",
    "kavitecha": "कवितेचा",
    "poem": "कविता",
    "poems": "कविता",
    "path": "पाठ",
    "patha": "पाठ",
    "pathat": "पाठात",
    "pathache": "पाठाचे",
    "pathacha": "पाठाचा",
    "lesson": "पाठ",
    "lessons": "पाठ",
    "chapter": "पाठ",
    "chapters": "पाठ",
    "dhada": "पाठ",
    "dhadya": "पाठ",
    "dhadyacha": "पाठाचा",
    "dhadyache": "पाठाचे",
    "mhnje": "म्हणजे",
    "mhanje": "म्हणजे",
    "mhnaje": "म्हणजे",
    "mhtoy": "म्हणतो",
    "nondvun": "नोंदवून",
    "sagle": "सगळे",
    "sarva": "सर्व",
    "all": "सर्व",
    "mala": "मला",
    "dinank": "दिनांक",
    "mahiti": "माहिती",
    "information": "माहिती",
    "mother": "आई",
    "father": "बाबा",
    "grandfather": "आजोबा",
    "grandmother": "आजी",
    "answer": "उत्तर",
    "answers": "उत्तरे",
    "question": "प्रश्न",
    "questions": "प्रश्न",
    "story": "गोष्ट",
    "stories": "गोष्टी",
    "letter": "पत्र",
    "date": "दिनांक",
    "name": "नाव",
    "names": "नावे",
    "title": "शीर्षक",
    "va": "व",
    "ani": "आणि",
    "and": "आणि",
    "of": "चे",
    "in": "मध्ये",
    "cha": "चा",
    "chi": "ची",
    "che": "चे",
    "la": "ला",
    "madhye": "मध्ये",
    "madhe": "मध्ये",
    "baddal": "बद्दल",
    "vishayi": "विषयी",
    "about": "बद्दल",
}

def call_google(text: str) -> str | None:
    try:
        encoded = urllib.parse.quote(text)
        url = f"https://inputtools.google.com/request?text={encoded}&itc=mr-t-i0-und&num=5"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data[1][0][1]
            return candidates[0] if candidates else None
    except Exception:
        return None

def robust_transliterate(query: str) -> str:
    stripped = query.strip()
    words = stripped.split()
    if not words:
        return query

    result_words = []
    for word in words:
        punct = ""
        clean = word
        if clean and clean[-1] in "?.,!;:":
            punct = clean[-1]
            clean = clean[:-1]
        lower = clean.lower()

        # 1. Direct dictionary match for concept/grammatical word
        if lower in _COMMON_WORDS:
            result_words.append(_COMMON_WORDS[lower] + punct)
            continue

        # 2. Suffix stripping (e.g. matherancha -> matheran + cha)
        matched_suffix = False
        for sfx, sfx_dev in [("che", "चे"), ("cha", "चा"), ("chi", "ची"), ("la", "ला"), ("t", "त")]:
            if lower.endswith(sfx) and len(lower) > len(sfx) + 2:
                base = lower[:-len(sfx)]
                if base in _COMMON_WORDS:
                    result_words.append(_COMMON_WORDS[base] + sfx_dev + punct)
                    matched_suffix = True
                    break
        if matched_suffix:
            continue

        # 3. If already Devanagari
        if re.search(r"[\u0900-\u097F]", clean):
            result_words.append(clean + punct)
            continue

        # 4. Transliterate via Google Input Tools
        got = call_google(clean)
        if got:
            result_words.append(got + punct)
        else:
            result_words.append(word)

    return " ".join(result_words)

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    test_queries = [
        "matheran author",
        "matheran auther",
        "who is writer of matheran",
        "saamsum meaning",
        "saamsum hone hych artha kay",
        "poet of rang mazha vegla",
        "ajobanche patra summary",
        "what is moral of story",
        "lekakh kon ahe",
        "nisargaramya matheran lekakh kon ahe",
    ]
    for q in test_queries:
        print(f"'{q}' -> '{robust_transliterate(q)}'")

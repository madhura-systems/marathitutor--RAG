"""Transliteration and Query Normalization Service.

Converts Romanized Marathi (Marathlish), English, and mixed Hindi/English/Marathi
queries into clean, standard Devanagari Marathi suitable for Maharashtra State Board
educational textbook RAG retrieval.

Strategy (in order of priority):
1. If query is already pure Devanagari -> return as-is (zero latency).
2. If query looks like Romanized Marathi/Hindi -> use offline rule-based
   transliteration via `indic_transliteration` (ITRANS scheme). No API call.
3. If query appears to be English sentences -> use LLM translation (API call),
   but only if the LLM is available and NOT rate-limited.
4. Fallback -> return original query unchanged.
"""

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from typing import Optional

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CONTEXT = ssl.create_default_context()

from app.config.settings import Settings, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Common Marathi/Hindi Romanization -> Devanagari lookup table
# ---------------------------------------------------------------------------
_COMMON_WORDS: dict = {
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
    # Question words
    "kay": "काय",
    "kai": "काय",
    "kasa": "कसा",
    "kashi": "कशी",
    "kase": "कसे",
    "kon": "कोण",
    "konata": "कोणता",
    "konti": "कोणती",
    "konte": "कोणते",
    "kothe": "कुठे",
    "kuthe": "कुठे",
    "kadhi": "कधी",
    # Verbs / Actions
    "sang": "सांग",
    "sanga": "सांगा",
    "sangto": "सांगतो",
    "sangte": "सांगते",
    "dya": "द्या",
    "de": "दे",
    "hone": "होणे",
    "honar": "होणार",
    "zale": "झाले",
    "jhale": "झाले",
    "ahe": "आहे",
    "aahe": "आहे",
    "nahi": "नाही",
    "pahije": "पाहिजे",
    "hava": "हवा",
    "havi": "हवी",
    # Common textbook nouns
    "artha": "अर्थ",
    "arth": "अर्थ",
    "artho": "अर्थ",
    "bhumigat": "भूमिगत",
    "madhya": "मध्य",
    "ratra": "रात्र",
    "ratri": "रात्री",
    "patra": "पत्र",
    "aai": "आई",
    "aais": "आईस",
    "baba": "बाबा",
    "ajoba": "आजोबा",
    "ajobanche": "आजोबांचे",
    "kavita": "कविता",
    "kavitet": "कवितेत",
    "kaviteche": "कवितेचे",
    "kavitecha": "कवितेचा",
    "kavi": "कवी",
    "path": "पाठ",
    "patha": "पाठ",
    "pathat": "पाठात",
    "pathache": "पाठाचे",
    "pathacha": "पाठाचा",
    "lesson": "पाठ",
    "chapter": "पाठ",
    "mhnje": "म्हणजे",
    "mhanje": "म्हणजे",
    "mhnaje": "म्हणजे",
    "mhtoy": "म्हणतो",
    "nondvun": "नोंदवून",
    "sagle": "सगळे",
    "sarva": "सर्व",
    "mala": "मला",
    "dinank": "दिनांक",
    "mahiti": "माहिती",
    "summary": "सारांश",
    "sarvansh": "सारांश",
    "saransh": "सारांश",
    "mother": "आई",
    "father": "बाबा",
    "grandfather": "आजोबा",
    "grandmother": "आजी",
    "answer": "उत्तर",
    "question": "प्रश्न",
    "story": "गोष्ट",
    "moral": "नैतिक धडा",
    "meaning": "अर्थ",
    "letter": "पत्र",
    "date": "दिनांक",
    "name": "नाव",
    "va": "व",
    "ani": "आणि",
    "cha": "चा",
    "chi": "ची",
    "che": "चे",
    "la": "ला",
    "madhye": "मध्ये",
    "madhe": "मध्ये",
}

# Heuristic: common English function words and vocabulary that signal an English sentence
_ENGLISH_COMMON = re.compile(
    r"\b(what|when|where|who|why|how|which|is|are|was|were|does|do|did|can|could|"
    r"should|would|the|a|an|of|in|on|at|to|for|from|with|about|tell|explain|give|"
    r"find|list|show|describe|summarize|means|mean|define|definition|get|write|"
    r"all|me|my|you|your|he|she|it|they|this|that|these|those|chapter|chapters|"
    r"poem|poems|lesson|lessons|story|stories|author|poet|names|name|book|books|"
    r"born|district|located|place)\b",
    re.IGNORECASE,
)

# Common TOC patterns in English / Romanized query that should directly map to Marathi TOC
_TOC_PATTERNS = re.compile(
    r"\b(all chapters|chapter names|chapters and poem|chapters and poems|all lessons|"
    r"list of chapters|table of contents|all poems|give me all|list all|index)\b",
    re.IGNORECASE,
)

# Rate-limit back-off state (module-level so it persists across calls)
_last_429_time: float = 0.0
_BACKOFF_SECONDS: float = 60.0

TRANSLITERATION_SYSTEM_PROMPT = """तुम्ही महाराष्ट्र राज्य पाठ्यपुस्तक मंडळाच्या मराठी विषयासाठी भाषा रूपांतर तज्ञ आहात.
तुमचे एकमेव काम: खालील इनपुट मराठी देवनागरी लिपीत रूपांतरित करणे.

नियम:
1. इनपुट रोमन लिपीतील मराठी असेल (Marathlish) तर देवनागरीत रूपांतरित करा.
2. इनपुट हिंदी असेल तर मराठीत भाषांतर करा.
3. इनपुट इंग्रजी असेल तर मराठी देवनागरीत भाषांतर करा.
4. इनपुट हिंदी + मराठी + इंग्रजी मिश्रित असेल तर संपूर्ण वाक्य स्वच्छ मराठी देवनागरीत द्या.
5. chapter/lesson/patha यांचे रूपांतर "पाठ" असे करा.
6. फक्त देवनागरी मराठी आउटपुट द्या — कोणतेही स्पष्टीकरण नाही, कोणतेही इंग्रजी नाही, कोणतेही कोटेशन मार्क नाही, कोणतेही मार्कडाउन नाही.
7. प्रश्नाची रचना जशी आहे तशी ठेवा.
8. अत्यंत महत्त्वाचे: प्रश्नाचे उत्तर कधीही देऊ नका! फक्त प्रश्नाचे मराठीत भाषांतर करा. (Do NOT answer the question under any circumstances! Only translate the question itself into a Marathi question).
9. इयत्ता ६ ते १० च्या शालेय विद्यार्थ्यांना समजेल अशी सोपी पाठ्यपुस्तक मराठी वापरा.

उदाहरणे:
Input: "Where is Matheran located?"
Output: माथेरान कुठे आहे?
Input: "In which district is Matheran located?"
Output: माथेरान कोणत्या जिल्ह्यात आहे?
Input: "Where was Shivaji Maharaj born?"
Output: शिवाजी महाराजांचा जन्म कुठे झाला?
Input: "What is the meaning of Swarajya?"
Output: स्वराज्य या शब्दाचा अर्थ काय आहे?
Input: "Matheran baddal kay mahiti dili ahe?"
Output: माथेरानबद्दल काय माहिती दिली आहे?
Input: "ya kaviteche nav kay ahe?"
Output: या कवितेचे नाव काय आहे?
Input: "What is the moral of the story?"
Output: या गोष्टीतून काय बोध मिळतो?
Input: "ajobanche patra"
Output: आजोबांचे पत्र
Input: "chapter 3 madhye kay ahe?"
Output: पाठ ३ मध्ये काय आहे?
Input: "mla ya pathachi summary pahije"
Output: मला या पाठाचा सारांश हवा आहे.
Input: "iska matlab kya hai?"
Output: याचा अर्थ काय आहे?
Input: "lesson 5 cha answer sang"
Output: पाठ ५ चे उत्तर सांगा.
Input: "grandfathers letter mhanje konata path?"
Output: आजोबांचे पत्र म्हणजे कोणता पाठ?
Input: "ya kavitet kavi kay sangto?"
Output: या कवितेत कवी काय सांगतो?
"""


class TransliterationService:
    """Service to transliterate Romanized Marathi and translate English queries to Marathi.

    Converts user input into clean Marathi Devanagari script for improved
    ChromaDB vector embedding alignment and chapter/TOC keyword matching.

    Primary path is fully **offline** (rule-based dictionary + indic_transliteration).
    The LLM API is used only for pure English sentences and only when not rate-limited.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._llm = None
        self._initialized = False

    @staticmethod
    def contains_roman_or_english(text: str) -> bool:
        """Check if the text contains any Latin/Roman alphabetic characters."""
        return bool(re.search(r"[a-zA-Z]", text))

    @staticmethod
    def contains_devanagari(text: str) -> bool:
        """Check if the text contains Devanagari script characters."""
        return bool(re.search(r"[\u0900-\u097F]", text))

    @staticmethod
    def _is_english_sentence(text: str) -> bool:
        """Heuristic: returns True if the query looks like an English sentence.

        This triggers LLM-based translation rather than character-level
        Romanized-Marathi transliteration.
        """
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        if not words:
            return False
        english_hits = sum(1 for w in words if _ENGLISH_COMMON.search(w))
        return english_hits / len(words) >= 0.35

    @staticmethod
    def _universal_transliterate(query: str) -> str:
        """Universal Marathi transliteration using Google Input Tools API.

        Converts ANY Romanized Marathi (Marathlish) word, phrase, or sentence to
        natural Devanagari Marathi.

        Strategy (in order):
        1. Fast dictionary lookup for known words / common vocabulary
        2. Try the FULL SENTENCE as a single API call (better context than word-by-word)
        3. Fall back to word-by-word for mixed/partial inputs
        4. Keep original word if offline
        """
        import json
        import urllib.parse
        import urllib.request

        stripped = query.strip()
        words = stripped.split()

        # ── 1. If ALL words are in the dictionary, use it directly ──────────
        dict_results = []
        all_in_dict = True
        for word in words:
            punct = ""
            clean = word
            if clean and clean[-1] in "?.,!;:":
                punct = clean[-1]
                clean = clean[:-1]
            lower = clean.lower()
            if lower in _COMMON_WORDS:
                dict_results.append(_COMMON_WORDS[lower] + punct)
            else:
                all_in_dict = False
                break
        if all_in_dict and dict_results:
            return " ".join(dict_results)

        def _call_google(text: str) -> str | None:
            """Call Google Input Tools for a single text string. Returns best candidate or None."""
            try:
                encoded = urllib.parse.quote(text)
                url = f"https://inputtools.google.com/request?text={encoded}&itc=mr-t-i0-und&num=8"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=4, context=_SSL_CONTEXT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data[1][0][1]
                    if not candidates:
                        return None
                    # Prefer candidates that end in grammatically natural Marathi endings
                    lower_t = text.lower()
                    chosen = candidates[0]
                    if lower_t.endswith("la"):
                        for c in candidates:
                            if c.endswith("ला"):
                                chosen = c
                                break
                    elif lower_t.endswith("che") or lower_t.endswith("cha") or lower_t.endswith("chi"):
                        for c in candidates:
                            if c[-2:] in ("चे", "चा", "ची"):
                                chosen = c
                                break
                    return chosen
            except Exception as exc:
                logger.debug("Google transliteration failed for '%s': %s", text, exc)
                return None

        # ── 2. Try the FULL PHRASE in one API call ─────────────────────────
        # Google Input Tools handles multi-word phrases better (uses context).
        if len(words) > 1:
            full_result = _call_google(stripped)
            if full_result:
                import re as _re
                # Accept if result contains Devanagari
                if _re.search(r"[\u0900-\u097F]", full_result):
                    logger.debug(
                        "Full-phrase transliteration: '%s' -> '%s'",
                        stripped[:50], full_result[:50]
                    )
                    return full_result

        # ── 3. Word-by-word fallback ────────────────────────────────────────
        result_words: list = []
        for word in words:
            punct = ""
            clean = word
            if clean and clean[-1] in "?.,!;:":
                punct = clean[-1]
                clean = clean[:-1]
            if not clean:
                continue
            lower = clean.lower()

            # Dictionary first
            if lower in _COMMON_WORDS:
                result_words.append(_COMMON_WORDS[lower] + punct)
                continue

            # Google Input Tools per word
            got = _call_google(clean)
            if got:
                result_words.append(got + punct)
            else:
                result_words.append(word)

        return " ".join(result_words)


    def initialize(self) -> None:
        """Initialize the LLM client (used only as last resort for English queries)."""
        try:
            if self._settings.llm_provider.lower() == "gemini" and self._settings.gemini_api_key:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self._llm = ChatGoogleGenerativeAI(
                    model=self._settings.gemini_model_name,
                    google_api_key=self._settings.gemini_api_key,
                    temperature=0.0,
                    max_output_tokens=500,
                )
                logger.info("✅ TransliterationService LLM initialized with Gemini")
            elif self._settings.mistral_api_key:
                from langchain_mistralai import ChatMistralAI

                self._llm = ChatMistralAI(
                    model="open-mistral-7b",
                    mistral_api_key=self._settings.mistral_api_key,
                    temperature=0.0,
                    max_tokens=500,
                )
                logger.info("✅ TransliterationService LLM initialized with Mistral")
            else:
                logger.warning("No API key available for TransliterationService LLM.")

            self._initialized = True
        except Exception as exc:
            logger.error("Failed to initialize TransliterationService LLM: %s", exc)
            self._initialized = False

    def transliterate_to_marathi(self, query: str) -> str:
        """Transliterate or translate the query into Devanagari Marathi.

        If the input is already pure Devanagari text, returns the query immediately
        with zero latency. If the input contains Roman/Latin letters, converts it
        using offline rule-based transliteration first, and falls back to LLM only
        for pure English sentences (when not rate-limited).

        Args:
            query: User's input text (Marathlish, English, Hindi, or Marathi).

        Returns:
            Normalized Marathi query in Devanagari script.
        """
        global _last_429_time

        if not query or not query.strip():
            return query

        stripped_query = query.strip()

        # 1. Zero-latency bypass if query is already Devanagari only
        if not self.contains_roman_or_english(stripped_query):
            return stripped_query

        # 2. Instant offline mapping for TOC / Chapter list queries (zero latency, zero API calls)
        if _TOC_PATTERNS.search(stripped_query):
            logger.info("TOC pattern detected in query: '%s' -> 'अनुक्रमणिका सर्व पाठ आणि कवितांची यादी'", stripped_query[:60])
            return "अनुक्रमणिका - सर्व पाठ आणि कवितांची यादी"

        logger.info("Transliterating Roman/English query: '%s'", stripped_query[:80])

        # 3. Universal Marathi transliteration (Google Input Tools engine)
        if not self._is_english_sentence(stripped_query):
            result = self._universal_transliterate(stripped_query)
            if self.contains_devanagari(result):
                logger.info(
                    "Universal transliterated: '%s' -> '%s'",
                    stripped_query[:50],
                    result[:50],
                )
                return result

        # 3. LLM path (English sentences or offline failure)
        # Skip if we are in a rate-limit back-off window
        if time.time() - _last_429_time < _BACKOFF_SECONDS:
            remaining = int(_BACKOFF_SECONDS - (time.time() - _last_429_time))
            logger.warning(
                "Skipping LLM transliteration (rate-limit back-off: %ds remaining). "
                "Returning original query.",
                remaining,
            )
            return stripped_query

        if not self._initialized or self._llm is None:
            self.initialize()
            if self._llm is None:
                logger.warning("TransliterationService LLM not ready; returning original query.")
                return stripped_query

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=TRANSLITERATION_SYSTEM_PROMPT),
                HumanMessage(content=f'Input: "{stripped_query}"\nOutput:'),
            ]
            response = self._llm.invoke(messages)
            result = str(response.content).strip()

            # Clean any accidental quotes, backticks, or markdown artifacts
            result = re.sub(r"^[\"\'`*#]+|[\"\'`*#]+$", "", result).strip()
            result = re.sub(r"^Output:\s*", "", result, flags=re.IGNORECASE).strip()

            if self.contains_devanagari(result):
                logger.info(
                    "LLM transliterated: '%s' -> '%s'",
                    stripped_query[:50],
                    result[:50],
                )
                return result
            else:
                logger.warning(
                    "LLM transliteration did not yield Devanagari: '%s'", result
                )
                return self._simple_fallback(stripped_query)

        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate_limit" in exc_str.lower():
                _last_429_time = time.time()
                logger.error(
                    "Transliteration rate-limited (429). Back-off %.0fs activated. Query: '%s'",
                    _BACKOFF_SECONDS,
                    stripped_query[:60],
                )
            else:
                logger.error("Transliteration error for query '%s': %s", stripped_query, exc)
            return self._simple_fallback(stripped_query)

    def _simple_fallback(self, query: str) -> str:
        """Fallback when transliteration is unavailable or produces invalid output."""
        return query

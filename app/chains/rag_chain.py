"""RAG Chain Module.

Builds the complete LCEL (LangChain Expression Language) pipeline:
Question → Retriever → Context → Prompt → Mistral → Parser.
Returns structured results with answer, retrieved chunks, metadata,
and similarity scores.
"""

import re
from enum import Enum
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from app.llm.mistral_service import MistralService
from app.prompts.rag_prompt import PromptService
from app.retrieval.retriever import RetrieverService
from app.utils.logger import get_logger
from app.vectorstore.chroma_service import ChromaService

logger = get_logger(__name__)


class QueryIntent(str, Enum):
    """User intent classification for tailored retrieval and response generation."""
    TOC_OVERVIEW = "TOC_OVERVIEW"
    POEM_EXPLANATION = "POEM_EXPLANATION"
    CHAPTER_SUMMARY = "CHAPTER_SUMMARY"
    WORD_MEANING = "WORD_MEANING"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"


class RetrievedChunk(BaseModel):
    """Represents a single retrieved chunk in the query result.

    Attributes:
        content: The chunk text content.
        page_number: Source page in the textbook.
        chapter: Detected chapter name.
        chunk_id: Unique chunk identifier.
        score: Similarity score from vector search.
    """

    content: str = Field(..., description="Chunk text content")
    page_number: int = Field(default=0, description="Source page number")
    chapter: str = Field(default="unknown", description="Chapter name")
    chunk_id: str = Field(default="", description="Chunk identifier")
    score: float = Field(default=0.0, description="Similarity score")


class QueryResult(BaseModel):
    """Structured result from the RAG chain.

    Attributes:
        question: The original user question.
        answer: The generated answer from the LLM.
        retrieved_chunks: List of chunks used for context.
        page_numbers: Unique page numbers referenced.
        source: Textbook source identifier.
    """

    question: str = Field(..., description="Original user question")
    marathi_question: Optional[str] = Field(
        default=None, description="Transliterated/translated Marathi question in Devanagari"
    )
    answer: str = Field(default="", description="Generated answer")
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list, description="Retrieved chunks with metadata"
    )
    page_numbers: list[int] = Field(
        default_factory=list, description="Referenced page numbers"
    )
    source: str = Field(default="", description="Textbook source identifier")


STANDARD_CHAPTER_MAP: dict[int, dict[int, str]] = {
    6: {
        1: "१. या भारतात बंधुभाव (प्रार्थना) - राष्ट्रसंत श्री तुकडोजी महाराज",
        2: "२. चिमणीचं घरटं - अलका उमरणीकर",
        3: "३. जिकडे तिकडे पाणीच पाणी (कविता) - शंकर वैद्य",
        4: "४. आम्ही महाराष्ट्रकन्या (कविता) - पद्मा गोळे",
        5: "५. आईचे पत्र",
        6: "६. अमुचा बाग विकसनशील छान (कविता) - यशवंत देव",
        7: "७. आजोबांचा तीन पुड्यांचा डबा - अदिती देवधर",
        8: "८. नव्या युगाची गाणी (कविता) - वंदना विटणकर",
        9: "९. निसर्गरम्य माथेरान - सुशील दुधाणे",
        10: "१०. मी मोठा की लहान? - सुनंदा उमर्जी",
        11: "११. गवतफुला रे! गवतफुला! (कविता) - इंदिरा संत",
        12: "१२. जाहिरात",
        13: "१३. गणितातील एक कूटप्रश्न - दिवाकर बापट",
        14: "१४. सुगी (कविता) - संतोष आळंजरकर",
        15: "१५. थोरांची ओळख - डॉ. ए. पी. जे. अब्दुल कलाम",
        16: "१६. मायबोली (कविता) - सुरेश भट",
        17: "१७. आपली संस्कृती, आपल्या परंपरा",
        18: "१८. नव्या तू (कविता) - राजेंद्र आरेकर",
        19: "१९. विभूतींची भंबेरी - प्र. के. अत्रे",
        20: "२०. विचारधन",
    },
    7: {
        1: "१. प्रार्थना - जगदीश खेबुडकर",
        2: "२. श्यामचे बंधुप्रेम - सानेगुरुजी",
        3: "३. माझ्या अंगणात (कविता) - ज्ञानेश्वर कोळी",
        4: "४. गोपाळाचे शौर्य - लक्ष्मीकमल गेडाम",
        5: "५. दादास पत्र",
        6: "६. टप् टप् पडती (कविता) - मंगेश पाडगावकर",
        7: "७. आजारी पडण्याचा प्रयोग - डॉ. मा. मिरासदार",
        8: "८. शब्दांचे घर (कविता) - कल्याण इनामदार",
        9: "९. वाचनाचे वेड - आशा पाटील",
        10: "१०. पंडिता रामाबाई - डॉ. अनुपमा उजगरे",
        11: "११. लेख (कविता) - अस्मिता जोगदंडे-चांदणे",
        12: "१२. रोजनिशी",
        13: "१३. अदलाबदल - पन्नालाल पटेल",
        14: "१४. संतवाणी - संत जनाबाई, संत तुकाराम",
    },
    8: {
        1: "१. भारत अमुचा देश (गीत) - शरद कांबळे",
        2: "२. चिव चिव चिमण्या - विजय तेंडुलकर",
        3: "३. जिकडे तिकडे पाणीच पाणी (कविता) - शंकर वैद्य",
        4: "४. सावलीतून जा आणि सावलीतून ये - सु. ह. जोशी",
        5: "५. विश्वविश्वात शास्त्रज्ञ - स्टीफन हॉकिंग",
        6: "६. कोळ्याची पोर (कविता) - सुरेखा गावडे",
        7: "७. ध्येयपूर्तीचा ध्यास - लक्ष्मण लोंढे",
        8: "८. पाखरांचे मागणे (कविता) - प्रेमचंद अहिराव",
        9: "९. भूमिगत - मुमताज रहिमतपुरे",
        10: "१०. जीवन सुंदर करू! (कविता) - शं. ल. नाईक",
        11: "११. प्राणी आणि आपण - ललितगौरी डांगे",
        12: "१२. संतवाणी - संत एकनाथ, संत श्रीनिवासा",
    },
}

CHAPTER_KEYWORDS: dict[str, list[str]] = {
    # ─── Standard 6 ─────────────────────────────────────────────────────────────
    "१. या भारतात बंधुभाव (प्रार्थना) - राष्ट्रसंत श्री तुकडोजी महाराज": ["बंधुभाव", "तुकडोजी महाराज"],
    "२. चिमणीचं घरटं - अलका उमरणीकर": ["चिमणीचं घरटं", "चिमणी", "अलका उमरणीकर", "ईशा"],
    "३. जिकडे तिकडे पाणीच पाणी (कविता) - शंकर वैद्य": ["पाणीच पाणी", "शंकर वैद्य"],
    "४. आम्ही महाराष्ट्रकन्या (कविता) - पद्मा गोळे": ["महाराष्ट्रकन्या", "पद्मा गोळे"],
    "५. आईचे पत्र": ["आईचे पत्र"],
    "६. अमुचा बाग विकसनशील छान (कविता) - यशवंत देव": ["अमुचा बाग", "यशवंत देव"],
    "७. आजोबांचा तीन पुड्यांचा डबा - अदिती देवधर": ["तीन पुड्यांचा डबा", "आजोबांचा डबा", "अदिती देवधर", "स्टीलचा डबा"],
    "८. नव्या युगाची गाणी (कविता) - वंदना विटणकर": ["नव्या युगाची गाणी", "वंदना विटणकर"],
    "९. निसर्गरम्य माथेरान - सुशील दुधाणे": ["माथेरान", "सुशील दुधाणे"],
    "१०. मी मोठा की लहान? - सुनंदा उमर्जी": ["मी मोठा की लहान", "सुनंदा उमर्जी"],
    "११. गवतफुला रे! गवतफुला! (कविता) - इंदिरा संत": ["गवतफुला", "इंदिरा संत"],
    "१२. जाहिरात": ["जाहिरात"],
    "१३. गणितातील एक कूटप्रश्न - दिवाकर बापट": ["कूटप्रश्न", "रामानुजन", "दिवाकर बापट"],
    "१४. सुगी (कविता) - संतोष आळंजरकर": ["सुगी", "संतोष आळंजरकर"],
    "१५. थोरांची ओळख - डॉ. ए. पी. जे. अब्दुल कलाम": ["अब्दुल कलाम", "कलाम", "लालबहादूर शास्त्री", "शास्त्री"],
    "१६. मायबोली (कविता) - सुरेश भट": ["मायबोली", "सुरेश भट"],
    "१७. आपली संस्कृती, आपल्या परंपरा": ["आपली संस्कृती", "आपल्या परंपरा"],
    "१८. नव्या तू (कविता) - राजेंद्र आरेकर": ["नव्या तू", "राजेंद्र आरेकर"],
    "१९. विभूतींची भंबेरी - प्र. के. अत्रे": ["विभूतींची भंबेरी", "अत्रे", "प्र. के. अत्रे"],
    "२०. विचारधन": ["विचारधन"],

    # ─── Standard 7 ─────────────────────────────────────────────────────────────
    "१. प्रार्थना - जगदीश खेबुडकर": ["खेबुडकर", "प्रार्थना जगदीश"],
    "२. श्यामचे बंधुप्रेम - सानेगुरुजी": ["श्यामचे बंधुप्रेम", "श्यामाचे बंधुप्रेम", "सानेगुरुजी", "श्याम"],
    "३. माझ्या अंगणात (कविता) - ज्ञानेश्वर कोळी": ["माझ्या अंगणात", "ज्ञानेश्वर कोळी"],
    "४. गोपाळाचे शौर्य - लक्ष्मीकमल गेडाम": ["गोपाळाचे शौर्य", "लक्ष्मीकमल गेडाम", "लक्ष्मीकांत गेडाम", "गोपाळ"],
    "५. दादास पत्र": ["दादास पत्र", "दादाला पत्र"],
    "६. टप् टप् पडती (कविता) - मंगेश पाडगावकर": ["टप् टप् पडती", "मंगेश पाडगावकर"],
    "७. आजारी पडण्याचा प्रयोग - डॉ. मा. मिरासदार": ["आजारी पडण्याचा प्रयोग", "मिरासदार"],
    "८. शब्दांचे घर (कविता) - कल्याण इनामदार": ["शब्दांचे घर", "कल्याण इनामदार"],
    "९. वाचनाचे वेड - आशा पाटील": ["वाचनाचे वेड", "आशा पाटील", "अशा पाटील"],
    "१०. पंडिता रामाबाई - डॉ. अनुपमा उजगरे": ["पंडिता रामाबाई", "रामाबाई", "अनुपमा उजगरे"],
    "११. लेख (कविता) - अस्मिता जोगदंडे-चांदणे": ["लेख कविता", "अस्मिता जोगदंडे"],
    "१२. रोजनिशी": ["रोजनिशी"],
    "१३. अदलाबदल - पन्नालाल पटेल": ["अदलाबदल", "पन्नालाल पटेल"],
    "१४. संतवाणी - संत जनाबाई, संत तुकाराम": ["संत जनाबाई", "संतवाणी सातवी"],

    # ─── Standard 8 ─────────────────────────────────────────────────────────────
    "१. भारत अमुचा देश (गीत) - शरद कांबळे": ["भारत अमुचा देश", "शरद कांबळे"],
    "२. चिव चिव चिमण्या - विजय तेंडुलकर": ["चिव चिव चिमण्या", "विजय तेंडुलकर", "तेंडुलकर"],
    "३. जिकडे तिकडे पाणीच पाणी (कविता) - शंकर वैद्य": ["पाणीच पाणी", "शंकर वैद्य"],
    "४. सावलीतून जा आणि सावलीतून ये - सु. ह. जोशी": ["सावलीतून जा", "सावलीतून ये", "जोशी सावली"],
    "५. विश्वविश्वात शास्त्रज्ञ - स्टीफन हॉकिंग": ["स्टीफन हॉकिंग", "हॉकिंग", "किशोर पवार"],
    "६. कोळ्याची पोर (कविता) - सुरेखा गावडे": ["कोळ्याची पोर", "सुरेखा गावडे"],
    "७. ध्येयपूर्तीचा ध्यास - लक्ष्मण लोंढे": ["ध्येयपूर्तीचा ध्यास", "लक्ष्मण लोंढे"],
    "८. पाखरांचे मागणे (कविता) - प्रेमचंद अहिराव": ["पाखरांचे मागणे", "प्रेमचंद अहिराव"],
    "९. भूमिगत - मुमताज रहिमतपुरे": ["भूमिगत", "मुमताज रहिमतपुरे", "शंकर", "अब्दुल", "जेबून", "जैबुन"],
    "१०. जीवन सुंदर करू! (कविता) - शं. ल. नाईक": ["जीवन सुंदर करू", "नाईक जीवन"],
    "११. प्राणी आणि आपण - ललितगौरी डांगे": ["प्राणी आणि आपण", "ललितगौरी डांगे"],
    "१२. संतवाणी - संत एकनाथ, संत श्रीनिवासा": ["संत एकनाथ", "संत श्रीनिवासा", "श्रीनिलोबाराय"],

    # ─── Standard 9 ─────────────────────────────────────────────────────────────
    "१. सर्वात्मका शिवसुंदरा (प्रार्थना) - कुसुमाग्रज": ["सर्वात्मका शिवसुंदरा", "कुसुमाग्रज प्रार्थना"],
    "२. संतवाणी - भेटीलागी जीवा (संत तुकाराम), संतकृपा झाली (संत बहिणाबाई)": ["भेटीलागी जीवा", "संतकृपा झाली", "बहिणाबाई"],
    "३. 'बेटा, मी एकटो आहे!' - वि.पु. काळे": ["बेटा मी एकटो आहे", "एकटो आहे", "वि.पु. काळे", "विपु काळे"],
    "४. जि. आय. पी. रेल्वे - प्रबोधनकार ठाकरे": ["जि. आय. पी. रेल्वे", "जीआयपी रेल्वे", "प्रबोधनकार ठाकरे"],
    "५. रंग माझा वेगळा (कविता) - इंदिरा संत": ["रंग माझा वेगळा", "इंदिरा संत"],
    "६. त्याचे जगणे - मुक्ता": ["त्याचे जगणे", "मुक्ता जगणे"],
    "७. मनाचे श्लोक (कविता) - समर्थ रामदास": ["मनाचे श्लोक", "समर्थ रामदास"],
    "८. कुसुमाग्रज - महाकवी वि. वा. शिरवाडकर": ["कुसुमाग्रज शिरवाडकर", "विष्णू वामन शिरवाडकर"],
    "९. आभाळातल्या पाऊलवाटा - जयंत विद्वांस": ["आभाळातल्या पाऊलवाटा", "जयंत विद्वांस"],
    "१०. वीणा (कविता) - ना. धो. महानोर": ["वीणा कविता", "महानोर"],
    "११. 'वसुधैव कुटुंबकम्' - रमेश वरखेडे": ["वसुधैव कुटुंबकम्", "रमेश वरखेडे"],
    "१२. वाट पाहताना (कविता) - बा. सी. मर्ढेकर": ["वाट पाहताना", "मर्ढेकर"],
    "१३. दिव्याची ज्योत - सुधा मूर्ती": ["दिव्याची ज्योत", "सुधा मूर्ती"],
    "१४. ते जीवनदायी झाड - विनायक रानडे": ["ते जीवनदायी झाड", "जीवनदायी झाड", "विनायक रानडे"],
    "१५. माझे शिक्षक व संस्कार - रा. ना. चव्हाण": ["माझे शिक्षक व संस्कार", "चव्हाण शिक्षक"],
}


TOC_KEYWORDS: list[str] = [
    "अनुक्रमणिका", "विषयसूची", "विषय सूची", "सर्व पाठ", "सर्व कविता", "सर्व धडे",
    "धड्यांची यादी", "पाठांची यादी", "कवितांची यादी", "सर्व विषय", "कोणकोणते धडे",
    "कोणकोणते पाठ", "कोणत्या कविता", "१ ते २०", "१ ते 20", "1 ते 20", "1 to 20",
    "पुस्तकातील सर्व", "पाठ्यपुस्तकातील सर्व", "धडे आणि कविता", "पाठ आणि कविता",
    "table of contents", "chapters list", "all lessons", "all chapters", "all poems",
    "chapter names", "chapter name", "chapters and poem", "chapters and poems",
    "give me all", "list all chapters", "list of chapters", "list of lessons",
    "all chapter names", "chapters in", "index",
    "प्रकरणांची यादी", "सर्व धड्यांची", "सर्व पाठांची", "सर्व धडे", "सर्व कविता",
]


class RAGChain:
    """Production-grade RAG pipeline implementing LCEL with strict grounding."""

    def __init__(
        self,
        retriever_service: RetrieverService,
        prompt_service: PromptService,
        mistral_service: MistralService,
        chroma_service: ChromaService,
        transliteration_service: Optional[Any] = None,
    ) -> None:
        self._retriever_service = retriever_service
        self._prompt_service = prompt_service
        self._mistral_service = mistral_service
        self._chroma_service = chroma_service
        self._transliteration_service = transliteration_service

    def _is_toc_query(self, query: str) -> bool:
        """Check if query is asking for Table of Contents, list of chapters/lessons/poems."""
        cleaned = re.sub(r"[‘'“”\"’]", "", query).strip().lower()
        return any(kw in cleaned for kw in TOC_KEYWORDS)

    def _extract_target_word(self, query: str) -> str:
        """Extract the target word or phrase from a vocabulary query."""
        q_clean = re.sub(r"[‘'“”\"’?!.,;:]", "", query).strip()
        m = re.search(
            r"([\u0900-\u097F\w\s]+?)\s*(?:या\s+)?(?:शब्दाचा|वाक्प्रचाराचा|शब्दांचे|शब्दांचा|वाक्प्रचारांचे|वाक्प्रचारांचा)\s+(?:अर्थ|काय\s+अर्थ|पाठातील\s+संदर्भावरून\s+अर्थ|संदर्भावरून\s+अर्थ)",
            q_clean,
        )
        if m:
            return m.group(1).strip()
        m = re.search(
            r"([\u0900-\u097F\w\s]+?)\s*(?:या\s+)?(?:शब्दाचा|वाक्प्रचाराचा)\s*(?:म्हणजे|काय)",
            q_clean,
        )
        if m:
            return m.group(1).strip()
        m = re.search(
            r"([\u0900-\u097F\w\s]+?)\s*म्हणजे\s+काय",
            q_clean,
        )
        if m:
            return m.group(1).strip()
        m = re.search(
            r"([\u0900-\u097F\w\s]+?)\s*(?:चा|ची|चे)\s+अर्थ",
            q_clean,
        )
        if m:
            return m.group(1).strip()
        return q_clean

    def _detect_intent(self, query: str, marathi_q: str) -> tuple[QueryIntent, Optional[str]]:
        """Classify query into one of 5 distinct intents and extract target entity if applicable."""
        # 1. Table of contents / index overview
        if self._is_toc_query(query) or self._is_toc_query(marathi_q):
            return QueryIntent.TOC_OVERVIEW, None

        combined = f"{query} {marathi_q}".lower()

        # 2. Poem explanation
        POEM_KEYWORDS = ["भावार्थ", "कवितेचा अर्थ", "कडवे", "मध्यवर्ती कल्पना", "कवितेचा भावार्थ", "कवितेचे स्पष्टीकरण"]
        if any(kw in combined for kw in POEM_KEYWORDS):
            return QueryIntent.POEM_EXPLANATION, None

        # 3. Chapter summary
        SUMMARY_KEYWORDS = ["सारांश", "सार", "गोष्ट", "पाठाचा सारांश", "धड्याचा सारांश", "summary"]
        if any(kw in combined for kw in SUMMARY_KEYWORDS):
            return QueryIntent.CHAPTER_SUMMARY, None

        # 4. Word / vocabulary meaning
        WORD_KEYWORDS = [
            "शब्दाचा अर्थ", "वाक्प्रचाराचा अर्थ", "शब्दाचा", "वाक्प्रचाराचा",
            "म्हणजे काय", "अर्थ संदर्भावरून", "अर्थ काय", "चा अर्थ", "ची अर्थ",
            "चे अर्थ", "meaning of", "word meaning", "शब्दार्थ"
        ]
        if any(kw in combined for kw in WORD_KEYWORDS):
            target_word = self._extract_target_word(marathi_q)
            return QueryIntent.WORD_MEANING, target_word

        # 5. Default: Question Answering
        return QueryIntent.QUESTION_ANSWERING, None

    def _detect_chapter(self, query: str, standard: Optional[int] = None) -> Optional[str]:
        """Detect if the query specifically targets a known chapter or lesson number."""
        if self._is_toc_query(query):
            return None
        cleaned_q = re.sub(r"[‘'“”\"’]", "", query)

        # 1. Lesson number pattern: e.g. "पाठ २", "धडा ४", "प्रकरण ९", "chapter 5", "lesson 2"
        # Support both Devanagari and Latin digits
        num_match = re.search(
            r"(?:पाठ|धडा|धड्याचा|पाठाचा|प्रकरण|chapter|lesson)\s*([०-९0-9]+)",
            cleaned_q,
            re.IGNORECASE,
        )
        if num_match:
            digit_str = num_match.group(1)
            marathi_digits = str.maketrans("०१२३४५६७८९", "0123456789")
            try:
                lesson_num = int(digit_str.translate(marathi_digits))
                if standard and standard in STANDARD_CHAPTER_MAP:
                    ch = STANDARD_CHAPTER_MAP[standard].get(lesson_num)
                    if ch:
                        return ch
            except ValueError:
                pass

        # 2. Match known chapter keywords
        for chapter, keywords in CHAPTER_KEYWORDS.items():
            for kw in keywords:
                if kw in cleaned_q:
                    return chapter
        return None

    def invoke(
        self,
        question: str,
        standard: Optional[int | str] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> QueryResult:
        """Execute the full RAG pipeline for a question.

        Args:
            question: User question in Marathi, Romanized Marathi, or English.
            standard: Grade/standard (6, 7, 8, 9, 10 or 'all'/None).
            filters: Optional metadata filters for retrieval.

        Returns:
            ``QueryResult`` with answer, chunks, metadata, and scores.
        """
        # Step 0: Transliterate / normalize Romanized Marathi or English input to Devanagari
        marathi_q = question
        if self._transliteration_service:
            marathi_q = self._transliteration_service.transliterate_to_marathi(question)

        logger.info(
            "RAG chain invoked: [bold]%s[/bold] (Marathi: %s, Standard: %s)",
            question[:80],
            marathi_q[:80] if marathi_q != question else "same",
            str(standard or "all"),
            extra={"markup": True},
        )

        cleaned_q = re.sub(r"[‘'“”\"’]", "", marathi_q)

        # Parse standard integer filter
        std_int: Optional[int] = None
        std_filter: dict[str, Any] = {}
        if standard is not None and str(standard).strip().lower() not in ("all", "0", ""):
            try:
                std_int = int(standard)
                std_filter["standard"] = std_int
            except ValueError:
                pass

        general_std_filter = dict(filters or {})
        if std_filter:
            general_std_filter.update(std_filter)

        # Step 1: Detect intent and target entity
        intent, target_entity = self._detect_intent(question, marathi_q)
        detected_chapter = self._detect_chapter(cleaned_q, std_int)

        logger.info(
            "Query Intent: [bold cyan]%s[/bold cyan] (Target: %s, Detected Chapter: %s)",
            intent.value,
            target_entity,
            detected_chapter,
            extra={"markup": True},
        )

        search_results: list[Any] = []
        llm_question = cleaned_q

        # ── ROUTE 1: TOC / Overview ───────────────────────────────────────────
        if intent == QueryIntent.TOC_OVERVIEW:
            logger.info("Handling TOC_OVERVIEW retrieval")
            cleaned_q_toc = "अनुक्रमणिका - सर्व पाठ आणि कवितांची यादी"
            toc_results = self._chroma_service.search(
                query="अनुक्रमणिका सर्व पाठ आणि कवितांची यादी सूची भाग १ भाग २",
                filter_dict=general_std_filter if general_std_filter else None,
            )
            p10_results = self._chroma_service.search(
                query="अनुक्रमणिका",
                filter_dict=general_std_filter if general_std_filter else None,
            )
            general_results = self._chroma_service.search(
                query=cleaned_q_toc,
                filter_dict=general_std_filter if general_std_filter else None,
            )
            seen_ids = set()
            for r in (toc_results + p10_results + general_results):
                cid = r.document.metadata.get("chunk_id", r.document.page_content[:50])
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    search_results.append(r)
            search_results = search_results[:10]
            prompt_intent = "QA"

        # ── ROUTE 2: Chapter Summary ──────────────────────────────────────────
        elif intent == QueryIntent.CHAPTER_SUMMARY:
            logger.info("Handling CHAPTER_SUMMARY retrieval for '%s'", detected_chapter)
            if detected_chapter:
                chapter_docs = self._chroma_service.get_chapter_documents(
                    chapter_name=detected_chapter,
                    standard=std_int,
                )
                search_results = chapter_docs[:16]  # Full sequential chapter context
            if not search_results:
                search_results = self._chroma_service.search(
                    query=cleaned_q,
                    filter_dict=general_std_filter if general_std_filter else None,
                )[:10]
            prompt_intent = "CHAPTER_SUMMARY"

        # ── ROUTE 3: Poem Explanation ─────────────────────────────────────────
        elif intent == QueryIntent.POEM_EXPLANATION:
            logger.info("Handling POEM_EXPLANATION retrieval for '%s'", detected_chapter)
            if detected_chapter:
                chapter_docs = self._chroma_service.get_chapter_documents(
                    chapter_name=detected_chapter,
                    standard=std_int,
                )
                search_results = chapter_docs[:10]  # Full sequential poem stanzas
            if not search_results:
                search_results = self._chroma_service.search(
                    query=cleaned_q,
                    filter_dict=general_std_filter if general_std_filter else None,
                )[:10]
            prompt_intent = "POEM_EXPLANATION"

        # ── ROUTE 4: Universal Word Meaning ───────────────────────────────────
        elif intent == QueryIntent.WORD_MEANING:
            raw_target = target_entity or cleaned_q
            clean_word = re.sub(
                r"^(?:या\s+|हा\s+|हे\s+|ती\s+)|(?:\s+या)$", "", raw_target
            ).strip()
            logger.info(
                "Handling WORD_MEANING retrieval for word='%s' (std=%s, chapter=%s)",
                clean_word, std_int, detected_chapter
            )

            # 1. Prioritize explicit glossary hits & in-context sentences
            word_hits = self._chroma_service.search_word_context(
                word=clean_word,
                standard=std_int,
                chapter=detected_chapter,
            )

            # 2. Semantic fallback search
            semantic_hits = self._chroma_service.search(
                query=f"{clean_word} शब्दार्थ नवीन शब्द अर्थ",
                filter_dict=general_std_filter if general_std_filter else None,
            )

            seen_ids = set()
            for r in (word_hits + semantic_hits):
                cid = r.document.metadata.get("chunk_id", r.document.page_content[:50])
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    search_results.append(r)

            search_results = search_results[:8]
            prompt_intent = "WORD_MEANING"

        # ── ROUTE 5: Standard QA (factual, comprehension, exercises) ──────────
        else:
            logger.info("Handling QUESTION_ANSWERING retrieval")
            _DEVANAGARI_WORD_RE = re.compile(r'[\u0900-\u097F]{4,}')
            _MARATHI_STOPWORDS = {
                'आहे', 'होते', 'होता', 'होती', 'आणि', 'म्हणजे', 'मला', 'आपण',
                'आपले', 'आपल्या', 'यांचे', 'यांचा', 'यांची', 'जरी', 'तरी',
                'असे', 'अशा', 'असेल', 'असतो', 'असते', 'असतात', 'काय', 'कसे',
                'कोण', 'कधी', 'कोठे', 'सांगा', 'लिहा', 'करा', 'त्यांचे',
                'त्यांचा', 'त्यांची', 'त्यांनी', 'त्यांना', 'ह्याचे', 'ह्याचा',
                'याचे', 'याचा', 'याची', 'यांना', 'त्याने', 'नाही', 'नसेल',
                'सर्व', 'सगळे', 'प्रत्येक', 'काही', 'कोणी', 'खूप', 'फार',
                'मध्ये', 'वरून', 'परंतु', 'किंवा', 'तुम्ही', 'तुमचे',
                'त्याला', 'त्याचे', 'त्याची', 'त्याचा',
                'कुठे', 'कोणत्या', 'कोणते', 'कोणता', 'कोणती', 'कोणाचा', 'कोणाची', 'कोणाचे',
                'झाला', 'झाली', 'झाले', 'झालेला', 'झालेली', 'झालेले', 'होणे', 'होणार', 'आहेत',
                'अर्थ', 'शब्दाचा', 'शब्दांचे', 'शब्दांचा', 'शब्दार्थ',
                'वाक्प्रचार', 'वाक्प्रचाराचा', 'वाक्प्रचारांचे', 'वाक्प्रचारांचा',
                'खालील', 'प्रश्नांची', 'उत्तरे', 'उत्तर', 'एका', 'वाक्यात',
            }
            devanagari_keywords = [
                w for w in _DEVANAGARI_WORD_RE.findall(cleaned_q)
                if w not in _MARATHI_STOPWORDS
            ]

            chapter_results = []
            if detected_chapter:
                chapter_filter = dict(filters or {})
                if std_filter:
                    chapter_filter.update(std_filter)
                chapter_filter["chapter"] = detected_chapter
                chapter_results = self._chroma_service.search(
                    query=cleaned_q,
                    filter_dict=chapter_filter,
                )

            general_results = self._chroma_service.search(
                query=cleaned_q,
                filter_dict=general_std_filter if general_std_filter else None,
            )

            strong_kws = []
            weak_kws = []
            if devanagari_keywords:
                keyword_results = self._chroma_service.keyword_search(
                    keywords=devanagari_keywords,
                    filter_dict=general_std_filter if general_std_filter else None,
                )[:5]
                if keyword_results:
                    strong_kws = [r for r in keyword_results if r.score <= 0.2]
                    weak_kws = [r for r in keyword_results if r.score > 0.2]

            seen_ids = set()
            for r in strong_kws + chapter_results + general_results + weak_kws:
                cid = r.document.metadata.get("chunk_id", r.document.page_content[:50])
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    search_results.append(r)

            search_results = search_results[:10]
            prompt_intent = "QA"

        # Step 2: Build RetrievedChunk list and document collection
        retrieved_chunks: list[RetrievedChunk] = []
        documents: list[Document] = []

        for result in search_results:
            doc = result.document
            documents.append(doc)

            chunk = RetrievedChunk(
                content=doc.page_content,
                page_number=doc.metadata.get("page_number", 0),
                chapter=doc.metadata.get("chapter", "unknown"),
                chunk_id=doc.metadata.get("chunk_id", ""),
                score=result.score,
            )
            retrieved_chunks.append(chunk)

        # Step 3: Format context from documents
        context_str = self._prompt_service.format_context(documents)

        # Step 4: Build and invoke the LCEL chain with intent-specialized prompt
        prompt = self._prompt_service.get_prompt(prompt_intent)
        llm = self._mistral_service.get_llm()
        parser = StrOutputParser()

        chain = prompt | llm | parser

        logger.debug("Invoking LLM with %d context chunks (intent=%s)", len(documents), prompt_intent)
        answer = chain.invoke({
            "context": context_str,
            "question": llm_question,
        })

        # Step 5: Collect unique page numbers
        page_numbers = sorted(set(
            chunk.page_number for chunk in retrieved_chunks if chunk.page_number > 0
        ))

        source_std = std_filter.get("standard", "all")
        source_id = (
            f"mh_state_board_marathi_std_{source_std}"
            if source_std != "all"
            else "mh_state_board_marathi_all"
        )

        result = QueryResult(
            question=question,
            marathi_question=marathi_q if marathi_q != question else None,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            page_numbers=page_numbers,
            source=source_id,
        )

        logger.info(
            "RAG chain complete [%s]: %d chunks, pages=%s",
            intent.value,
            len(retrieved_chunks),
            page_numbers,
        )

        return result

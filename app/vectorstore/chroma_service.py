"""ChromaDB Vector Store Service.

Provides a managed wrapper around ChromaDB for storing, searching,
and managing textbook document embeddings. All vector store operations
in the application go through this service.
"""

from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config.settings import Settings, get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.utils.helpers import timer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SearchResult:
    """Represents a single search result with score.

    Attributes:
        document: The retrieved LangChain Document.
        score: Similarity score (lower is more similar for L2 distance).
    """

    def __init__(self, document: Document, score: float) -> None:
        self.document = document
        self.score = score

    @property
    def page_content(self) -> str:
        """Shortcut to the document's text content."""
        return self.document.page_content

    @property
    def metadata(self) -> dict[str, Any]:
        """Shortcut to the document's metadata."""
        return self.document.metadata

    def __repr__(self) -> str:
        preview = self.page_content[:80].replace("\n", " ")
        return f"SearchResult(score={self.score:.4f}, preview='{preview}…')"


class ChromaService:
    """Managed ChromaDB vector store service.

    Wraps ``langchain_chroma.Chroma`` and provides standardized methods
    for CRUD operations on the vector store. Accepts an ``EmbeddingService``
    via dependency injection.

    Args:
        embedding_service: Initialized ``EmbeddingService`` instance.
        settings: Application settings. Injected for testability.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedding_service = embedding_service
        self._vectorstore: Optional[Chroma] = None

    def reset_collection(self) -> None:
        """Reset or delete the existing collection to guarantee a fresh index."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self._settings.chroma_persist_dir))
            try:
                client.delete_collection(self._settings.collection_name)
                logger.info("Deleted existing ChromaDB collection: %s", self._settings.collection_name)
            except Exception:
                pass
            self._vectorstore = None
        except Exception as exc:
            logger.warning("Could not reset ChromaDB collection: %s", exc)

    def create(self) -> Chroma:
        """Create or connect to a persistent ChromaDB collection.

        Returns:
            The ``Chroma`` vector store instance.
        """
        if self._vectorstore is not None:
            logger.debug("ChromaDB collection already created, reusing.")
            return self._vectorstore

        logger.info(
            "Creating ChromaDB collection: [bold]%s[/bold] at [cyan]%s[/cyan]",
            self._settings.collection_name,
            self._settings.chroma_persist_dir,
            extra={"markup": True},
        )

        self._vectorstore = Chroma(
            collection_name=self._settings.collection_name,
            embedding_function=self._embedding_service.get_embeddings(),
            persist_directory=str(self._settings.chroma_persist_dir),
        )

        logger.info("✅ ChromaDB collection ready")
        return self._vectorstore

    @timer
    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents with embeddings to the vector store.

        Args:
            documents: List of LangChain ``Document`` objects to store.

        Returns:
            List of document IDs assigned by ChromaDB.

        Raises:
            RuntimeError: If the collection has not been created.
        """
        store = self._ensure_store()

        logger.info(
            "Adding [cyan]%d documents[/cyan] to ChromaDB...",
            len(documents),
            extra={"markup": True},
        )

        ids = store.add_documents(documents)

        logger.info(
            "✅ Added [green]%d documents[/green] to ChromaDB",
            len(ids),
            extra={"markup": True},
        )
        return ids

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search for similar documents by query text.

        Args:
            query: Search query text.
            k: Number of results to return. Defaults to ``retriever_top_k``.
            filter_dict: Optional metadata filter (e.g., ``{"textbook_id": "..."}``)

        Returns:
            List of ``SearchResult`` objects sorted by relevance.

        Raises:
            RuntimeError: If the collection has not been created.
        """
        store = self._ensure_store()
        top_k = k or self._settings.retriever_top_k

        logger.debug("Searching ChromaDB: query='%s', k=%d, filters=%s", query[:50], top_k, filter_dict)

        kwargs: dict[str, Any] = {"k": top_k}
        if filter_dict:
            cleaned_filters = {k: v for k, v in filter_dict.items() if v is not None}
            if len(cleaned_filters) > 1:
                kwargs["filter"] = {"$and": [{k: v} for k, v in cleaned_filters.items()]}
            elif len(cleaned_filters) == 1:
                kwargs["filter"] = cleaned_filters

        results = store.similarity_search_with_score(query, **kwargs)

        search_results = [
            SearchResult(document=doc, score=score)
            for doc, score in results
        ]

        logger.info(
            "Search returned [cyan]%d results[/cyan]",
            len(search_results),
            extra={"markup": True},
        )

        return search_results

    def keyword_search(
        self,
        keywords: list[str],
        filter_dict: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search for documents containing any of the given keywords (exact substring match).

        This provides a keyword-boost layer on top of semantic similarity search.
        It guarantees that chunks explicitly containing the searched term are always
        included in results, even when cosine similarity ranks them poorly.

        Works for ANY user-entered word — zero hardcoding required.

        Args:
            keywords: List of Devanagari words/substrings to look for.
            filter_dict: Optional metadata filter (e.g., ``{"standard": 8}``).

        Returns:
            List of ``SearchResult`` objects with score=0.0 (exact match, highest priority).
        """
        if not keywords:
            return []

        store = self._ensure_store()
        collection = store._collection

        try:
            # Fetch all docs (with optional standard filter) from ChromaDB
            get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
            if filter_dict:
                cleaned = {k: v for k, v in filter_dict.items() if v is not None}
                if len(cleaned) > 1:
                    get_kwargs["where"] = {"$and": [{k: v} for k, v in cleaned.items()]}
                elif len(cleaned) == 1:
                    get_kwargs["where"] = cleaned

            all_data = collection.get(**get_kwargs)
        except Exception as exc:
            logger.warning("keyword_search: collection.get() failed: %s", exc)
            return []

        raw_docs = all_data.get("documents") or []
        raw_metas = all_data.get("metadatas") or []

        candidates: list[tuple[int, int, SearchResult]] = []
        seen_cids: set[str] = set()

        num_keywords = len(keywords)
        for doc_text, meta in zip(raw_docs, raw_metas):
            matching_kws = [kw for kw in keywords if kw in doc_text]
            if not matching_kws:
                continue

            cid = meta.get("chunk_id", doc_text[:50])
            if cid in seen_cids:
                continue
            seen_cids.add(cid)

            match_count = len(matching_kws)
            total_freq = sum(doc_text.count(kw) for kw in matching_kws)

            # Calibrate score according to keyword coverage:
            # - If all query keywords match, or single target keyword query matches: score 0.05
            # - If multiple keywords matched: score scales with coverage (e.g. 0.1 / match_count)
            # - If weak single match out of multiple keywords: score 0.65
            if match_count == num_keywords or num_keywords == 1:
                score = 0.05
            elif match_count > 1:
                score = round(0.1 / match_count, 3)
            else:
                score = 0.65

            candidates.append((
                match_count,
                total_freq,
                SearchResult(
                    document=Document(page_content=doc_text, metadata=meta),
                    score=score,
                ),
            ))

        # Rank candidates by multi-term match count descending, then total term frequency descending
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        matched = [c[2] for c in candidates]

        logger.info(
            "Keyword search for %s returned [green]%d exact-match chunks[/green]",
            keywords,
            len(matched),
            extra={"markup": True},
        )
        return matched

    def get_chapter_documents(
        self,
        chapter_name: str,
        standard: Optional[int] = None,
    ) -> list[SearchResult]:
        """Retrieve all documents belonging to a chapter, sorted chronologically.

        Extracts all unique chunks for the chapter and orders them by
        (page_number, chunk_id) to reconstruct the full narrative flow
        required for chapter summaries and poem explanations.

        Args:
            chapter_name: Exact chapter name or key phrase to match metadata.
            standard: Optional standard filter (e.g., 6, 7, 8).

        Returns:
            List of SearchResult objects sorted chronologically.
        """
        store = self._ensure_store()
        collection = store._collection

        try:
            where_clauses: list[dict[str, Any]] = [{"chapter": chapter_name}]
            if standard:
                where_clauses.append({"standard": int(standard)})

            where = {"$and": where_clauses} if len(where_clauses) > 1 else where_clauses[0]
            data = collection.get(where=where, include=["documents", "metadatas"])
        except Exception as exc:
            logger.warning("get_chapter_documents failed for '%s': %s", chapter_name, exc)
            return []

        raw_docs = data.get("documents") or []
        raw_metas = data.get("metadatas") or []

        unique_chunks: dict[str, tuple[int, str, str, dict[str, Any]]] = {}
        for doc_text, meta in zip(raw_docs, raw_metas):
            cid = meta.get("chunk_id", doc_text[:50])
            if cid not in unique_chunks:
                pg = meta.get("page_number", 0)
                unique_chunks[cid] = (pg, cid, doc_text, meta)

        sorted_chunks = sorted(unique_chunks.values(), key=lambda x: (x[0], x[1]))
        results = [
            SearchResult(
                document=Document(page_content=doc_text, metadata=meta),
                score=0.0,
            )
            for pg, cid, doc_text, meta in sorted_chunks
        ]
        logger.info(
            "Retrieved %d chronological chunks for chapter '%s' (std=%s)",
            len(results), chapter_name, standard
        )
        return results

    def search_word_context(
        self,
        word: str,
        standard: Optional[int] = None,
        chapter: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search for word meanings, prioritizing glossary sections over chapter context.

        Categorizes matching chunks into:
        1. Explicit Glossary Matches: chunks containing the word alongside
           'शब्दार्थ', 'नवीन शब्द', '–', or '-' (highest priority, score 0.05).
        2. In-Context Chapter Matches: chunks containing the word in sentences (score 0.20).

        Args:
            word: The target Marathi word or phrase.
            standard: Optional standard filter.
            chapter: Optional chapter filter.

        Returns:
            List of SearchResult objects sorted with glossary definitions first.
        """
        if not word or not word.strip():
            return []

        store = self._ensure_store()
        collection = store._collection

        clean_word = word.strip()
        where_clauses: list[dict[str, Any]] = []
        if standard:
            where_clauses.append({"standard": int(standard)})
        if chapter:
            where_clauses.append({"chapter": chapter})

        get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if len(where_clauses) > 1:
            get_kwargs["where"] = {"$and": where_clauses}
        elif len(where_clauses) == 1:
            get_kwargs["where"] = where_clauses[0]

        try:
            data = collection.get(**get_kwargs)
        except Exception as exc:
            logger.warning("search_word_context failed: %s", exc)
            return []

        raw_docs = data.get("documents") or []
        raw_metas = data.get("metadatas") or []

        glossary_results: list[SearchResult] = []
        context_results: list[SearchResult] = []
        seen_cids: set[str] = set()

        _GLOSSARY_INDICATORS = ("शब्दार्थ", "नवीन शब्द", "वाक्प्रचार", "–", "-")
        search_terms = [clean_word]
        if " " in clean_word:
            search_terms.extend([t for t in clean_word.split() if len(t) >= 3])

        for doc_text, meta in zip(raw_docs, raw_metas):
            matched_terms = [t for t in search_terms if t in doc_text]
            if not matched_terms:
                continue
            cid = meta.get("chunk_id", doc_text[:50])
            if cid in seen_cids:
                continue
            seen_cids.add(cid)

            is_glossary = any(ind in doc_text for ind in _GLOSSARY_INDICATORS)
            is_explicit_def = is_glossary and any(
                (f"{t} –" in doc_text or f"{t} -" in doc_text or "शब्दार्थ" in doc_text or "नवीन शब्द" in doc_text)
                for t in matched_terms
            )
            if is_explicit_def:
                glossary_results.append(
                    SearchResult(
                        document=Document(page_content=doc_text, metadata=meta),
                        score=0.05,
                    )
                )
            else:
                context_results.append(
                    SearchResult(
                        document=Document(page_content=doc_text, metadata=meta),
                        score=0.20,
                    )
                )

        # Glossary hits come first, followed by contextual occurrences
        matched = glossary_results + context_results
        logger.info(
            "Word context search for '%s': %d glossary hits, %d contextual hits",
            clean_word, len(glossary_results), len(context_results)
        )
        return matched

    def delete(self, ids: list[str]) -> None:
        """Delete documents from the vector store by ID.

        Args:
            ids: List of document IDs to delete.

        Raises:
            RuntimeError: If the collection has not been created.
        """
        store = self._ensure_store()
        collection = store._collection
        collection.delete(ids=ids)
        logger.info("Deleted %d documents from ChromaDB", len(ids))

    def reset(self) -> None:
        """Delete all documents from the current collection.

        Drops and recreates the collection, effectively clearing all data.

        Raises:
            RuntimeError: If the collection has not been created.
        """
        store = self._ensure_store()
        store.reset_collection()
        logger.warning("⚠️ ChromaDB collection reset — all documents deleted")

    def persist(self) -> None:
        """Persist vector store data to disk.

        Note: ChromaDB with persistent directory auto-persists,
        but this method is kept for explicit lifecycle management.
        """
        if self._vectorstore is not None:
            logger.info("ChromaDB data persisted to %s", self._settings.chroma_persist_dir)

    @property
    def document_count(self) -> int:
        """Return the number of documents currently in the collection."""
        if self._vectorstore is None:
            return 0
        try:
            collection = self._vectorstore._collection
            return collection.count()
        except Exception:
            return 0

    def get_vectorstore(self) -> Chroma:
        """Return the underlying Chroma instance.

        Returns:
            The ``Chroma`` vector store.

        Raises:
            RuntimeError: If the collection has not been created.
        """
        return self._ensure_store()

    def _ensure_store(self) -> Chroma:
        """Ensure the vector store has been created.

        Returns:
            The ``Chroma`` instance.

        Raises:
            RuntimeError: If ``create()`` has not been called.
        """
        if self._vectorstore is None:
            raise RuntimeError(
                "ChromaService not initialized. Call create() first."
            )
        return self._vectorstore

"""Retrieval-Augmented Generation (RAG): ingest documents into ChromaDB and retrieve context.

We wrap ChromaDB in a small ``KnowledgeBase`` class so the rest of the codebase does not need
to know any ChromaDB details — it only calls ``ingest()`` and ``retrieve()``. This is the same
abstraction idea we use for the LLM client in Step 4: hide the third-party tool behind a simple
interface that we control.

Embeddings are produced locally with sentence-transformers (no API cost), and the vectors are
stored in a local, persistent ChromaDB collection.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from src.paths import CHROMA_DIR, KNOWLEDGE_BASE_DIR

logger = logging.getLogger(__name__)

# Lightweight, widely used embedding model (384-dimensional vectors). Good quality-to-size
# ratio and fast enough to run on a laptop / in CI without a GPU.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Name of the ChromaDB collection that holds our knowledge base chunks.
COLLECTION_NAME = "knowledge_base"


@dataclass
class RetrievedChunk:
    """A single chunk returned by a retrieval query.

    Attributes:
        text: The chunk text.
        source: Name of the source document the chunk came from (e.g. "cards_policy.md").
        distance: Vector distance to the query (lower = more similar, cosine distance).
    """

    text: str
    source: str
    distance: float


def _chunk_document(text: str) -> list[str]:
    """Split a markdown document into chunks at heading boundaries.

    Chunking strategy: each markdown section (a heading line starting with '#' together with
    the text below it, up to the next heading) becomes one chunk. We chose heading-based
    ("semantic") chunking rather than fixed-size character/token windows because:

    * our documents are short and well structured, so each section is already a self-contained
      idea (e.g. "Blocking a card");
    * splitting on natural boundaries avoids cutting a sentence in half, which would hurt
      retrieval quality;
    * it is simple to explain and has no magic chunk-size/overlap numbers to tune.

    For long, unstructured documents a fixed-size window with overlap would be the better
    default, but that is overkill here.

    Args:
        text: Raw markdown content of one document.

    Returns:
        A list of non-empty chunk strings.
    """
    chunks: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        # A heading starts a new section — flush whatever we have collected so far.
        if line.startswith("#") and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    # Don't forget the last section after the loop ends.
    if current:
        chunks.append("\n".join(current).strip())

    # Drop any chunks that ended up empty after stripping.
    return [chunk for chunk in chunks if chunk]


class KnowledgeBase:
    """Thin wrapper around a local, persistent ChromaDB vector store.

    Exposes two operations:
        * ``ingest()`` — (re)build the collection from the markdown documents on disk;
        * ``retrieve()`` — find the chunks most relevant to a query.
    """

    def __init__(
        self,
        persist_dir: Path = CHROMA_DIR,
        embedding_model: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        """Initialize the knowledge base.

        Args:
            persist_dir: Directory where ChromaDB stores its data on disk.
            embedding_model: Name of the sentence-transformers model used for embeddings.
        """
        # PersistentClient writes to disk, so the index survives between runs (and process
        # restarts). anonymized_telemetry=False turns off ChromaDB's usage telemetry.
        self._client: ClientAPI = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        # The embedding function turns text into vectors. ChromaDB calls it automatically
        # both when we add documents and when we query, so embedding stays consistent.
        # The cast keeps mypy happy about ChromaDB's broad embedding-function type.
        self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

    def _get_or_create_collection(self) -> Collection:
        """Return the collection, creating it (with cosine distance) if it does not exist."""
        return self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_function,
            # Cosine distance is the standard choice for sentence-embedding similarity.
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR) -> int:
        """Rebuild the collection from all markdown files in a directory.

        We rebuild from scratch on every ingest (delete + recreate) so the stored index always
        matches exactly what is on disk — no stale chunks left over from removed text. For a
        small knowledge base this is simpler and safer than incremental updates.

        Args:
            knowledge_base_dir: Directory containing the .md knowledge base documents.

        Returns:
            The number of chunks that were ingested.

        Raises:
            FileNotFoundError: When the directory contains no .md files.
        """
        # Delete the existing collection if present, so we start clean.
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:  # noqa: BLE001 — collection may simply not exist yet on first run
            logger.debug("No existing collection to delete (first ingest).")

        collection = self._get_or_create_collection()

        md_files = sorted(knowledge_base_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"No .md documents found in {knowledge_base_dir}")

        # ChromaDB's add() takes parallel lists: documents, ids and metadata.
        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, str | int]] = []

        for md_file in md_files:
            text = md_file.read_text(encoding="utf-8")
            for chunk_index, chunk in enumerate(_chunk_document(text)):
                documents.append(chunk)
                # Stable, unique id per chunk, e.g. "cards_policy.md::0".
                ids.append(f"{md_file.name}::{chunk_index}")
                # Metadata lets us trace a chunk back to its source document (used for
                # the faithfulness metric in Step 5).
                metadatas.append({"source": md_file.name, "chunk_index": chunk_index})

        # add() embeds the documents (via our embedding function) and stores the vectors.
        collection.add(documents=documents, ids=ids, metadatas=metadatas)

        logger.info(
            "Ingested %d chunks from %d documents into collection '%s'.",
            len(documents),
            len(md_files),
            COLLECTION_NAME,
        )
        return len(documents)

    def retrieve(self, query: str, n_results: int = 3) -> list[RetrievedChunk]:
        """Find the chunks most relevant to a query.

        Args:
            query: The user question (or any text) to search for.
            n_results: How many top chunks to return.

        Returns:
            A list of RetrievedChunk, ordered from most to least relevant.
        """
        collection = self._get_or_create_collection()

        # query_texts is a list because ChromaDB supports batching many queries at once;
        # we pass a single query, so we read index [0] of each result list below.
        result = collection.query(query_texts=[query], n_results=n_results)

        # ChromaDB returns parallel lists wrapped one level deeper (one entry per query).
        # They can be None if the collection is empty, hence the "or [[]]" guards.
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=False):
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source=str(metadata.get("source", "unknown")),
                    distance=float(distance),
                )
            )
        return chunks
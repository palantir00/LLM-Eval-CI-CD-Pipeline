"""Ingest the sample banking documents into the ChromaDB vector store.

Run:  uv run python -m scripts.load_knowledge_base

After ingesting, the script runs a couple of sample retrievals as a quick smoke test, so you
can see that semantic search actually returns the relevant chunks.
"""

import logging

from src.pipeline.rag import KnowledgeBase

# Logging configuration: we use the logging module instead of print (project requirement).
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# A few example questions used only to demonstrate that retrieval works.
SMOKE_TEST_QUERIES = [
    "How do I block my card?",
    "What is the difference between SEPA and SWIFT?",
    "What documents do I need to open an account?",
]


def main() -> None:
    """Ingest the knowledge base and run sample retrievals as a smoke test."""
    knowledge_base = KnowledgeBase()
    knowledge_base.ingest()

    for query in SMOKE_TEST_QUERIES:
        logger.info("Query: %s", query)
        for chunk in knowledge_base.retrieve(query, n_results=1):
            # First line of the chunk is the markdown heading — handy as a short label.
            heading = chunk.text.splitlines()[0]
            logger.info("  -> [%s] %s (distance=%.3f)", chunk.source, heading, chunk.distance)


if __name__ == "__main__":
    main()
"""Prompt templates for the RAG pipeline.

Keeping prompts in one place (instead of scattered string literals) makes them easy to review,
version and tweak — and changing a prompt is exactly the kind of change this whole project is
designed to evaluate.
"""

# System prompt: sets the assistant's behavior. We explicitly tell it to answer ONLY from the
# provided context and to admit when it does not know — this is the first line of defense
# against hallucinations.
ANSWER_SYSTEM_PROMPT = (
    "You are a helpful banking assistant. Answer the user's question using ONLY the provided "
    "context. If the context does not contain the answer, say that you do not know instead of "
    "guessing. Keep the answer concise and factual."
)


def build_answer_prompt(question: str, context_chunks: list[str]) -> tuple[str, str]:
    """Build the (system, user) prompt pair for answering a question with RAG context.

    Args:
        question: The user's question.
        context_chunks: Retrieved knowledge base chunks to ground the answer in.

    Returns:
        A tuple of (system_prompt, user_prompt) ready to pass to an LLMClient.
    """
    context = "\n\n".join(context_chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return ANSWER_SYSTEM_PROMPT, user_prompt
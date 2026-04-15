def build_prompt_no_retrieval(question):
    
    # Strategy 1 — No Retrieval.
    return (
        f"Answer the following question in a few words.\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def build_prompt_always_retrieval(question, contexts):
    
    # Strategy 2 — Always Retrieval (Core RAG Method).
    # Build a list of context passages, each capped at 200 characters
    # to stay within the flan-t5-base 512-token input limit
    context_str = "\n".join([
        f"- {ctx[:200]}"
        for ctx in contexts
        if ctx.strip()           # skip any empty passages
    ])

    return (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def build_prompt_adaptive(question, contexts, needs_retrieval):
    
    # Strategy 3 — Adaptive Retrieval (Simplified Oracle Version).
    if needs_retrieval == 0:
        # Question requires external knowledge -> use RAG prompt
        return build_prompt_always_retrieval(question, contexts)
    else:
        # Question is answerable from memory -> skip retrieval
        return build_prompt_no_retrieval(question)

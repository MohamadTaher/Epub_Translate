import tiktoken

_encoding = None


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken, falling back to a character estimate if unavailable."""
    global _encoding
    try:
        if _encoding is None:
            _encoding = tiktoken.get_encoding("cl100k_base")
        return len(_encoding.encode(text))
    except Exception:
        return len(text) // 4

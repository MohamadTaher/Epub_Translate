"""
Talking to Gemini, and making sense of what comes back.

Nothing here holds any state or knows what a chapter is. It exists because the
interesting part of a failed request is *whether asking again could possibly
help*, and getting that wrong is expensive in both directions: retrying a
refusal spends ten requests to be told the same thing ten times, and giving up
on a hiccup loses a batch that would have worked on the second try.
"""

import re
from typing import Optional

# Google states how long to wait when it refuses a request, in the body of the
# error rather than in a header the client library exposes.
_STATED_RETRY_DELAY = re.compile(r"retry_delay\s*\{\s*seconds:\s*(\d+)", re.IGNORECASE)
_MAX_BACK_OFF_SECONDS = 60

# Why the model stopped, in the cases where asking the same thing again cannot
# change the answer. Everything else — an unspecified reason, OTHER, a dropped
# connection — is worth another go.
_HOPELESS = {
    "SAFETY": "the safety filter refused these chapters",
    "PROHIBITED_CONTENT": "the safety filter refused these chapters",
    "BLOCKLIST": "the safety filter refused these chapters",
    "SPII": "the model saw what it took for personal information in these chapters",
    "RECITATION": "the model stopped, saying it was reciting copyrighted material",
    "LANGUAGE": "the model does not support this language pair",
    "MAX_TOKENS": ("the reply hit the model's output limit and was cut off — the batch is too "
                   "big to answer in one go, so lower TOKENS_PER_REQUEST"),
}


class PatchError(Exception):
    """
    One failed attempt at a batch.

    `retriable` is the whole point of the class: a caller in a retry loop needs
    to know the difference between "the network hiccuped" and "this will be
    refused every time until the end of the world".
    """

    def __init__(self, message: str, retriable: bool = True):
        super().__init__(message)
        self.retriable = retriable


def read_reply(response) -> str:
    """
    The text of a reply, or a `PatchError` saying why there isn't one.

    `response.text` is a property that *raises* when the model returned no
    usable part, which is how a safety block arrives — so it can't be guarded
    with a truth test, and the reason has to be dug out of the candidate.
    """
    if response is None:
        raise PatchError("the API returned nothing")

    blocked = _prompt_block_reason(response)
    if blocked:
        raise PatchError(f"the prompt itself was blocked ({blocked})", retriable=False)

    reason = _finish_reason(response)

    try:
        text = response.text
    except ValueError as error:
        explanation = _HOPELESS.get(reason)
        if explanation:
            raise PatchError(explanation, retriable=False) from error
        raise PatchError(f"the reply had no usable content ({reason or 'no reason given'})") from error

    # A truncated reply still has text, and it is the dangerous kind of bad:
    # plausible, incomplete, and cut off mid-chapter. Refuse it here rather than
    # letting a short answer be mistaken for a complete one.
    if reason == "MAX_TOKENS":
        raise PatchError(_HOPELESS["MAX_TOKENS"], retriable=False)

    if not text or not text.strip():
        raise PatchError("the API returned an empty reply")

    return text


def strip_code_fence(text: str) -> str:
    """Remove a wrapping markdown code fence, which the model sometimes adds."""
    for marker in ["```html", "```"]:
        if text.startswith(marker):
            text = text[len(marker):]
        if text.endswith(marker):
            text = text[:-len(marker)]
    return text.strip()


def retry_after(error: Exception, attempt: int) -> float:
    """
    How long every worker should be held back after a failed request.

    Only a refusal about rate or quota earns a pause; a garbled reply or a
    dropped connection is worth asking again immediately. Google's own stated
    delay is used when the error carries one, and doubling from five seconds
    covers the case where it doesn't.
    """
    message = str(error)
    lowered = message.lower()
    if "429" not in message and "quota" not in lowered and "exhaust" not in lowered:
        return 0.0

    stated = _STATED_RETRY_DELAY.search(message)
    if stated:
        return min(float(stated.group(1)), _MAX_BACK_OFF_SECONDS)

    return min(5.0 * 2 ** attempt, _MAX_BACK_OFF_SECONDS)


def describe_error(error: Exception) -> str:
    """
    A failed attempt in one line, for a log a reader is watching.

    `PatchError` messages are written to be read, so they are passed through
    whole. Anything else is a library's exception, whose first line carries the
    status code and whose remaining paragraphs carry documentation links.
    """
    if isinstance(error, PatchError):
        return str(error)

    text = str(error).strip()
    if not text:
        return type(error).__name__

    first_line = text.splitlines()[0]
    return first_line if len(first_line) <= 160 else first_line[:157] + "..."


def _finish_reason(response) -> str:
    """
    Why the model stopped, as a bare name like "SAFETY", or "" if it won't say.

    Every step of the way is optional. This runs while working out how to report
    a failure, and a second failure raised from inside the first would replace a
    useful message with a confusing one.
    """
    try:
        reason = response.candidates[0].finish_reason
    except Exception:
        return ""

    return getattr(reason, "name", None) or str(reason or "")


def _prompt_block_reason(response) -> Optional[str]:
    """Why the prompt was rejected before the model ever ran, if it was."""
    try:
        blocked = response.prompt_feedback.block_reason
    except Exception:
        return None

    if not blocked:
        return None

    return getattr(blocked, "name", None) or str(blocked)

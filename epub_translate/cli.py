"""Command-line entry point: argument parsing and wiring into EPUBTranslator."""

import argparse
import sys
from pathlib import Path

from . import defaults
from .console import logger
from .translator import EPUBTranslator


def _default_output_path(input_path: str) -> str:
    path = Path(input_path)
    return str(path.with_name(f"{path.stem} (Translated){path.suffix}"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate an EPUB book using Google's Gemini API.")
    parser.add_argument("input", help="Path to the source EPUB file")
    parser.add_argument("-o", "--output", help="Path to write the translated EPUB (default: '<input> (Translated).epub')")
    parser.add_argument("--source-lang", default="auto", help="Source language (default: auto-detect)")
    parser.add_argument("--target-lang", default="English", help="Target language (default: English)")
    parser.add_argument("--glossary", help="Path to a glossary JSON file for term consistency. "
                                           "Terms learned during the run are merged back into it")

    # The pacing defaults come from settings.env, the same file the server
    # reads, so one book runs at one speed however it is started. A flag still
    # wins over the file.
    parser.add_argument("--model", default=defaults.GEMINI_MODEL,
                        help=f"Gemini model name, or set GEMINI_MODEL in settings.env (default: {defaults.GEMINI_MODEL})")
    parser.add_argument("--requests-per-minute", type=int, default=defaults.REQUESTS_PER_MINUTE,
                        help="API request rate limit, which is also how many run at a time, "
                             f"or set REQUESTS_PER_MINUTE in settings.env (default: {defaults.REQUESTS_PER_MINUTE})")
    parser.add_argument("--tokens-per-minute", type=int, default=defaults.TOKENS_PER_MINUTE,
                        help=f"API token rate limit, or set TOKENS_PER_MINUTE in settings.env (default: {defaults.TOKENS_PER_MINUTE:,})")
    parser.add_argument("--tokens-per-request", type=int, default=defaults.TOKENS_PER_REQUEST,
                        help=f"Max tokens per translation request, or set TOKENS_PER_REQUEST in settings.env (default: {defaults.TOKENS_PER_REQUEST:,})")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not defaults.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set. Put it in .env or in the environment.")
        sys.exit(1)

    output_path = args.output or _default_output_path(args.input)

    translator = EPUBTranslator(
        api_key=defaults.GEMINI_API_KEY,
        source_language=args.source_lang,
        target_language=args.target_lang,
        glossary_file_path=args.glossary,
        requests_per_minute=args.requests_per_minute,
        tokens_per_minute=args.tokens_per_minute,
        model_name=args.model,
    )

    translator.translate_book(args.input, output_path, args.tokens_per_request)


if __name__ == "__main__":
    main()

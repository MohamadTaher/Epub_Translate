"""Command-line entry point: argument parsing and wiring into EPUBTranslator."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .logging_utils import logger
from .translator import EPUBTranslator

# From the project root, so the key is found no matter where this is run from.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _default_output_path(input_path: str) -> str:
    path = Path(input_path)
    return str(path.with_name(f"{path.stem} (Translated){path.suffix}"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate an EPUB book using Google's Gemini API.")
    parser.add_argument("input", help="Path to the source EPUB file")
    parser.add_argument("-o", "--output", help="Path to write the translated EPUB (default: '<input> (Translated).epub')")
    parser.add_argument("--source-lang", default="auto", help="Source language (default: auto-detect)")
    parser.add_argument("--target-lang", default="English", help="Target language (default: English)")
    parser.add_argument("--glossary", help="Path to a glossary JSON file for term consistency")
    default_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
    parser.add_argument("--model", default=default_model,
                        help=f"Gemini model name, or set GEMINI_MODEL in .env (default: {default_model})")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Number of patches to translate simultaneously")
    parser.add_argument("--max-requests-per-minute", type=int, default=4, help="API request rate limit")
    parser.add_argument("--max-tokens-per-minute", type=int, default=250000, help="API token rate limit")
    parser.add_argument("--max-tokens-per-patch", type=int, default=15000, help="Max tokens per translation batch")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    output_path = args.output or _default_output_path(args.input)

    translator = EPUBTranslator(
        api_key=api_key,
        source_language=args.source_lang,
        target_language=args.target_lang,
        max_concurrent=args.max_concurrent,
        glossary_file_path=args.glossary,
        max_requests_per_minute=args.max_requests_per_minute,
        max_tokens_per_minute=args.max_tokens_per_minute,
        model_name=args.model,
    )

    translator.translate_book(args.input, output_path, args.max_tokens_per_patch)


if __name__ == "__main__":
    main()

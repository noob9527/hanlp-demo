"""
Sentence splitting module for multilingual text processing.

This module provides intelligent sentence splitting functionality that handles:
- English and Chinese text with appropriate punctuation
- Abbreviations and acronyms to prevent incorrect splits
- Mixed language content
- Various edge cases and special characters

The module uses regex-based preprocessing and sophisticated pattern matching
to achieve accurate sentence boundaries while preserving context.
"""

import re
from typing import Iterator, Tuple

# Special separator character used internally to mark potential split points
# This character is temporarily inserted and later replaced to avoid conflicts
_SEPARATOR = r'@'

# Main sentence detection regex pattern
# Matches sentences ending with .!? or text ending at newlines
# Group 1: (\S.+?[.!?])(?=\s+|$) - Sentences ending with punctuation followed by whitespace or end
# Group 2: (\S.+?)(?=[\n]|$) - Text ending at newlines
_RE_SENTENCE = re.compile(r'(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)', re.UNICODE)

# Regex to identify senior titles/abbreviations (e.g., Mr., Dr., Ms.)
# Matches: Capital letter + 1-2 lowercase letters + period + space + word
# Examples: "Mr. Smith", "Dr. Johnson", "Ms. Davis"
_AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)\s(\w)', re.UNICODE)

# Regex to identify acronyms (e.g., U.S.A., U.K., A.I.)
# Matches: Period + letter + period + space + word
# Examples: "U.S.A. is", "U.K. has", "A.I. research"
_AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)\s(\w)', re.UNICODE)

# Regex to undo separator replacements for senior titles
# Used to restore spaces after processing abbreviations
_UNDO_AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)' + _SEPARATOR + r'(\w)', re.UNICODE)

# Regex to undo separator replacements for acronyms
# Used to restore spaces after processing acronyms
_UNDO_AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)' + _SEPARATOR + r'(\w)', re.UNICODE)


def _replace_with_separator(text, separator, regexs):
    """
    Helper function to replace regex matches with a separator character.

    This function is used to temporarily mark potential split points
    that should not actually cause sentence breaks (like abbreviations).

    Args:
        text (str): Input text to process
        separator (str): Character to use as separator
        regexs (list): List of compiled regex patterns to apply

    Returns:
        str: Text with separators inserted at regex match boundaries
    """
    # Replacement pattern: keep group 1, add separator, keep group 2
    replacement = r"\1" + separator + r"\2"
    result = text

    # Apply each regex pattern sequentially
    for regex in regexs:
        result = regex.sub(replacement, result)

    return result

def split_sentence_with_index(text, best=True) -> Iterator[Tuple[str, int]]:
    """
    Like split_sentence, but also yields the starting character index of each
    returned sentence relative to the original input text.

    The index points to the first character of the emitted sentence in the
    original text (after the same leading/trailing whitespace trimming that
    split_sentence performs for each chunk/sentence).
    """
    if not text:
        return

    # Find all split insertion points according to the same Chinese punctuation
    # preprocessing rules used in split_sentence. Each match inserts a split
    # after group 1, so the split position is match.end(1).
    split_points = []
    for regex in (
        re.compile(r'([。！？?])([^”’])'),
        re.compile(r'(\.{6})([^”’])'),
        re.compile(r'(…{2})([^”’])'),
        re.compile(r'([。！？?][”’])([^，。！？?])'),
    ):
        for m in regex.finditer(text):
            split_points.append(m.end(1))

    # Sort and unique the split points
    split_points = sorted(set(split_points))

    # Build chunks identical to the newline-based preprocessing
    prev = 0
    segments = []  # list of (chunk_text, chunk_start_index_in_original)
    for sp in split_points:
        if sp <= prev:
            continue
        chunk = text[prev:sp]
        # Trim like split_sentence does per chunk
        lstripped = chunk.lstrip()
        if lstripped:
            leading_ws = len(chunk) - len(lstripped)
            segments.append((lstripped, prev + leading_ws))
        prev = sp
    # Tail segment
    if prev < len(text):
        chunk = text[prev:]
        lstripped = chunk.lstrip()
        if lstripped:
            leading_ws = len(chunk) - len(lstripped)
            segments.append((lstripped, prev + leading_ws))

    # If no segments detected (e.g., empty after strip), return nothing
    if not segments:
        return

    for chunk_text, base_index in segments:
        if not best:
            yield (chunk_text, base_index)
            continue

        processed = _replace_with_separator(
            chunk_text, _SEPARATOR, [_AB_SENIOR, _AB_ACRONYM]
        )
        sents = list(_RE_SENTENCE.finditer(processed))
        if not sents:
            yield (chunk_text, base_index)
            continue
        for sentence in sents:
            sent_text = _replace_with_separator(
                sentence.group(), r" ", [_UNDO_AB_SENIOR, _UNDO_AB_ACRONYM]
            )
            start_in_chunk = sentence.start()
            yield (sent_text, base_index + start_in_chunk)


def split_sentence(text, best=True) -> Iterator[str]:
    """
    Split text into sentences using intelligent boundary detection.

    This function handles multilingual text with sophisticated preprocessing
    to avoid splitting on abbreviations, acronyms, and other false positives.

    Args:
        text (str): Input text to split into sentences
        best (bool): If True, use advanced processing for better accuracy.
                    If False, use simple newline-based splitting.

    Yields:
        str: Individual sentences from the input text

    Examples:
        >>> list(split_sentence("Hello world. How are you?"))
        ['Hello world.', 'How are you?']

        >>> list(split_sentence("Dr. Smith said hello. Mr. Johnson replied."))
        ['Dr. Smith said hello.', 'Mr. Johnson replied.']
    """
    # Step 1: Preprocess Chinese punctuation by adding newlines
    # This creates natural break points for Chinese text

    # Add newlines after Chinese sentence endings (。！？) when not followed by quotes
    text = re.sub(r'([。！？?])([^”’])', r"\1\n\2", text)

    # Add newlines after multiple dots (......) when not followed by quotes
    text = re.sub(r'(\.{6})([^”’])', r"\1\n\2", text)

    # Add newlines after Chinese ellipsis (……) when not followed by quotes
    text = re.sub(r'(…{2})([^”’])', r"\1\n\2", text)

    # Add newlines after Chinese punctuation + quotes when not followed by more punctuation
    text = re.sub(r'([。！？?][”’])([^，。！？?])', r'\1\n\2', text)

    # Step 2: Process each chunk (separated by newlines)
    for chunk in text.split("\n"):
        chunk = chunk.strip()

        # Skip empty chunks
        if not chunk:
            continue

        # If not using best mode, yield the chunk as-is
        if not best:
            yield chunk
            continue

        # Step 3: Advanced processing for best mode
        # Temporarily replace abbreviations/acronyms with separators
        processed = _replace_with_separator(chunk, _SEPARATOR, [_AB_SENIOR, _AB_ACRONYM])

        # Find sentence boundaries in the processed text
        sents = list(_RE_SENTENCE.finditer(processed))

        # If no sentences found, yield the original chunk
        if not sents:
            yield chunk
            continue

        # Step 4: Process each detected sentence
        for sentence in sents:
            # Restore spaces by replacing separators with actual spaces
            sentence = _replace_with_separator(sentence.group(), r" ", [_UNDO_AB_SENIOR, _UNDO_AB_ACRONYM])
            yield sentence

# urllib.error.HTTPError: HTTP Error 400: {"detail":"The 2-th sentence exceeds max-length of 150 characters."}
# Text length threshold for using paragraph pipeline (in characters)
__TEXT_LENGTH_THRESHOLD = 120


def should_split(text) -> bool:
    return len(text) > __TEXT_LENGTH_THRESHOLD

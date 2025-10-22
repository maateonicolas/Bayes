"""Utilities for building character and word n-gram probability models and generating text.

The module normalises the input text by removing diacritics, punctuation and by
replacing spaces with underscores, as requested.  These functions are written to be
reusable so they can be invoked from scripts or imported as a library.
"""
from __future__ import annotations

import collections
import random
import re
import unicodedata
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


def normalise_text(raw_text: str) -> str:
    """Return a cleaned, lowercase representation of *raw_text*.

    The transformation removes diacritics, strips punctuation, collapses consecutive
    whitespace and finally replaces spaces with underscores so that the underscore
    becomes part of the character alphabet, as required by the specification.
    """

    # Remove diacritics.
    decomposed = unicodedata.normalize("NFD", raw_text)
    without_diacritics = "".join(ch for ch in decomposed if not unicodedata.combining(ch))

    lowered = without_diacritics.lower()

    # Replace any whitespace with a single space so later replacement with underscores
    # keeps a single separator per token.
    collapsed_whitespace = re.sub(r"\s+", " ", lowered)

    # Remove punctuation and symbols that can complicate the modelling process.  We
    # keep alphanumeric characters and spaces (which will become underscores later).
    cleaned = re.sub(r"[^a-z0-9 ]", "", collapsed_whitespace)

    # Collapse multiple spaces that may appear after removing punctuation.
    collapsed_spaces = re.sub(r" +", " ", cleaned).strip()

    return collapsed_spaces.replace(" ", "_")


def _build_char_ngram_counts(text: str, n: int) -> Tuple[collections.Counter[str], Mapping[str, collections.Counter[str]]]:
    """Compute counts for character n-grams and their prefixes.

    Parameters
    ----------
    text:
        The normalised text.
    n:
        Length of the n-grams.  Must be greater than 1.
    """

    if n <= 1:
        raise ValueError("n must be greater than 1 for character n-grams")

    ngram_counts: collections.Counter[str] = collections.Counter()
    prefix_to_next: MutableMapping[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)

    if len(text) < n:
        return ngram_counts, prefix_to_next

    for i in range(len(text) - n + 1):
        gram = text[i : i + n]
        ngram_counts[gram] += 1
        prefix = gram[:-1]
        next_char = gram[-1]
        prefix_to_next[prefix][next_char] += 1

    return ngram_counts, prefix_to_next


def _counts_to_joint_probabilities(ngram_counts: Mapping[str, int]) -> Dict[str, float]:
    total = sum(ngram_counts.values())
    if total == 0:
        return {}
    return {gram: count / total for gram, count in ngram_counts.items()}


def _counts_to_conditional_probabilities(
        prefix_to_next: Mapping[str, Mapping[str, int]]
) -> Dict[str, Dict[str, float]]:
    conditional: Dict[str, Dict[str, float]] = {}
    for prefix, next_counts in prefix_to_next.items():
        total = sum(next_counts.values())
        if total == 0:
            continue
        conditional[prefix] = {char: count / total for char, count in next_counts.items()}
    return conditional


def build_char_ngram_model(text: str, n: int) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Return the joint and conditional distributions for a character n-gram."""

    counts, prefix_counts = _build_char_ngram_counts(text, n)
    return _counts_to_joint_probabilities(counts), _counts_to_conditional_probabilities(prefix_counts)


def generate_char_sequence(
        start: str,
        conditional_probabilities: Mapping[str, Mapping[str, float]],
        target_length: int,
        order: int,
        rng: random.Random | None = None,
) -> str:
    """Generate a string of characters using the provided conditional probabilities.

    Parameters
    ----------
    start:
        The starting sequence (already normalised) used to bootstrap generation.
    conditional_probabilities:
        Mapping from prefix (length ``order - 1``) to probability distributions over the
        next character.
    target_length:
        Desired length of the resulting text (in characters).
    order:
        Order of the n-gram model (2 for bigram, 3 for trigram, ...).
    rng:
        Optional :class:`random.Random` instance for deterministic behaviour.
    """

    if order <= 1:
        raise ValueError("order must be greater than 1 for character models")

    if rng is None:
        rng = random.Random()

    generated = list(start)
    prefix_length = order - 1

    if len(generated) < prefix_length:
        raise ValueError("start sequence must be at least as long as order - 1")

    available_prefixes = list(conditional_probabilities.keys())
    if not available_prefixes:
        return "".join(generated[:target_length])

    while len(generated) < target_length:
        prefix = "".join(generated[-prefix_length:])
        distribution = conditional_probabilities.get(prefix)
        if not distribution:
            reseed = rng.choice(available_prefixes)
            generated.extend(reseed)
            continue
        next_chars = list(distribution.keys())
        probabilities = list(distribution.values())
        generated.append(rng.choices(next_chars, weights=probabilities, k=1)[0])

    return "".join(generated[:target_length])


def evaluate_char_sequence_probability(
        sequence: str,
        conditional_probabilities: Mapping[str, Mapping[str, float]],
        order: int,
) -> float:
    """Compute the average conditional probability for the provided sequence.

    The metric is the geometric mean of the transition probabilities observed in
    ``sequence``.  Higher values indicate that the sequence is more consistent with the
    model.
    """

    if order <= 1:
        raise ValueError("order must be greater than 1 for evaluation")

    prefix_length = order - 1
    if len(sequence) <= prefix_length:
        return 0.0

    probabilities: List[float] = []
    for i in range(prefix_length, len(sequence)):
        prefix = sequence[i - prefix_length : i]
        next_char = sequence[i]
        distribution = conditional_probabilities.get(prefix)
        if not distribution:
            continue
        probability = distribution.get(next_char)
        if probability is None or probability == 0:
            continue
        probabilities.append(probability)

    if not probabilities:
        return 0.0

    # Geometric mean expressed via logarithms for numerical stability.
    import math

    log_sum = sum(math.log(p) for p in probabilities)
    return math.exp(log_sum / len(probabilities))


def build_word_bigram_model(words: Sequence[str]) -> Tuple[Dict[Tuple[str, str], float], Dict[str, Dict[str, float]]]:
    """Return the joint and conditional distributions for word bigrams."""

    bigram_counts: collections.Counter[Tuple[str, str]] = collections.Counter()
    next_counts: MutableMapping[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)

    if len(words) < 2:
        return {}, {}

    for current_word, next_word in zip(words, words[1:]):
        bigram_counts[(current_word, next_word)] += 1
        next_counts[current_word][next_word] += 1

    total = sum(bigram_counts.values())
    joint_probabilities = (
        {pair: count / total for pair, count in bigram_counts.items()} if total else {}
    )

    conditional_probabilities = _counts_to_conditional_probabilities(next_counts)
    return joint_probabilities, conditional_probabilities


def generate_word_sequence(
        start_words: Iterable[str],
        conditional_probabilities: Mapping[str, Mapping[str, float]],
        target_length: int,
        rng: random.Random | None = None,
) -> List[str]:
    """Generate a sequence of words from a conditional probability table."""

    if rng is None:
        rng = random.Random()

    generated = list(start_words)
    if not generated:
        raise ValueError("start_words must contain at least one element")

    if not conditional_probabilities:
        return generated[:target_length]

    available_prefixes = list(conditional_probabilities.keys())

    while len(generated) < target_length:
        last_word = generated[-1]
        distribution = conditional_probabilities.get(last_word)
        if not distribution:
            reseed = rng.choice(available_prefixes)
            generated.append(reseed)
            continue
        next_words = list(distribution.keys())
        probabilities = list(distribution.values())
        generated.append(rng.choices(next_words, weights=probabilities, k=1)[0])

    return generated[:target_length]


def evaluate_word_sequence_probability(
        sequence: Sequence[str],
        conditional_probabilities: Mapping[str, Mapping[str, float]],
) -> float:
    """Return the geometric mean of conditional probabilities for a word sequence."""

    if len(sequence) < 2:
        return 0.0

    import math

    probabilities: List[float] = []
    for current_word, next_word in zip(sequence, sequence[1:]):
        distribution = conditional_probabilities.get(current_word)
        if not distribution:
            continue
        probability = distribution.get(next_word)
        if probability is None or probability == 0:
            continue
        probabilities.append(probability)

    if not probabilities:
        return 0.0

    log_sum = sum(math.log(p) for p in probabilities)
    return math.exp(log_sum / len(probabilities))


def split_words_from_normalised_text(text: str) -> List[str]:
    """Split the normalised text into words using underscores as separators."""

    if not text:
        return []
    return [token for token in text.split("_") if token]
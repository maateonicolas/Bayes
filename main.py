"""Command line interface for building n-gram models and generating text.

The tool follows the requirements described by the user:
* normalise the input removing diacritics and punctuation;
* replace spaces with underscores so that the underscore is part of the alphabet;
* estimate joint and conditional distributions for character n-grams (n = 2, 3, 4);
* estimate conditional probabilities for word bigrams;
* generate text using the specified starting fragments.

Example
-------
python main.py input.txt --char-length 250 --word-length 250 --seed 1234
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Return an ASCII table given *headers* and *rows*."""

    if not headers:
        return ""

    string_rows: List[List[str]] = [list(map(str, headers))]
    for row in rows:
        string_rows.append([str(cell) for cell in row])

    widths = [max(len(row[i]) for row in string_rows) for i in range(len(headers))]

    def render_row(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)

    lines = [render_row(string_rows[0]), separator]
    for row in string_rows[1:]:
        lines.append(render_row(row))
    return "\n".join(lines)


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]], indent: int = 0) -> None:
    table = format_table(headers, rows)
    indent_str = " " * indent
    for line in table.splitlines():
        print(f"{indent_str}{line}")

from src.ngram_models import (
    build_char_ngram_model,
    build_word_bigram_model,
    evaluate_char_sequence_probability,
    evaluate_word_sequence_probability,
    generate_char_sequence,
    generate_word_sequence,
    normalise_text,
    split_words_from_normalised_text,
)


def summarise_conditional_distribution(
    conditional: Mapping[str, Mapping[str, float]], sample_size: int = 5
) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """Return a deterministic sample of conditional probabilities."""

    sampled: List[Tuple[str, List[Tuple[str, float]]]] = []
    if sample_size <= 0:
        prefixes = sorted(conditional.keys())
    else:
        prefixes = sorted(conditional.keys())[:sample_size]
    for prefix in prefixes:
        transitions = conditional[prefix]
        sorted_transitions = sorted(transitions.items(), key=lambda item: item[1], reverse=True)
        if sample_size > 0:
            sorted_transitions = sorted_transitions[:sample_size]
        sampled.append((prefix, sorted_transitions))
    return sampled


def load_and_normalise_text(path: Path) -> str:
    raw_text = path.read_text(encoding="utf-8")
    return normalise_text(raw_text)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate n-gram models and generate text")
    parser.add_argument("input_file", type=Path, help="Plain text file to analyse")
    parser.add_argument(
        "--char-length",
        type=int,
        default=250,
        help="Target length (in characters) for generated character sequences",
    )
    parser.add_argument(
        "--word-length",
        type=int,
        default=250,
        help="Target length (in words) for generated word sequences",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed used for deterministic generation",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of prefixes displayed when summarising conditional probabilities",
    )
    parser.add_argument(
        "--max-variations",
        type=int,
        default=0,
        help=(
            "Maximum number of alternative character continuations displayed per seed. "
            "Use 0 to show all available transitions."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    normalised_text = load_and_normalise_text(args.input_file)
    print(f"Longitud del texto normalizado: {len(normalised_text)} caracteres")

    char_orders = (2, 3, 4)
    char_models: Dict[int, Tuple[Dict[str, float], Dict[str, Dict[str, float]]]] = {}

    for order in char_orders:
        joint, conditional = build_char_ngram_model(normalised_text, order)
        char_models[order] = (joint, conditional)
        print(f"\nModelo de caracteres de orden {order}:")
        print(f"  n-gramas únicos: {len(joint)}")
        if joint:
            limit = len(joint) if args.sample_size <= 0 else min(args.sample_size, len(joint))
            most_probable = sorted(joint.items(), key=lambda item: item[1], reverse=True)[:limit]
            print("  Principales probabilidades conjuntas:")
            rows = [(repr(gram), f"{prob:.4f}") for gram, prob in most_probable]
            print_table(["n-grama", "P(n-grama)"], rows, indent=4)
        if conditional:
            print("  Muestra de probabilidades condicionales:")
            sample = summarise_conditional_distribution(conditional, args.sample_size)
            for prefix, transitions in sample:
                print(f"    Contexto {prefix!r}:")
                rows = [(repr(next_char), f"{prob:.4f}") for next_char, prob in transitions]
                print_table(["Siguiente", "P(siguiente|contexto)"], rows, indent=8)
        else:
            print("  No hay transiciones suficientes para este orden.")

    words = split_words_from_normalised_text(normalised_text)
    word_joint, word_conditional = build_word_bigram_model(words)
    print("\nModelo de palabras (bigramas):")
    print(f"  Número de palabras únicas: {len(set(words))}")
    print(f"  Bigramas únicos: {len(word_joint)}")
    if word_joint:
        word_limit = len(word_joint) if args.sample_size <= 0 else min(args.sample_size, len(word_joint))
        most_probable_words = sorted(word_joint.items(), key=lambda item: item[1], reverse=True)[
            :word_limit
        ]
        print("  Principales probabilidades conjuntas de palabras:")
        word_rows = [
            (f"('{w1}', '{w2}')", f"{prob:.4f}") for (w1, w2), prob in most_probable_words
        ]
        print_table(["Bigramas", "P(bigrama)"], word_rows, indent=4)
    if word_conditional:
        sample_word_cond = summarise_conditional_distribution(word_conditional, args.sample_size)
        print("  Muestra de probabilidades condicionales de palabras:")
        for prefix, transitions in sample_word_cond:
            print(f"    Palabra '{prefix}':")
            rows = [(next_word, f"{prob:.4f}") for next_word, prob in transitions]
            print_table(["Siguiente", "P(siguiente|palabra)"], rows, indent=8)
    else:
        print("  No hay transiciones suficientes para palabras.")

    print("\nGeneración de texto basada en caracteres:")
    seeds = {2: "el", 3: "el_", 4: "el_p"}
    char_results: Dict[int, Tuple[str, float]] = {}
    for order, seed in seeds.items():
        _, conditional = char_models[order]
        rng = random.Random(args.seed + order)
        generated = generate_char_sequence(seed, conditional, args.char_length, order, rng=rng)
        probability = evaluate_char_sequence_probability(generated, conditional, order)
        char_results[order] = (generated, probability)
        print(f"\nOrden {order} (inicio '{seed}'):")
        print(f"  Texto generado ({len(generated)} caracteres):\n    {generated}")
        print(f"  Promedio geométrico de P(siguiente|contexto): {probability:.4f}")

        prefix_length = order - 1
        prefix = seed[-prefix_length:] if prefix_length > 0 else ""
        transitions = conditional.get(prefix, {})
        if transitions:
            sorted_transitions = sorted(transitions.items(), key=lambda item: item[1], reverse=True)
            print("  Probabilidades de los posibles siguientes caracteres:")
            transition_rows = [
                (repr(next_char), f"{prob:.4f}") for next_char, prob in sorted_transitions
            ]
            print_table(["Siguiente", "P(siguiente|contexto)"], transition_rows, indent=4)

            print("  Variaciones generadas para cada posible siguiente caracter:")
            max_variations = len(sorted_transitions)
            if args.max_variations > 0:
                max_variations = min(args.max_variations, max_variations)

            for idx, (next_char, next_prob) in enumerate(sorted_transitions[:max_variations]):
                variation_seed = seed + next_char
                variation_rng = random.Random(args.seed + order * 100 + idx)
                variation_text = generate_char_sequence(
                    variation_seed,
                    conditional,
                    args.char_length,
                    order,
                    rng=variation_rng,
                )
                variation_probability = evaluate_char_sequence_probability(
                    variation_text, conditional, order
                )
                display_char = repr(next_char)
                print(f"    Variante con siguiente {display_char}:")
                print(f"      Inicio extendido: {variation_seed}")
                print(f"      P({display_char}|{prefix!r}) = {next_prob:.4f}")
                print(
                    "      Promedio geométrico de P(siguiente|contexto): "
                    f"{variation_probability:.4f}"
                )
                print(f"      Texto generado ({len(variation_text)} caracteres):")
                print(f"        {variation_text}")
        else:
            print("  No hay transiciones disponibles para el contexto dado.")

    print("\nGeneración de texto basada en palabras:")
    word_seeds = {
        "el_principito": ["el", "principito"],
        "el_rey_hablo_con": ["el", "rey", "hablo", "con"],
    }
    word_results: Dict[str, Tuple[List[str], float]] = {}
    for label, seed_words in word_seeds.items():
        rng = random.Random(args.seed + len(seed_words))
        generated_words = generate_word_sequence(seed_words, word_conditional, args.word_length, rng=rng)
        probability = evaluate_word_sequence_probability(generated_words, word_conditional)
        word_results[label] = (generated_words, probability)
        rendered = "_".join(generated_words)
        print(f"\nSemilla '{label}':")
        print(f"  Texto generado ({len(generated_words)} palabras):\n    {rendered}")
        print(f"  Promedio geométrico de P(palabra siguiente|palabra actual): {probability:.4f}")

    # Compare the average conditional probabilities to select the most consistent output.
    best_char_order = max(char_results.items(), key=lambda item: item[1][1])[0] if char_results else None
    best_word_seed = max(word_results.items(), key=lambda item: item[1][1])[0] if word_results else None

    print("\nResumen de consistencia:")
    if best_char_order is not None:
        print(
            f"  El modelo de caracteres más consistente es el de orden {best_char_order} "
            f"(promedio geométrico = {char_results[best_char_order][1]:.4f})."
        )
    if best_word_seed is not None:
        print(
            f"  La generación de palabras más consistente proviene de la semilla '{best_word_seed}' "
            f"(promedio geométrico = {word_results[best_word_seed][1]:.4f})."
        )

    if best_char_order is not None and best_word_seed is not None:
        if char_results[best_char_order][1] >= word_results[best_word_seed][1]:
            print(
                "  En este conjunto, el texto por caracteres ofrece mayor consistencia que el modelo por palabras."
            )
        else:
            print(
                "  En este conjunto, el texto por palabras ofrece mayor consistencia que el modelo por caracteres."
            )


if __name__ == "__main__":
    main()

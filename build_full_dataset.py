import json
from pathlib import Path
from typing import Dict, Iterable

from datasets import load_dataset
from transformers import AutoTokenizer


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def iter_tokens_with_line_lengths(
    poem_idx: int, poem: str, tokenizer
) -> Iterable[Dict]:
    """
    Yields per-token rows containing only:
    - text: poem text from start through this token (inclusive)
    - line_length: current line length after processing this token
    """
    text = normalize_newlines(poem)
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    offsets = encoded["offset_mapping"]

    line_length = 0

    for token_index, (token, (start, end)) in enumerate(zip(tokens, offsets)):
        piece = text[start:end]
        for char in piece:
            line_length = 0 if char == "\n" else line_length + 1

        yield {
            "text": text[:end],
            "line_length": line_length,
        }


def main() -> None:
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "line_length_dataset.jsonl"

    model_id = "google/gemma-3-270m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset("suayptalha/Poetry-Foundation-Poems", split="train")

    total_rows = 0
    with output_path.open("w", encoding="utf-8") as fp:
        for poem_idx, row in enumerate(dataset):
            poem = row["Poem"]
            for record in iter_tokens_with_line_lengths(poem_idx, poem, tokenizer):
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_rows += 1

            if (poem_idx + 1) % 1000 == 0:
                print(f"Processed {poem_idx + 1} poems...")

    print(f"Finished. Wrote {total_rows} rows to {output_path}")


if __name__ == "__main__":
    main()

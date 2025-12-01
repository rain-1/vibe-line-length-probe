import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from transformers import AutoTokenizer


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def tokens_with_line_lengths(poem: str, tokenizer, limit: int | None = None) -> List[Dict]:
    text = normalize_newlines(poem)
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    offsets = encoded["offset_mapping"]

    line_length = 0
    rows: List[Dict] = []

    for idx, (token, (start, end)) in enumerate(zip(tokens, offsets)):
        piece = text[start:end]
        for char in piece:
            line_length = 0 if char == "\n" else line_length + 1

        rows.append(
            {
                "index": idx,
                "token": token,
                "text": piece,
                "offset": [start, end],
                "line_length": line_length,
            }
        )

        if limit is not None and idx + 1 >= limit:
            break

    return rows


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    dataset = load_dataset("suayptalha/Poetry-Foundation-Poems", split="train")

    sample_poem = dataset[0]["Poem"]
    preview = tokens_with_line_lengths(sample_poem, tokenizer, limit=40)

    for row in preview:
        print(
            f"{row['index']:>3} | {row['token']:<18} | "
            f"line_len={row['line_length']:>3} | text={repr(row['text'])}"
        )

    output_path = Path("sample_line_lengths.jsonl")
    with output_path.open("w", encoding="utf-8") as fp:
        for row in preview:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote preview to {output_path}")


if __name__ == "__main__":
    main()

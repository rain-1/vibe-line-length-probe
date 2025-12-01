import json
from pathlib import Path
import random
from typing import Iterator

from transformers import AutoTokenizer
from tqdm import tqdm


def iter_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    source_path = Path("data/line_length_dataset.jsonl")
    output_path = Path("data/line_length_dataset_10k_max300.jsonl")
    model_id = "google/gemma-3-270m"
    max_tokens = 300
    sample_size = 10_000

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    candidates = []
    total = 0
    for row in tqdm(iter_rows(source_path), desc="Filtering rows"):
        total += 1
        if row["line_length"] >= 70:
            continue
        token_len = len(tokenizer(row["text"]).input_ids)
        if token_len > max_tokens:
            continue
        candidates.append({"text": row["text"], "line_length": row["line_length"]})
    print(f"Filtered down to {len(candidates)} candidates out of {total} processed.")

    random.seed(42)
    random.shuffle(candidates)
    subset = candidates[:sample_size]

    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_fp:
        for row in subset:
            out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(subset)} rows to {output_path}")


if __name__ == "__main__":
    main()

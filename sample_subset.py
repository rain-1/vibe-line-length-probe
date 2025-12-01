import json
import random
from pathlib import Path


def reservoir_sample_jsonl(input_path: Path, k: int, seed: int = 42):
    random.seed(seed)
    reservoir = []
    with input_path.open("r", encoding="utf-8") as fp:
        for i, line in enumerate(fp, start=1):
            if len(reservoir) < k:
                reservoir.append(line)
            else:
                j = random.randint(1, i)
                if j <= k:
                    reservoir[j - 1] = line
    random.shuffle(reservoir)
    return reservoir


def main() -> None:
    input_path = Path("data/line_length_dataset.jsonl")
    output_path = Path("data/line_length_dataset_sample.jsonl")
    sample_size = 5000

    print(f"Sampling {sample_size} lines from {input_path} ...")
    sample = reservoir_sample_jsonl(input_path, sample_size, seed=42)

    with output_path.open("w", encoding="utf-8") as fp:
        for line in sample:
            fp.write(line.rstrip("\n") + "\n")

    print(f"Wrote sample to {output_path}")


if __name__ == "__main__":
    main()

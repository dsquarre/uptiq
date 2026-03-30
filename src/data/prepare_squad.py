import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset


def build_records(split: str) -> list[dict]:
    dataset = load_dataset("squad_v2", split=split)
    records: list[dict] = []
    for row in dataset:
        answers = row.get("answers", {}).get("text", [])
        cleaned_answers = [a.strip() for a in answers if isinstance(a, str) and a.strip()]
        item = {
            "id": str(row.get("id", "")),
            "question": str(row.get("question", "")).strip(),
            "context": str(row.get("context", "")).strip(),
            "answers": cleaned_answers,
            "is_impossible": bool(row.get("is_impossible", False)),
        }
        if item["id"] and item["question"] and item["context"]:
            records.append(item)
    return records


def write_jsonl(records: list[dict], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = build_records(args.split)
    if args.num_samples > len(records):
        raise ValueError(f"Requested {args.num_samples} samples, but only {len(records)} available")

    random.seed(args.seed)
    sampled = random.sample(records, args.num_samples)
    sampled = sorted(sampled, key=lambda x: x["id"])

    write_jsonl(sampled, args.output)

    print(json.dumps({
        "output": os.path.abspath(args.output),
        "num_samples": len(sampled),
        "seed": args.seed,
        "split": args.split,
    }, indent=2))


if __name__ == "__main__":
    main()

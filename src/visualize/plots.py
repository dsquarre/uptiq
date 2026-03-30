import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


METRICS = [
    "em",
    "f1",
    "recall_at_k",
    "mrr",
    "judge_correctness",
    "judge_completeness",
    "judge_reasoning",
    "latency_ms",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    for metric in METRICS:
        labels = []
        values = []
        for architecture, stats in summary.items():
            if metric in stats:
                labels.append(architecture)
                values.append(stats[metric])

        if not labels:
            continue

        plt.figure(figsize=(7, 4))
        ax = sns.barplot(x=labels, y=values)
        ax.set_title(f"{metric} comparison")
        ax.set_xlabel("Architecture")
        ax.set_ylabel(metric)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=140)
        plt.close()

    print(json.dumps({"output_dir": str(output_dir), "metrics_plotted": METRICS}, indent=2))


if __name__ == "__main__":
    main()

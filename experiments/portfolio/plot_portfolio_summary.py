import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator, NullLocator


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot portfolio summary timing.")
    parser.add_argument(
        "--input",
        type=str,
        default="experiments/portfolio/logs/portfolio_summary.csv",
        help="Path to portfolio_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/portfolio/logs/portfolio_summary.pdf",
        help="Path to save the plot image",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    if df.empty:
        raise SystemExit(f"No data in {input_path}")

    display_name = {
        "dXPP": "dXPP"
    }

    x_labels = [c for c in df.columns if c != "model"]
    x = [int(c) for c in x_labels]

    plt.figure(figsize=(6, 4.5))
    for _, row in df.iterrows():
        y = [row[c] for c in x_labels]
        label = display_name.get(row["model"], row["model"])
        plt.plot(x, y, marker="o", markersize=4, linewidth=1.5, label=label)

    plt.xscale("log")
    plt.xticks(x, x_labels)
    plt.gca().xaxis.set_minor_locator(NullLocator())
    plt.yscale("log")
    plt.gca().yaxis.set_major_locator(LogLocator(base=10.0))
    plt.gca().yaxis.set_minor_locator(NullLocator())
    plt.xlabel("H")
    plt.ylabel("Total time (ms)")
    plt.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.ticker import NullLocator, NullFormatter
from pathlib import Path

MODEL_ORDER = ["dXPP", "dQP_gurobi", "OptNet", "SCQPTH", "Cvxpy"]
DISPLAY_NAME = {
    "dXPP": "dXPP",
    "dQP_gurobi": "dQP",
    "OptNet": "OptNet",
    "SCQPTH": "SCQPTH",
    "Cvxpy": "CvxpyLayer",
}


def load_backward(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "dim" not in df.columns:
        raise ValueError(f"'dim' column missing in {csv_path}")
    df = df.set_index("dim")
    # Ensure consistent model order
    df = df.reindex(columns=MODEL_ORDER)
    return df.index.tolist(), df


def plot_panel(ax, x_labels, df, title, ylabel):
    x_pos = list(range(len(x_labels)))
    
    # Draw horizontal dashed line at 1.0 as baseline
    ax.axhline(1, color="black", linestyle="--", linewidth=1, alpha=0.8, zorder=1)
    
    # Take only dXPP
    if "dXPP" in df.columns:
        series = df["dXPP"]
        mask = series.notna()
        ax.bar(
            [x_pos[i] for i, m in enumerate(mask) if m],
            series[mask].astype(float).values,
            color="tab:blue",
            alpha=0.7,
            label=DISPLAY_NAME["dXPP"],
            zorder=3,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30)
    # Linear scale for y-axis
    ax.set_yscale("linear")
    ax.grid(True, linestyle="--", alpha=0.5, which="major")


def main():
    base = Path(__file__).parent / "results"
    out_path = base / "plot_backward_ratio_panels.pdf"
    base.mkdir(parents=True, exist_ok=True)

    chain_bw_labels, chain_bw = load_backward(base / "chain_backward_ms.csv")
    rp_bw_labels, rp_bw = load_backward(base / "random_projection_backward_ms.csv")

    # Calculate ratio relative to dQP_gurobi
    chain_bw_ratio = chain_bw.div(chain_bw["dQP_gurobi"], axis=0)
    rp_bw_ratio = rp_bw.div(rp_bw["dQP_gurobi"], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_panel(
        axes[0],
        rp_bw_labels,
        rp_bw_ratio,
        "Projection - Backward Runtime Ratio",
        "Ratio (dXPP / dQP)",
    )
    plot_panel(
        axes[1],
        chain_bw_labels,
        chain_bw_ratio,
        "Chain - Backward Runtime Ratio",
        "Ratio (dXPP / dQP)",
    )

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()


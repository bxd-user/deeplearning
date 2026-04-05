from __future__ import annotations

import argparse

from src.ra_bias.evaluation import Metrics
from src.ra_bias.runner import (
    MAIN_METHODS,
    run_ablations,
    run_methods,
)


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _print_main_table(metrics: list[Metrics]) -> None:
    print("\n== Main Results (statics.md §1) ==")
    print(f"{'Method':<22} {'Success Rate':>12} {'Avg Steps':>10} {'Wasted Ratio':>13} {'Cost/Success':>13}")
    print("-" * 74)
    for m in metrics:
        print(
            f"{m.method_name:<22} {_pct(m.success_rate):>12} "
            f"{m.avg_steps_to_success:>10.2f} {m.wasted_exploration_ratio:>13.3f} "
            f"{m.cost_per_success:>13.0f}"
        )


def _print_ablation_table(title: str, metrics: list[Metrics]) -> None:
    print(f"\n== {title} ==")
    print(f"{'Method':<22} {'Success Rate':>12} {'Avg Steps':>10} {'Wasted Ratio':>13}")
    print("-" * 60)
    for m in metrics:
        print(
            f"{m.method_name:<22} {_pct(m.success_rate):>12} "
            f"{m.avg_steps_to_success:>10.2f} {m.wasted_exploration_ratio:>13.3f}"
        )


def _print_quality_trend(metrics: list[Metrics]) -> None:
    """
    Per-round avg selected-candidate quality (statics.md §2.3).
    Shows how bias guidance improves selection quality over rounds.
    """
    # Determine the max round across all methods
    all_rounds = sorted({r for m in metrics for r in m.round_quality_trend})
    if not all_rounds:
        return

    print("\n== Per-Round Avg. Selected Candidate Quality (statics.md §2.3) ==")
    header = f"{'Round':>6}" + "".join(f"{m.method_name:>22}" for m in metrics)
    print(header)
    print("-" * (6 + 22 * len(metrics)))
    for r in all_rounds:
        row = f"{r:>6}"
        for m in metrics:
            val = m.round_quality_trend.get(r)
            row += f"{val:>22.3f}" if val is not None else f"{'—':>22}"
        print(row)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reflection-as-Bias experiments.")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes per method.")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Root output directory for logs.")
    parser.add_argument("--no-ablation", action="store_true",
                        help="Skip ablation experiments.")
    args = parser.parse_args()

    # ── Main experiment ───────────────────────────────────────────────────────
    main_metrics = run_methods(
        methods=MAIN_METHODS,
        num_episodes=args.episodes,
        output_dir=f"{args.output}/main",
    )
    _print_main_table(main_metrics)
    _print_quality_trend(main_metrics)

    if args.no_ablation:
        return

    # ── Ablation experiments ──────────────────────────────────────────────────
    ablation_results = run_ablations(
        num_episodes=args.episodes,
        output_dir=f"{args.output}/ablations",
    )

    _print_ablation_table(
        "Ablation 1: Hard Pruning vs Soft Bias (statics.md §3.1)",
        ablation_results["hard_vs_soft"],
    )
    _print_ablation_table(
        "Ablation 2: Tag-based vs LLM-based Bias (statics.md §3.2)",
        ablation_results["bias_type"],
    )
    _print_ablation_table(
        "Ablation 3: No Bias / Weak Bias / Full Bias (statics.md §3.3)",
        ablation_results["bias_strength"],
    )


if __name__ == "__main__":
    main()

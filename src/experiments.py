# src/experiments.py
# Multi-scenario experiment runner with replications and summary CSVs.
# Works with run_baseline.py (two-type mix, blocking, monitoring).

from dataclasses import replace
from pathlib import Path
import pandas as pd
import numpy as np

# Import simulator entrypoints
from run_baseline import SimConfig, run_once

def summarize_run(sys) -> dict:
    """Compute compact KPIs from a completed run_once(cfg)."""
    # Monitoring-based KPIs
    avg_prep_q = float(np.mean(sys.prep_q_samples)) if sys.prep_q_samples else float("nan")
    avg_or_util = float(np.mean(sys.or_util_samples)) if sys.or_util_samples else float("nan")

    # Throughput
    n_done = len(sys.records)
    sim_hours = sys.cfg.sim_time / 60.0
    th_per_hour = n_done / sim_hours if sim_hours > 0 else float("nan")

    # Patient-level KPIs
    mean_ttis = float(np.mean([r["total_time_in_system"] for r in sys.records])) if sys.records else float("nan")
    mean_block1 = float(np.mean([r["block_prep_to_or"] for r in sys.records])) if sys.records else float("nan")
    mean_block2 = float(np.mean([r["block_or_to_rec"] for r in sys.records])) if sys.records else float("nan")

    return {
        # Scenario parameters (baseline + light mix)
        "scenario_label": sys.cfg.run_label,
        "p_prep": sys.cfg.p_prep,
        "p_or": sys.cfg.p_or,
        "p_rec": sys.cfg.p_rec,
        "mean_iat": sys.cfg.mean_iat,
        "mean_prep": sys.cfg.mean_prep,
        "mean_or": sys.cfg.mean_or,
        "mean_rec": sys.cfg.mean_rec,
        "frac_light": sys.cfg.frac_light,
        "mean_prep_light": sys.cfg.mean_prep_light,
        "mean_or_light": sys.cfg.mean_or_light,
        "mean_rec_light": sys.cfg.mean_rec_light,
        "sim_time_min": sys.cfg.sim_time,
        "seed": sys.cfg.rng_seed,

        # KPIs
        "completed": n_done,
        "throughput_per_hour": th_per_hour,
        "avg_prep_queue_len": avg_prep_q,
        "avg_or_utilisation": avg_or_util,
        "mean_total_time_in_system": mean_ttis,
        "mean_block_prep_to_or": mean_block1,
        "mean_block_or_to_rec": mean_block2,
    }

def run_replications(base_cfg: SimConfig, n_reps: int, label: str, seed0: int = 500) -> pd.DataFrame:
    """Run n_reps with different seeds for the same scenario and return a DataFrame of summaries."""
    rows = []
    for k in range(n_reps):
        cfg = replace(base_cfg, rng_seed=seed0 + k, run_label=f"{label}_rep{k+1}")
        sys = run_once(cfg)  # writes per-run CSVs (patient + monitor)
        rows.append(summarize_run(sys))
    return pd.DataFrame(rows)

def main():
    # ----- Define base and scenarios -----
    base = SimConfig()  # baseline from assignment

    scenarios = {
        # label: config tweaks
        "baseline": base,
        "light40": replace(base, frac_light=0.4, run_label="light40"),
        "light80": replace(base, frac_light=0.8, run_label="light80"),
        "more_prep": replace(base, p_prep=4, run_label="more_prep"),
        "more_recovery": replace(base, p_rec=4, run_label="more_recovery"),
        "faster_or": replace(base, mean_or=15.0, run_label="faster_or"),
        "heavier_load": replace(base, mean_iat=20.0, run_label="heavier_load"),
    }

    N_REPS = 2  # increase to 5–10 for tighter CIs

    all_rows = []
    for label, cfg in scenarios.items():
        df_reps = run_replications(cfg, N_REPS, label, seed0=500)
        df_reps["scenario"] = label
        all_rows.append(df_reps)

    df_all = pd.concat(all_rows, ignore_index=True)

    # ----- Save per-replication results -----
    Path("results").mkdir(parents=True, exist_ok=True)
    raw_path = Path("results") / "summary_runs_raw.csv"
    df_all.to_csv(raw_path, index=False)
    print(f"Saved per-replication results to: {raw_path}")

    # ----- Aggregate mean & std by scenario/params -----
    agg_cols = [
        "completed", "throughput_per_hour", "avg_prep_queue_len",
        "avg_or_utilisation", "mean_total_time_in_system",
        "mean_block_prep_to_or", "mean_block_or_to_rec",
    ]

    group_keys = [
        "scenario",
        "p_prep", "p_or", "p_rec",
        "mean_iat", "mean_prep", "mean_or", "mean_rec",
        "frac_light", "mean_prep_light", "mean_or_light", "mean_rec_light",
        "sim_time_min",
    ]

    grouped = (
        df_all
        .groupby(group_keys, dropna=False)[agg_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    agg_path = Path("results") / "summary.csv"
    grouped.to_csv(agg_path, index=False)
    print(f"Saved aggregated scenario summary to: {agg_path}")

    print("\nPreview (aggregated):")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(grouped.head(20))

if __name__ == "__main__":
    main()

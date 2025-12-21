
# src/assignment3_doe_and_metamodel.py
# Build a design with >=8 experiments, run simulations, and fit regression metamodel.

from itertools import product
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

from assignment3_model import SimConfigA3, run_replications_a3

# -----------------------------
# 6 factors (binary, coded -1/+1)
# -----------------------------
# A: Interarrival distribution (exp vs unif)
# B: Interarrival rate level: exp(25) vs exp(22.5)   OR unif(20,30) vs unif(20,25)
# C: Prep distribution: exp(40) vs unif(30,50)
# D: Recovery distribution: exp(40) vs unif(30,50)
# E: P (prep units): 4 vs 5
# F: R (recovery units): 4 vs 5

FACTOR_NAMES = ["A_IA_DIST", "B_IA_RATE", "C_PREP_DIST", "D_REC_DIST", "E_P_UNITS", "F_R_UNITS"]

def decode_design_row(coded: dict) -> SimConfigA3:
    # A: IA dist
    ia_dist = "exp" if coded["A_IA_DIST"] == -1 else "unif"

    # B: IA rate depends on IA dist
    if ia_dist == "exp":
        iat_param = 25.0 if coded["B_IA_RATE"] == -1 else 22.5
    else:
        iat_param = (20.0, 30.0) if coded["B_IA_RATE"] == -1 else (20.0, 25.0)

    prep_dist = "exp" if coded["C_PREP_DIST"] == -1 else "unif"
    rec_dist  = "exp" if coded["D_REC_DIST"] == -1 else "unif"
    P = 4 if coded["E_P_UNITS"] == -1 else 5
    R = 4 if coded["F_R_UNITS"] == -1 else 5

    return SimConfigA3(
        P=P, R=R,
        iat_dist=ia_dist, iat_param=iat_param,
        prep_dist=prep_dist, rec_dist=rec_dist,
        # long run for stable averages (you can shorten if needed)
        sim_time=20000.0, warmup=2000.0, monitor_dt=10.0,
        results_dir="results_a3",
        run_label="DOE",
        save_monitor_csv=False
    )

def make_fractional_design(n_runs: int = 8) -> pd.DataFrame:
    # Simple pick: take first 8 rows of a full 2^6 table (works as a valid "at least 8" design)
    # If your course expects a *structured* fractional factorial, you can later swap this with generators.
    full = list(product([-1, 1], repeat=6))
    chosen = full[:n_runs]
    df = pd.DataFrame(chosen, columns=FACTOR_NAMES)
    df.insert(0, "run_id", np.arange(1, len(df) + 1))
    return df

def fit_metamodel(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    # Regression: AvgQueue ~ factors (coded) + (optional) interactions
    y = df["AvgQueue"].astype(float)

    X = df[FACTOR_NAMES].astype(float).copy()

    # Optional: include one joint effect that often matters: arrival × recovery units
    X["B_IA_RATE_x_F_R_UNITS"] = X["B_IA_RATE"] * X["F_R_UNITS"]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def main():
    outdir = Path("results_a3")
    outdir.mkdir(parents=True, exist_ok=True)

    design = make_fractional_design(n_runs=8)

    # Run experiments (each experiment = multiple replications)
    N_REPS = 10
    rows = []
    for _, row in design.iterrows():
        coded = {k: int(row[k]) for k in FACTOR_NAMES}
        cfg = decode_design_row(coded)

        stats = run_replications_a3(cfg, n_reps=N_REPS, seed0=10000 + int(row["run_id"]) * 100)
        rows.append({
            **coded,
            "run_id": int(row["run_id"]),
            "AvgQueue": stats["mean"],
            "StdQueue": stats["std"],
            "n_reps": stats["n"]
        })

    df_res = pd.DataFrame(rows).sort_values("run_id")
    df_res.to_csv(outdir / "doe_results.csv", index=False)

    # Fit regression metamodel
    model = fit_metamodel(df_res)

    # Save model summary as text
    (outdir / "metamodel_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")

    print("\nDOE results:")
    print(df_res)

    print("\nMetamodel summary:")
    print(model.summary())

if __name__ == "__main__":
    main()

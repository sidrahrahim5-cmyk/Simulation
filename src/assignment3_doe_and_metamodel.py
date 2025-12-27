# src/assignment3_doe_and_metamodel.py
# DOE (>=8 experiments) + regression metamodel for entrance queue length.

from itertools import product
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

from src.assignment3_model import SimConfigA3, run_replications_a3

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
    rec_dist = "exp" if coded["D_REC_DIST"] == -1 else "unif"
    P = 4 if coded["E_P_UNITS"] == -1 else 5
    R = 4 if coded["F_R_UNITS"] == -1 else 5

    return SimConfigA3(
        P=P, R=R,
        iat_dist=ia_dist, iat_param=iat_param,
        prep_dist=prep_dist, rec_dist=rec_dist,
        sim_time=20000.0, warmup=2000.0, monitor_dt=10.0,
        results_dir="results_a3",
        run_label="DOE",
        save_monitor_csv=False
    )


def make_design(n_runs: int = 8) -> pd.DataFrame:
    # L8 orthogonal array style design (balanced 8-run for main effects)
    # Columns are -1/+1 and each factor changes across runs.
    design = np.array([
        [-1, -1, -1, -1, -1, -1],
        [-1, -1,  1,  1,  1,  1],
        [-1,  1, -1, -1,  1,  1],
        [-1,  1,  1,  1, -1, -1],
        [ 1, -1, -1,  1, -1,  1],
        [ 1, -1,  1, -1,  1, -1],
        [ 1,  1, -1,  1,  1, -1],
        [ 1,  1,  1, -1, -1,  1],
    ], dtype=int)

    df = pd.DataFrame(design, columns=FACTOR_NAMES)
    df.insert(0, "run_id", np.arange(1, len(df) + 1))
    return df


def fit_metamodel(df: pd.DataFrame):
    # MAIN EFFECTS ONLY (valid for 8-run screening design)
    y = df["AvgQueue"].astype(float)
    X = df[FACTOR_NAMES].astype(float).copy()
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model


def main():
    outdir = Path("results_a3")
    outdir.mkdir(parents=True, exist_ok=True)

    design = make_design(n_runs=8)

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

    model = fit_metamodel(df_res)
    (outdir / "metamodel_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")

    print("\nDOE results:")
    print(df_res)

    print("\nMetamodel summary:")
    print(model.summary())


if __name__ == "__main__":
    main()

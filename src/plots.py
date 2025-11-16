def plot_scenario_bars(summary_csv_path: str):
    """
    Robust bar charts comparing scenarios using results/summary.csv from Step 8.
    Handles MultiIndex-like column names that CSVs often get after .agg(["mean","std"]).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    RESULTS_DIR = Path("results")
    df = pd.read_csv(summary_csv_path)

    # --- Flatten/normalize column names to simple strings
    flat_cols = []
    for c in df.columns:
        s = str(c)
        s = s.replace("(", "").replace(")", "").replace("'", "").replace(" ", "")
        s = s.replace("[", "").replace("]", "")
        s = s.replace("__", "_")
        flat_cols.append(s)
    df.columns = flat_cols

    # --- Ensure we have a scenario label column
    scen_col = None
    for cand in ["scenario", "Scenario", "SCENARIO"]:
        if cand in df.columns:
            scen_col = cand
            break
    if scen_col is None:
        raise KeyError("Could not find a 'scenario' column in summary.csv")

    labels = df[scen_col].astype(str).fillna("unknown")
    x = np.arange(len(labels))  # numeric x positions

    # --- Helper to pick a column containing both the base name and 'mean' (prefer mean)
    def pick(base: str) -> str:
        base = base.lower()
        candidates = [c for c in df.columns if base in c.lower()]
        if not candidates:
            raise KeyError(f"Could not find any column containing '{base}'")
        mean_first = [c for c in candidates if "mean" in c.lower()]
        return mean_first[0] if mean_first else candidates[0]

    # Resolve metric columns
    thr_col  = pick("throughput_per_hour")
    util_col = pick("avg_or_utilisation")
    ttis_col = pick("mean_total_time_in_system")

    # 1) Throughput per hour
    plt.figure()
    plt.bar(x, df[thr_col].values)
    plt.ylabel("Throughput (patients/hour)")
    plt.title("Scenario comparison: throughput per hour")
    plt.xticks(x, labels, rotation=20, ha="right")
    out1 = RESULTS_DIR / "plot_throughput_per_hour.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()

    # 2) Average OR utilisation
    plt.figure()
    plt.bar(x, df[util_col].values)
    plt.ylabel("Avg OR utilisation (busy fraction)")
    plt.title("Scenario comparison: OR utilisation")
    plt.xticks(x, labels, rotation=20, ha="right")
    out2 = RESULTS_DIR / "plot_or_utilisation.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()

    # 3) Mean total time in system
    plt.figure()
    plt.bar(x, df[ttis_col].values)
    plt.ylabel("Mean total time in system (min)")
    plt.title("Scenario comparison: total time in system")
    plt.xticks(x, labels, rotation=20, ha="right")
    out3 = RESULTS_DIR / "plot_total_time_in_system.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    print(f"Saved: {out3}")

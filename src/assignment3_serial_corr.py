# src/assignment3_serial_corr.py
# Serial correlation experiment for a "long-memory" scenario.

import numpy as np
import pandas as pd
from pathlib import Path

from src.assignment3_model import SimConfigA3, run_once_a3


def lag_corr(x: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(x) <= lag:
        return float("nan")
    a = x[:-lag]
    b = x[lag:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def extract_block_samples(series: np.ndarray, n_blocks: int, window_size: int, gap_size: int) -> np.ndarray:
    samples = []
    idx = 0
    for _ in range(n_blocks):
        start = idx
        end = idx + window_size
        if end > len(series):
            break
        samples.append(np.mean(series[start:end]))
        idx = end + gap_size
    return np.array(samples, dtype=float)


def main():
    # Choose a "high utilisation" scenario (but now model is correct, queue should be moderate)
    cfg = SimConfigA3(
        P=4, R=4,
        iat_dist="exp", iat_param=22.5,  # start with 25 (util ~0.8). you can test 22.5 too.
        prep_dist="exp",
        rec_dist="exp",
        sim_time=20000.0,
        warmup=2000.0,
        monitor_dt=10.0,
        save_monitor_csv=False,
        results_dir="results_a3",
        run_label="serialcorr"
    )

    # block sampling settings
    n_blocks = 10
    window_size = 30
    gap_size = 60

    lags = [1, 2, 3]
    rows = []

    for run_id in range(10):
        cfg_run = SimConfigA3(**{**cfg.__dict__, "rng_seed": 5000 + run_id})
        sys = run_once_a3(cfg_run)

        raw_series = np.array(sys.prep_q_samples, dtype=float)
        block_series = extract_block_samples(raw_series, n_blocks, window_size, gap_size)

        row = {
            "run": run_id + 1,
            "raw_mean_queue": float(np.mean(raw_series)),
            "raw_n_samples": len(raw_series),
            "block_n_samples": len(block_series),
            "window_size_points": window_size,
            "gap_size_points": gap_size,
        }
        for L in lags:
            row[f"lag{L}_corr_blocks"] = lag_corr(block_series, L)

        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nSerial correlation (BLOCK samples) per run:")
    print(df)

    lag_cols = [c for c in df.columns if c.startswith("lag")]
    means = df[lag_cols].mean()

    print("\nAverage BLOCK-sample correlations across runs:")
    print(means)

    outdir = Path("results_a3")
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "serial_correlation_blocks.csv", index=False)
    means.to_csv(outdir / "serial_correlation_blocks_means.csv", header=["mean_corr"])


if __name__ == "__main__":
    main()

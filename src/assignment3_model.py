# src/assignment3_model.py
# Assignment 3: Experiment design + metamodelling
# OR system: PREP (P units) -> OR (1 unit) -> REC (R units)
#
# Correct patient life cycle (model-solution style, non-blocking):
#   PREP -> wait for OR -> OR -> wait for REC -> REC
# Resources are released immediately after each stage completes.

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple, Dict, Any, List, Union

import simpy
import numpy as np
import pandas as pd

DistName = Literal["exp", "unif"]
IATParam = Union[float, Tuple[float, float]]


@dataclass
class SimConfigA3:
    # Resources
    P: int = 4
    R: int = 4

    # Interarrival distribution
    iat_dist: DistName = "exp"
    iat_param: IATParam = 25.0  # exp mean OR unif(a,b)

    # Prep distribution
    prep_dist: DistName = "exp"  # exp(40) OR unif(30,50)

    # Recovery distribution
    rec_dist: DistName = "exp"   # exp(40) OR unif(30,50)

    # OR distribution fixed
    or_mean: float = 20.0

    # Run control
    sim_time: float = 20000.0
    warmup: float = 2000.0
    monitor_dt: float = 10.0
    rng_seed: int = 1

    # Output (optional)
    results_dir: str = "results_a3"
    run_label: str = "a3_run"
    save_monitor_csv: bool = False


class HospitalA3:
    def __init__(self, env: simpy.Environment, cfg: SimConfigA3, rng: np.random.Generator):
        self.env = env
        self.cfg = cfg
        self.rng = rng

        self.prep = simpy.Resource(env, capacity=cfg.P)
        self.or_room = simpy.Resource(env, capacity=1)
        self.rec = simpy.Resource(env, capacity=cfg.R)

        # Monitoring buffers
        self.t_samples: List[float] = []
        self.prep_q_samples: List[int] = []

    # ---- Distribution samplers ----
    def sample_iat(self) -> float:
        if self.cfg.iat_dist == "exp":
            mean = float(self.cfg.iat_param)  # type: ignore[arg-type]
            if mean <= 0:
                raise ValueError("Exp interarrival mean must be > 0")
            return float(self.rng.exponential(mean))

        a, b = self.cfg.iat_param  # type: ignore[misc]
        a = float(a)
        b = float(b)
        if not (a < b):
            raise ValueError("Uniform interarrival must satisfy a < b")
        return float(self.rng.uniform(a, b))

    def sample_prep(self) -> float:
        if self.cfg.prep_dist == "exp":
            return float(self.rng.exponential(40.0))
        return float(self.rng.uniform(30.0, 50.0))

    def sample_rec(self) -> float:
        if self.cfg.rec_dist == "exp":
            return float(self.rng.exponential(40.0))
        return float(self.rng.uniform(30.0, 50.0))

    def sample_or(self) -> float:
        return float(self.rng.exponential(float(self.cfg.or_mean)))  # Exp(20) fixed


def patient_process(env: simpy.Environment, sys: HospitalA3):
    """
    Correct patient life cycle (compare with model solution patient lifecycle):
      PREP -> wait for OR -> OR -> wait for REC -> REC

    Key fix:
    - PREP is released immediately after preparation ends.
    - OR is released immediately after operation ends.
    (No blocking that artificially reduces capacity.)
    """

    # --- PREP stage ---
    with sys.prep.request() as req_prep:
        yield req_prep
        yield env.timeout(sys.sample_prep())
    # PREP released here

    # --- OR stage ---
    with sys.or_room.request() as req_or:
        yield req_or
        yield env.timeout(sys.sample_or())
    # OR released here

    # --- REC stage ---
    with sys.rec.request() as req_rec:
        yield req_rec
        yield env.timeout(sys.sample_rec())
    # REC released here


def arrival_generator(env: simpy.Environment, sys: HospitalA3):
    while True:
        env.process(patient_process(env, sys))
        yield env.timeout(sys.sample_iat())


def monitor_process(env: simpy.Environment, sys: HospitalA3):
    """Samples entrance queue length (queue before PREP) after warmup."""
    while True:
        if env.now >= sys.cfg.warmup:
            sys.t_samples.append(env.now)
            sys.prep_q_samples.append(len(sys.prep.queue))
        yield env.timeout(sys.cfg.monitor_dt)


def run_once_a3(cfg: SimConfigA3) -> HospitalA3:
    rng = np.random.default_rng(cfg.rng_seed)
    env = simpy.Environment()

    sys = HospitalA3(env, cfg, rng)
    env.process(arrival_generator(env, sys))
    env.process(monitor_process(env, sys))
    env.run(until=cfg.sim_time)

    if cfg.save_monitor_csv:
        outdir = Path(cfg.results_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"t": sys.t_samples, "prep_queue_len": sys.prep_q_samples})
        out = outdir / f"{cfg.run_label}_seed{cfg.rng_seed}_monitor.csv"
        df.to_csv(out, index=False)

    return sys


def avg_entrance_queue(sys: HospitalA3) -> float:
    if not sys.prep_q_samples:
        return float("nan")
    return float(np.mean(np.array(sys.prep_q_samples, dtype=float)))


def run_replications_a3(cfg: SimConfigA3, n_reps: int = 10, seed0: int = 1000) -> Dict[str, Any]:
    vals: List[float] = []
    for k in range(n_reps):
        cfg_k = SimConfigA3(**{**cfg.__dict__, "rng_seed": seed0 + k})
        sys = run_once_a3(cfg_k)
        vals.append(avg_entrance_queue(sys))

    vals_np = np.array(vals, dtype=float)
    std = float(np.std(vals_np, ddof=1)) if len(vals_np) > 1 else 0.0

    return {
        "mean": float(np.mean(vals_np)),
        "std": std,
        "n": int(len(vals_np)),
        "values": vals,
    }

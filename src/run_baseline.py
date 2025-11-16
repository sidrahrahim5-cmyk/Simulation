# src/run_baseline.py
# Complete baseline simulator with:
# - P prep rooms, 1 OR, R recovery rooms
# - No-intermediate-buffers (blocking between stages)
# - Per-patient exponential service times (carried by each patient)
# - Two patient types: baseline + "light" (configurable mix via frac_light)
# - Monitoring (entrance queue length & OR utilisation) sampled every dt
# - Patient-level CSV + monitoring time-series CSV
# - Central configuration + reproducible RNG

from dataclasses import dataclass
from pathlib import Path
import simpy
import numpy as np
import pandas as pd

# ----------------- Central configuration -----------------
@dataclass
class SimConfig:
    # Time
    sim_time: float = 8 * 60           # total minutes to simulate
    monitor_dt: float = 1.0            # minutes between monitor samples
    # Resources
    p_prep: int = 3                    # P: number of preparation rooms
    p_or: int = 1                      # OR capacity (fixed to 1)
    p_rec: int = 3                     # R: number of recovery rooms
    # Arrivals & (baseline) services — means in minutes
    mean_iat: float = 25.0             # interarrival time (Exp)
    mean_prep: float = 40.0            # baseline PREP (Exp)
    mean_or: float = 20.0              # baseline OR (Exp)
    mean_rec: float = 40.0             # baseline REC (Exp)
    # Patient-mix (light surgeries)
    frac_light: float = 0.40           # fraction of arrivals that are "light"
    mean_prep_light: float = 30.0      # light PREP (Exp)
    mean_or_light: float = 12.0        # light OR (Exp)
    mean_rec_light: float = 30.0       # light REC (Exp)
    # Randomness
    rng_seed: int = 42                 # reproducibility
    # Output
    results_dir: str = "results"
    run_label: str = "baseline"

# Default config matches assignment baseline (plus a 40% light mix for sensitivity)
CFG = SimConfig()

# ----------------- Utilities -----------------
def log(env, msg: str):
    print(f"[t={env.now:6.1f}] {msg}")

# ----------------- Model classes -----------------
class HospitalSystem:
    def __init__(self, env: simpy.Environment, cfg: SimConfig, rng: np.random.Generator):
        self.env = env
        self.cfg = cfg
        self.rng = rng
        # Resource pools
        self.prep = simpy.Resource(env, capacity=cfg.p_prep)
        self.or_room = simpy.Resource(env, capacity=cfg.p_or)
        self.rec = simpy.Resource(env, capacity=cfg.p_rec)
        # Monitoring buffers (time series snapshots)
        self.t_samples: list[float] = []
        self.prep_q_samples: list[int] = []     # entrance queue length
        self.or_util_samples: list[float] = []  # OR busy fraction (count/capacity)
        # Per-patient results
        self.records: list[dict] = []

    # Exponential interarrival sampler
    def sample_iat(self) -> float:
        return self.rng.exponential(self.cfg.mean_iat)

    # Two-type service sampler (returns prep, or, rec, type_label)
    def sample_service_times(self):
        if self.rng.random() < self.cfg.frac_light:
            # light patient
            return (
                self.rng.exponential(self.cfg.mean_prep_light),
                self.rng.exponential(self.cfg.mean_or_light),
                self.rng.exponential(self.cfg.mean_rec_light),
                "light",
            )
        else:
            # baseline patient
            return (
                self.rng.exponential(self.cfg.mean_prep),
                self.rng.exponential(self.cfg.mean_or),
                self.rng.exponential(self.cfg.mean_rec),
                "baseline",
            )

class Patient:
    def __init__(self, pid: int, prep: float, op: float, rec: float, ptype: str):
        self.pid = pid
        self.prep_time = prep
        self.or_time = op
        self.rec_time = rec
        self.type = ptype  # "baseline" or "light"

        # Timestamps (filled during process)
        self.t_arrival = None
        self.t_prep_start = None
        self.t_prep_end = None
        self.t_or_start = None
        self.t_or_end = None
        self.t_rec_start = None
        self.t_exit = None

        # Derived metrics
        self.wait_prep_queue = None
        self.block_prep_to_or = None
        self.block_or_to_rec = None
        self.total_time_in_system = None

# ----------------- Processes -----------------
def patient_process(env: simpy.Environment, patient: Patient, sys: HospitalSystem):
    pid = patient.pid
    patient.t_arrival = env.now
    log(env, f"Patient {pid} arrives (type={patient.type}; "
             f"prep={patient.prep_time:.1f}, or={patient.or_time:.1f}, rec={patient.rec_time:.1f})")

    # ---- PREP (blocking to OR) ----
    with sys.prep.request() as req_prep:
        yield req_prep
        patient.t_prep_start = env.now
        patient.wait_prep_queue = patient.t_prep_start - patient.t_arrival

        log(env, f"Patient {pid} starts PREP")
        yield env.timeout(patient.prep_time)
        patient.t_prep_end = env.now
        log(env, f"Patient {pid} finished PREP (waiting for OR if full)")

        # BLOCKING: acquire OR BEFORE releasing PREP
        with sys.or_room.request() as req_or:
            t_block1_start = env.now  # equals t_prep_end
            yield req_or  # Hold PREP until OR is granted
            patient.block_prep_to_or = env.now - t_block1_start
            patient.t_or_start = env.now

            log(env, f"Patient {pid} starts OR")
            yield env.timeout(patient.or_time)
            patient.t_or_end = env.now
            log(env, f"Patient {pid} finished OR (waiting for RECOVERY if full)")

            # BLOCKING: acquire REC BEFORE releasing OR
            with sys.rec.request() as req_rec:
                t_block2_start = env.now  # equals t_or_end
                yield req_rec  # Hold OR until REC is granted
                patient.block_or_to_rec = env.now - t_block2_start
                patient.t_rec_start = env.now

                log(env, f"Patient {pid} starts RECOVERY")
                yield env.timeout(patient.rec_time)
                patient.t_exit = env.now
                patient.total_time_in_system = patient.t_exit - patient.t_arrival
                log(env, f"Patient {pid} finishes RECOVERY & exits")

                # Store record
                sys.records.append({
                    "pid": patient.pid,
                    "type": patient.type,
                    "t_arrival": patient.t_arrival,
                    "t_prep_start": patient.t_prep_start,
                    "t_prep_end": patient.t_prep_end,
                    "t_or_start": patient.t_or_start,
                    "t_or_end": patient.t_or_end,
                    "t_rec_start": patient.t_rec_start,
                    "t_exit": patient.t_exit,
                    "wait_prep_queue": patient.wait_prep_queue,
                    "block_prep_to_or": patient.block_prep_to_or,
                    "block_or_to_rec": patient.block_or_to_rec,
                    "svc_prep": patient.prep_time,
                    "svc_or": patient.or_time,
                    "svc_rec": patient.rec_time,
                    "total_time_in_system": patient.total_time_in_system,
                })

def arrival_generator(env: simpy.Environment, sys: HospitalSystem):
    pid = 1
    while True:
        prep_t, or_t, rec_t, ptype = sys.sample_service_times()
        env.process(patient_process(env, Patient(pid, prep_t, or_t, rec_t, ptype), sys))
        yield env.timeout(sys.sample_iat())
        pid += 1

def monitor_process(env: simpy.Environment, sys: HospitalSystem, dt: float):
    while True:
        sys.t_samples.append(env.now)
        sys.prep_q_samples.append(len(sys.prep.queue))  # entrance queue length
        sys.or_util_samples.append(sys.or_room.count / sys.or_room.capacity)  # busy fraction
        yield env.timeout(dt)

# ----------------- Reporting & IO -----------------
def summarize(sys: HospitalSystem):
    avg_prep_q = float(np.mean(sys.prep_q_samples)) if sys.prep_q_samples else float('nan')
    avg_or_util = float(np.mean(sys.or_util_samples)) if sys.or_util_samples else float('nan')
    n_done = len(sys.records)
    sim_hours = sys.cfg.sim_time / 60.0
    th_per_hour = n_done / sim_hours if sim_hours > 0 else float('nan')

    print("\n=== CONFIG ===")
    print(sys.cfg)
    print("\n=== MONITORING REPORT ===")
    print(f"Samples: {len(sys.t_samples)}  (dt={sys.cfg.monitor_dt} min)")
    print(f"Average entrance queue length (PREP): {avg_prep_q:.3f}")
    print(f"Average OR utilisation (busy fraction): {avg_or_util:.3f}")
    print(f"Completed patients: {n_done}  (throughput ≈ {th_per_hour:.2f} / hour)")
    print("=========================\n")

def save_results(sys: HospitalSystem):
    outdir = Path(sys.cfg.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Patient-level results
    df_pat = pd.DataFrame(sys.records)
    fname_pat = f"run_seed{sys.cfg.rng_seed}_{sys.cfg.run_label}.csv"
    out_pat = outdir / fname_pat
    df_pat.to_csv(out_pat, index=False)

    # 2) Monitoring time-series
    df_mon = pd.DataFrame({
        "t": sys.t_samples,
        "prep_queue_len": sys.prep_q_samples,
        "or_utilisation": sys.or_util_samples
    })
    fname_mon = f"run_seed{sys.cfg.rng_seed}_{sys.cfg.run_label}_monitor.csv"
    out_mon = outdir / fname_mon
    df_mon.to_csv(out_mon, index=False)

    print(f"Saved patient results to: {out_pat}")
    print(f"Saved monitoring series to: {out_mon}")

# ----------------- Runner -----------------
def run_once(cfg: SimConfig) -> HospitalSystem:
    rng = np.random.default_rng(cfg.rng_seed)
    env = simpy.Environment()
    sys = HospitalSystem(env, cfg, rng)
    log(env, "Simulation starts")
    env.process(arrival_generator(env, sys))
    env.process(monitor_process(env, sys, cfg.monitor_dt))
    env.run(until=cfg.sim_time)
    log(env, "Simulation ends")
    summarize(sys)
    save_results(sys)
    return sys

def main():
    run_once(CFG)

if __name__ == "__main__":
    main()

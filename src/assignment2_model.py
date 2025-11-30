from dataclasses import dataclass, field, replace
import random
import math
from typing import List
import simpy


@dataclass
class SimConfig:
    P: int
    R: int
    mean_interarrival: float
    mean_prep: float = 40.0
    mean_or_normal: float = 20.0
    mean_or_severe: float = 40.0
    severe_prob: float = 0.0   # 0 for original, 0.2 for twist
    mean_rec: float = 40.0
    sim_time: float = 1200.0
    warmup: float = 200.0
    monitor_dt: float = 1.0
    seed: int = 1


@dataclass
class SimSystem:
    cfg: SimConfig
    env: simpy.Environment
    prep: simpy.Resource
    or_room: simpy.Resource
    rec: simpy.Resource

    prep_q_samples: List[int] = field(default_factory=list)
    prep_idle_samples: List[int] = field(default_factory=list)
    or_blocked_samples: List[int] = field(default_factory=list)
    rec_full_samples: List[int] = field(default_factory=list)

    or_blocked: bool = False


def expovariate(rng, mean):
    return rng.expovariate(1.0 / mean)


def patient_proc(env, sys, pid, rng):
    cfg = sys.cfg

    prep_req = sys.prep.request()
    yield prep_req
    yield env.timeout(expovariate(rng, cfg.mean_prep))

    or_req = sys.or_room.request()
    yield or_req
    sys.prep.release(prep_req)

    # Choose OR time
    if rng.random() < cfg.severe_prob:
        or_time = expovariate(rng, cfg.mean_or_severe)
    else:
        or_time = expovariate(rng, cfg.mean_or_normal)

    yield env.timeout(or_time)

    needs_block = (sys.rec.count >= sys.rec.capacity)
    if needs_block:
        sys.or_blocked = True

    rec_req = sys.rec.request()
    yield rec_req

    if needs_block:
        sys.or_blocked = False

    sys.or_room.release(or_req)

    yield env.timeout(expovariate(rng, cfg.mean_rec))
    sys.rec.release(rec_req)


def arrival_generator(env, sys, rng):
    cfg = sys.cfg
    pid = 0
    while True:
        yield env.timeout(expovariate(rng, cfg.mean_interarrival))
        pid += 1
        env.process(patient_proc(env, sys, pid, rng))


def monitor(env, sys):
    cfg = sys.cfg
    while True:
        if env.now >= cfg.warmup:
            sys.prep_q_samples.append(len(sys.prep.queue))
            sys.prep_idle_samples.append(cfg.P - sys.prep.count)
            sys.or_blocked_samples.append(1 if sys.or_blocked else 0)
            sys.rec_full_samples.append(1 if sys.rec.count == sys.rec.capacity else 0)
        yield env.timeout(cfg.monitor_dt)


def mean_ci_95(samples):
    n = len(samples)
    m = sum(samples) / n
    if n < 2:
        return m, m, m
    var = sum((x - m) ** 2 for x in samples) / (n - 1)
    s = math.sqrt(var)
    t = 2.093
    half = t * s / math.sqrt(n)
    return m, m - half, m + half


def run_once(cfg):
    rng = random.Random(cfg.seed)
    env = simpy.Environment()

    sys = SimSystem(cfg=cfg,
                    env=env,
                    prep=simpy.Resource(env, capacity=cfg.P),
                    or_room=simpy.Resource(env, capacity=1),
                    rec=simpy.Resource(env, capacity=cfg.R))

    env.process(arrival_generator(env, sys, rng))
    env.process(monitor(env, sys))

    env.run(until=cfg.sim_time)
    return sys


def run_paired(a, b, label_a, label_b, n_rep=20):
    dq, didle, dblock, drecfull = [], [], [], []
    base_seed = 90000

    for i in range(n_rep):
        seed = base_seed + i
        sysA = run_once(replace(a, seed=seed))
        sysB = run_once(replace(b, seed=seed))

        nA = len(sysA.prep_q_samples)
        nB = len(sysB.prep_q_samples)

        avgA_q = sum(sysA.prep_q_samples)/nA
        avgB_q = sum(sysB.prep_q_samples)/nB

        avgA_idle = sum(sysA.prep_idle_samples)/nA
        avgB_idle = sum(sysB.prep_idle_samples)/nB

        avgA_block = sum(sysA.or_blocked_samples)/nA
        avgB_block = sum(sysB.or_blocked_samples)/nB

        avgA_rf = sum(sysA.rec_full_samples)/nA
        avgB_rf = sum(sysB.rec_full_samples)/nB

        dq.append(avgB_q-avgA_q)
        didle.append(avgB_idle-avgA_idle)
        dblock.append(avgB_block-avgA_block)
        drecfull.append(avgB_rf-avgA_rf)

    print(f"\n=== Paired twist: {label_b} minus {label_a} ===")
    print("Δ prep queue:", mean_ci_95(dq))
    print("Δ idle prep:", mean_ci_95(didle))
    print("Δ OR block:", mean_ci_95(dblock))
    print("Δ rec FULL:", mean_ci_95(drecfull))


def main():
    # baseline config (P=3,R=5)
    base = SimConfig(P=3, R=5, mean_interarrival=25, severe_prob=0.0)

    # twist scenario: 20% severe + arrival=30
    twist = SimConfig(P=3, R=5, mean_interarrival=30, severe_prob=0.2)

    run_paired(base, twist, "baseline", "twist")


if __name__ == "__main__":
    main()

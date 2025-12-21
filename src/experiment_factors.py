import random
import numpy as np

# -----------------------------
# Distribution helpers
# -----------------------------
def exp_time(mean):
    return random.expovariate(1 / mean)

def unif_time(a, b):
    return random.uniform(a, b)

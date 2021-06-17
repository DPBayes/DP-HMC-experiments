import numpy as np
from hmc import HMCParams

def params(**args):
    return HMCParams(
        tau = 0.10,
        tau_g = 0.55,
        eta = 0.006,
        L = 25,
        mass = 1,
        r_clip= 0.1,
        grad_clip = 0.05
    )

import numpy as np
from hmc import HMCParams

def params(**args):
    return HMCParams(
        tau = 0.1,
        tau_g = 0.25,
        eta = 0.00015,
        L = 12,
        mass = 1,
        r_clip= 6,
        grad_clip = 10
    )

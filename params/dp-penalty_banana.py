import numpy as np
from dp_penalty import PenaltyParams

def params(**args):
    return PenaltyParams(
        tau = 0.17,
        prop_sigma = np.repeat(0.06, 2),
        r_clip = 0.15,
        ocu = False,
        grw = False,
    )

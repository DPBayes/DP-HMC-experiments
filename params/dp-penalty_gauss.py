import numpy as np
from dp_penalty import PenaltyParams

def params(**args):
    return PenaltyParams(
        tau = 0.05,
        prop_sigma = np.repeat(0.0007, 10),
        r_clip = 10,
        ocu = False,
        grw = False,
    )

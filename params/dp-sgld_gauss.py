import numpy as np
from sgld import SGLDParams

def params(**args):
    return SGLDParams(
        subsample_size = 1000,
        clip_bound = 15,
        eta = 0.008#lambda i: 2 * (i+1)**(-1/3),
    )

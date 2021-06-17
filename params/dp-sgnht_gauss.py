import numpy as np
from sgld import SGNHTParams

def params(**args):
    return SGNHTParams(
        subsample_size = 1000,
        clip_bound = 12,
        eta = 0.2,
        A = 30
    )

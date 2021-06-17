import numpy as np
from sgld import SGNHTParams

def params(**args):
    return SGNHTParams(
        subsample_size = 1000,
        clip_bound = 0.2,
        eta = 100,
        A = 50
    )

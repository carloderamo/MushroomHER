import numpy as np


def normalize_and_clip(state, mu, sigma2):
    return np.clip((np.array(state) - mu) / (np.sqrt(sigma2) + 1e-10), -5., 5.)

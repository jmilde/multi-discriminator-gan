import numpy as np


def sample(n, seed= 0):
    """yields samples from `n` nats."""
    data = list(range(n))
    while True:
        np.random.seed(seed)
        np.random.shuffle(data)
        yield from data



def unison_shfl(*inpts, seed=0):
    """shuffels several lists while preserving their interrelation inpt:
    list of lists, all need to be the same lengh
    """
    shfl_idx = np.arange(len(inpts[0]))
    np.random.seed(seed)
    np.random.shuffle(shfl_idx)
    return [i[shfl_idx] for i in inpts]


def batch_sample(n, m, seed= 0):
    """yields `m` samples from `n` nats."""
    stream = sample(n, seed)
    while True:
        yield np.fromiter(stream, np.int, m)

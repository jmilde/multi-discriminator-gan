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


def partition(n, m, discard= False):
    """yields pairs of indices which partitions `n` nats by `m`.  if not
    `discard`, also yields the final incomplete partition.
    """
    steps = range(0, 1 + n, m)
    yield from zip(steps, steps[1:])
    if n % m and not discard:
        yield n - (n % m), n

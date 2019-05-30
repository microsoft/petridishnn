# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import numpy as np

def online_sampling(x_gen, k):
    """
    an online sampler. With a generator x_gen, return  k random samples.
    """
    sampled = []
    cnt = 0
    while True:
        try:
            x = next(x_gen)
            cnt += 1
            if np.random.uniform() * cnt <= k:
                if len(sampled) < k:
                    sampled.append(x)
                else:
                    evict_idx = np.random.choice(range(k))
                    sampled[evict_idx] = x
        except StopIteration:
            break
    return sampled


def stochastic_log_dense(n, k):
    """
    Sample k items from n options in log-dense fashion.
    """
    n_chunks = int(np.log2(n - 1)) + 2
    prob_chunks = np.power(2.0, range(n_chunks))
    prob_chunks /= float(np.sum(prob_chunks))
    if k > n_chunks:
        k = n_chunks
        # TODO fix this later
    selected_chunks = np.random.choice(
        range(n_chunks), k, replace=False, p=prob_chunks)

    chunki_to_start = [
        max(0, n - np.power(2, i)) for i in reversed(range(n_chunks))
    ]

    sampled_idx = []
    for chunki in selected_chunks:
        start = chunki_to_start[chunki]
        next_start = (
            n if chunki + 1 == n_chunks else chunki_to_start[chunki + 1])
        idx = np.random.choice(range(start, next_start))
        sampled_idx.append(idx)
    return sampled_idx
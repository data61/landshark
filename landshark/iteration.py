import numpy as np
from tqdm import tqdm
import itertools

def batch(it, batchsize, total_size):
    with tqdm(total=total_size) as pbar:
        while True:
            batch = list(itertools.islice(it, batchsize))
            if not batch:
                return
            yield batch
            pbar.update(len(batch))

def random(p, random_state):
    rnd = np.random.RandomState(random_state)
    c = np.array([True, False])
    p = np.array([p, 1.0 - p])
    while True:
        r = rnd.choice(c, p=p)
        yield r


import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

boundaries = np.array([[0.0, 1.0], [0.2, 0.5], [0.3, 0.8]])
target = 0.9


def get_rnd_numbers(boundaries, target, vscale):
    lo = boundaries[:, 0]
    hi = boundaries[:, 1]
    s = target - np.sum(lo)
    alpha_i = (0.5 * (hi-lo) / s) * vscale
    #print(np.sum(alpha_i))

    x_i = np.random.dirichlet(alpha_i, size=1)
    v_i = lo + s * x_i

    good_lo = not np.any(v_i < lo)
    good_hi = not np.any(v_i > hi)

    return good_lo, good_hi, v_i[0]


def get_more(boundaries, target):
    while True:
        a = random.uniform(0.0, 1.0)
        b = random.uniform(0.2, 0.5)
        c = target - a - b
        if (c > 0.3 and c < 0.8):
            break
    return a, b, c


vscale = 3.0

gl, gh, v = get_rnd_numbers(boundaries, target, vscale)
print((gl, gh, v, np.sum(v)))
if gl and gh:
    print("Good sample, use it")

gl, gh, v = get_rnd_numbers(boundaries, target, vscale)
print((gl, gh, v, np.sum(v)))
if gl and gh:
    print("Good sample, use it")

gl, gh, v = get_rnd_numbers(boundaries, target, vscale)
print((gl, gh, v, np.sum(v)))
if gl and gh:
    print("Good sample, use it")

l = []
for i in tqdm(range(10000)):
    # l.append(get_rnd_numbers(boundaries, target, vscale)[2])
    l.append(get_more(boundaries, target))
df = pd.DataFrame(l)
print(df)
plt.hist(df[0], histtype='step', label='a')
plt.hist(df[1], histtype='step', label='b')
plt.hist(df[2], histtype='step', label='c')
plt.legend()
plt.savefig('../temp/hist.png')
plt.show()


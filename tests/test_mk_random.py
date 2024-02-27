import itertools
from time import time

import numpy as np
from tqdm import tqdm

import panoramix
import pandas as pd


def test_get_random_number():
    """Test get_random_number()"""
    testd = {'PDF': 'weibull', 'shape': 5.0, 'loc': 1.0}
    calculated = np.median(panoramix.mk_random.get_random_number(testd, amount=1000000))
    expected = 1 * np.log(2.) ** (1/5.0)
    print(expected, calculated)
    assert np.isclose(expected, calculated, rtol=0.001)


def test_create_grid():
    pass
    to_be_gridded = {'a': {'min': 0.50, 'max': 1.00},
                     'b': {'min': 0.00, 'max': 0.05},
                     'c': {'min': 0.00, 'max': 0.10}}
    res = panoramix.get_grid(to_be_gridded, resolution=0.01, target=1.0)
    print(res)
    assert sum([len(res[k]) for k in res]) / len(res) == 66, \
        sum([len(res[k]) for k in res]) / len(res)


def GetValidParamValues(Params, constriantSum, prevVals):
    validParamValues = []
    if (len(Params) == 1):
        if (constriantSum >= Params[0][0] and constriantSum <= Params[0][1]):
            validParamValues.append(constriantSum)
        for v in validParamValues:
            print(prevVals + v)
        return
    sumOfLowParams = sum([Params[x][0] for x in range(1, len(Params))])
    sumOfHighParams = sum([Params[x][1] for x in range(1, len(Params))])
    lowEnd = max(Params[0][0], constriantSum - sumOfHighParams)
    highEnd = min(Params[0][1], constriantSum - sumOfLowParams) + 1
    if (len(Params) == 2):
        for av in range(lowEnd, highEnd):
            bv  = constriantSum - av
            if (bv <= Params[1][1]):
                validParamValues.append([av, bv])
        for v in validParamValues:
            print(prevVals + v)
        return
    for av in range(lowEnd, highEnd):
        nexPrevVals = prevVals + [av]
        subSeParams = Params[1:]
        GetValidParamValues(subSeParams, constriantSum - av, nexPrevVals)



if __name__ == "__main__":
    #test_get_random_number()
    #test_create_grid()
    p = [[0.0, 23.0],
         [22.0, 88.0],
         [0.0, 23.0],
         [0.0, 12.0],
         [0.0, 9.0],
         [0.0, 2.0],
         [0.0, 10.0],
         [0.0, 8.0],
         # [0.0],
         [0.0, 32.0],
         # [0.0]
         ]
    p = [[int(i) for i in j] for j in p]
    print(p)
    start = time()
    print(GetValidParamValues(p, 100, []))
    print(time() - start)


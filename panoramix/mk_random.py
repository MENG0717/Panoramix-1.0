#!/usr/bin/env python
#  -*- coding: utf-8 -*-
import itertools
import sys

import numpy as np
import scipy.stats
import pandas as pd
from tqdm import tqdm
import random as rnd

from panoramix import *


def get_random_number(d_in, amount=1):
    """
    Samples random numbers from a given distribution
    :param d_in: input dictionary containing the keys ['PDF', 'loc', 'shape1', 'min', 'max','mean','div']
    :param amount: amount of random numbers
    :return: list of random number(s) within the given distribution
    """
    np.random.seed()  # http://stackoverflow.com/a/12915206/2075003
    res = []
    if d_in['PDF'] == 'uniform':
        return np.random.uniform(d_in['min'], d_in['max'], size=amount)
    elif d_in['PDF'] in 'inert':
        return None
    elif d_in['PDF'] == 'log-normal':
        return scipy.stats.lognorm.rvs(d_in['shape'], scale=np.exp(d_in['loc']), size=amount)
    elif d_in['PDF'] == 'weibull':
        return np.random.weibull(d_in['shape'], size=amount) * d_in['loc']
    elif d_in['PDF'] == 'discrete_choice':
        return np.random.choice(np.array(d_in['loc']), size=amount, p=d_in['shape'])
    elif d_in['PDF'] == 'normal':
        return np.random.normal(d_in['mean'], d_in['div'], size=amount)
    else:
        raise NotImplementedError("Distribution '%s' is not implemented here" % d_in['PDF'])


def isolate_components(header, exclude=('raw material', 'citation'), full=False):
    return_components = []
    for candidate in header:
        if not full:
            candidate = candidate.split(' ')[0]
        if candidate in exclude:
            continue
        elif candidate in return_components:
            continue
        else:
            return_components.append(candidate)
    return return_components


def create_rnd_dicts(raw_mat_bounds, exclude=('unit',)):
    """
    Creates a random dictionary for the raw material – components matrix
    :param raw_mat_bounds: Data imported from the 'input.xlsx file', sheet '1_raw_material_bounds'
    :param exclude: in case certain values from raw_mat_bounds must be excluded
    :return: Dictionary with 1. raw materials, then 2. components as keys, specifying the minimum and maximum bounds
        as well as distribution. For instance,
        {'clinker, Portland cement': {'CaO': {'min': 64.0, 'max': 70.0, 'PDF': 'uniform'}, ...}
    """
    p_dict = {}
    excluded_mat = []
    excluded_col = []
    components = isolate_components(raw_mat_bounds)
    for raw_mat in raw_mat_bounds.index:
        # as soon as any value (except 'citation') is NA, then the raw material is ignored
        if raw_mat_bounds.loc[raw_mat].drop('citation').isnull().any():
            excluded_mat.append(raw_mat)
            continue
        p_dict[raw_mat] = {}
        for component in components:
            if component in exclude:
                excluded_col.append(component)
                continue
            p_dict[raw_mat][component] = {'min': raw_mat_bounds.loc[raw_mat, component + ' lower'],
                                          'max': raw_mat_bounds.loc[raw_mat, component + ' upper'],
                                          'PDF': raw_mat_bounds.loc[raw_mat, component + ' distribution']}
    if len(excluded_mat) > 0 or len(excluded_col) > 0:
        print("Ignored the following rows and columns:", excluded_mat, excluded_col)
    return p_dict


def read_database(data=settings.database, sheet_name='Full materials', exclude=('ID number', 'Special Notes', 'Citation', 'Sum Check', "")):
    """
    Creates a dictionary for the raw material – phases matrix
    :param data: Full database of phase measurements in materials
    :param sheet_name: selected materials for analysis
    :param exclude: columns to ignore in the database
    :return: Dictionary with 1. raw materials, then 2. phases as keys, specifying the minimum and maximum bounds
        as well as distribution. For instance,
        {'clinker, Portland cement': {'C3S': {'min': 64.0, 'max': 70.0, 'PDF': 'uniform'}, ...}

        NOTE: Distribution is set as uniform for now. If enough data is collected for each material,
        a distrbution fitting method could be implemented
    """
    phase_data = read_xls_data(file=data, sheet_name=sheet_name, index_col=[0])
    phase_data.index = [v.strip() for v in phase_data.index]
    phase_data.columns = [v.strip() for v in phase_data.columns]
    p_dict = {}
    components = isolate_components(phase_data, exclude, full=True)
    # get rid of extra characters
    for i in range(phase_data.shape[0]):  # iterate over rows
        for j in range(2, phase_data.shape[1]):  # iterate over columns
            if type(phase_data.iloc[i, j]) == str:     # get cell value
                phase_data.iloc[i, j] = float("NaN")
    # remove any unused phases and set rest of NaN to zero
    for c in components:
        if all(phase_data[c].isnull()):
            phase_data = phase_data.drop(columns=[c])
            components.remove(c)
    phase_data = phase_data.fillna(0)
    # sort through materials, make sure extra spaces don't matter in material name
    phase_data.index = [s.strip() for s in phase_data.index]
    mat_types = sorted(set(phase_data.index))
    for raw_mat in mat_types:
        # collect all entries for that material
        cur_mat = phase_data.loc[raw_mat]
        p_dict[raw_mat] = {}
        # iterate through phases and find min, max, etc
        for phase in components:
            # if only one entry, set values to those given
            if type(cur_mat) != pd.DataFrame:
                p_dict[raw_mat][phase] = {'min': cur_mat.loc[phase], 'max': cur_mat.loc[phase],
                                          'PDF': 'uniform'}
                continue
            # if multiple entries for a given material, get all inputs for the current phase
            vals = cur_mat.loc[:, phase]
            # otherwise get max and min of entries to set range
            p_dict[raw_mat][phase] = {'min': vals.min(), 'max': vals.max(), 'PDF': 'uniform'}
    return p_dict


def get_grid(grid_me, target=100.0):
    """

    :param grid_me: Input dictionary {'a': {'min': 0.50, 'max': 1.00}, ...}
    :param resolution:
    :param target:
    :return:
    """
    # print('here',grid_me)
    res = []
    print(grid_me)
    assert sum([grid_me[d]['min'] for d in grid_me]) < target, \
        "Sum of the lower grid limits is higher than the target value (%s vs. %s) – no feasible combination!" % \
        (sum([grid_me[d]['min'] for d in grid_me]), target)
    for par in grid_me:
        # in case no sampling range is defined,
        # skip the fixing step below because it would cause an error res[-1][-1]
        if grid_me[par]['min'] == grid_me[par]['max']:
            res.append([grid_me[par]['min']])
            continue
        else:
            res.append(list(np.arange(grid_me[par]['min'], grid_me[par]['max'], grid_me[par]['PDF'])))
        # fix numpy non-inclusive issue https://stackoverflow.com/q/10011302/2075003
        if res[-1][-1] != grid_me[par]['max']:
            res[-1].append(grid_me[par]['max'])
    result = []
    total = np.prod([len(i) for i in res])
    print(total)
    print(res)
    approx_target = target * 1.01
    for combination in tqdm(itertools.product(*res), total=total, unit_scale=True):
        # do a first check - so much faster than np.isclose
        if sum(combination) > approx_target:
            continue
        elif np.isclose(sum(combination), target):
            # print(combination, sum(combination))
            result.append(combination)
    print("Found %i combinations" % len(result))
    # print({k: res[n] for n, k in enumerate(grid_me)})
    result_df = pd.DataFrame(result)
    result_df.columns = grid_me.keys()
    return result_df

def get_grid_fast(grid_me, target=100):
    res = []
    print(grid_me)

    if all([grid_me[d]['min'] == grid_me[d]['max'] for d in grid_me]):
        return pd.DataFrame({k: [grid_me[k]['min']] for k in grid_me})

    assert sum([grid_me[d]['min'] for d in grid_me if d != 'water']) < target, \
        "Sum of the lower grid limits is higher than the target value (%s vs. %s) – no feasible combination!" % \
        (sum([grid_me[d]['min'] for d in grid_me]), target)

    for par in grid_me:
        if grid_me[par]['min'] == grid_me[par]['max']:
            res.append([grid_me[par]['min']])
        else:
            values = np.arange(grid_me[par]['min'], grid_me[par]['max'] + grid_me[par]['PDF'], grid_me[par]['PDF'])
            # If the last element is slightly greater than 'max', clip it to 'max'
            values[-1] = grid_me[par]['max']
            res.append(values)

    total = np.prod([len(i) for i in res])
    print(total)
    print(res)

    # Generate all combinations of values using itertools.product
    import itertools
    combinations = list(itertools.product(*res))
    summary = np.sum(combinations, axis=1)

    indices = np.where(np.isclose(summary, target))
    print(summary)

    # Access the combinations directly using the 'indices' variable
    result = [combinations[idx] for idx in indices[0]]
    print("Found %i combinations" % len(result))

    result_df = pd.DataFrame(result, columns=grid_me.keys())
    return result_df

def get_grid_fast_old(grid_me, target=100):
    res = []
    print(grid_me)
    # if there is no grid to be created, ie min == max
    if all([grid_me[k]['min'] == grid_me[k]['max'] for k in grid_me]):
        return pd.DataFrame(pd.Series({k: grid_me[k]['min'] for k in grid_me})).transpose()
    assert sum([grid_me[d]['min'] for d in grid_me if d != 'water']) < target, \
        "Sum of the lower grid limits is higher than the target value (%s vs. %s) – no feasible combination!" % \
        (sum([grid_me[d]['min'] for d in grid_me]), target)
    for par in grid_me:
        # in case no sampling range is defined,
        # skip the fixing step below because it would cause an error res[-1][-1]
        if grid_me[par]['min'] == grid_me[par]['max']:
            res.append([grid_me[par]['min']])
            continue
        else:
            res.append(list(np.arange(grid_me[par]['min'], grid_me[par]['max'], grid_me[par]['PDF'])))
        # fix numpy non-inclusive issue https://stackoverflow.com/q/10011302/2075003
        if res[-1][-1] != grid_me[par]['max']:
            res[-1].append(grid_me[par]['max'])
    total = np.prod([len(i) for i in res])

    print(total)
    print(res)
    grid = np.meshgrid(*res, sparse=True)
    summary = np.array(grid, dtype=object).sum(axis=0)
    indices = np.where(np.isclose(summary, target))
    result = [[row[i] for row, i in zip(res, idx)]
              for idx in zip(*indices)]
    print("Found %i combinations" % len(result))
    # print({k: res[n] for n, k in enumerate(grid_me)})
    result_df = pd.DataFrame(result)
    result_df.columns = grid_me.keys()
    return result_df



def select_rnd_combinations_from_grid(grid_data, no):
    """
    Selects random combinations from the total grid space
    :param grid_data: The total grid space, all combinations
    :param no: The number of combinations that should be returned
    :return:
    """
    assert len(grid_data) >= no
    # List of indices that will be selected.This will not contain duplicates:
    # https://stackoverflow.com/questions/16655089/python-random-numbers-into-a-list#comment23958182_16655177
    select_indices = rnd.sample(range(0, len(grid_data)), no)
    return [grid_data[i] for i in select_indices]


def get_grid_worker(args, target=100.0):
    """

    :param grid_me: Input dictionary {'a': {'min': 0.50, 'max': 1.00}, ...}
    :param resolution:
    :param target:
    :return:
    """
    n, raw_mat_name, grid_me = args
    # print([grid_me[d]['min'] for d in grid_me])
    res = []
    assert sum([grid_me[d]['min'] for d in grid_me]) < target, \
        "Sum of the lower grid limits is higher than the target value (%s vs. %s) – no feasible combination!" % \
        (sum([grid_me[d]['min'] for d in grid_me]), target)
    for par in grid_me:
        if grid_me[par]['min'] == grid_me[par]['max']:
            # in case no sampling range is defined
            res.append([grid_me[par]['min']])
            # skip the fixing step below because it would cause an error res[-1][-1]
            continue
        else:
            res.append(list(np.arange(grid_me[par]['min'], grid_me[par]['max'], grid_me[par]['PDF'])))
        # fix numpy non-inclusive issue https://stackoverflow.com/q/10011302/2075003
        if res[-1][-1] != grid_me[par]['max']:
            res[-1].append(grid_me[par]['max'])
    # print(np.prod([len(i) for i in res]))
    result = []
    total = np.prod([len(i) for i in res])
    # print(res)
    for combination in tqdm(itertools.product(*res), total=total, unit_scale=True, position=n, desc=raw_mat_name):
        if np.isclose(sum(combination), target):
            # print(combination, sum(combination))
            result.append(combination)
    result_df = pd.DataFrame(result)
    result_df.to_csv(settings.temp_folder + raw_mat_name + '.csv')
    # return {k: res[n].tolist() for n, k in enumerate(grid_me)}


def get_random_order_random_sample(d_in, target):
    """
    Draws a random sample for a random dictionary using a random order, while correcting n+1 on the fly. This is
        necessary because 'blind' random sampling
    :type d_in: input dictionary. Example: {'CaO': {'min': 64.0, 'max': 70.0, 'PDF': 'uniform'}, ...}
    :type target: Target value that must be respected by the random sample.
    :return: list of random number(s) within the given distribution, all having the sum `target`
    """
    pass


if __name__ == "__main__":
    pass
    # raw_mat_bounds = read_xls_data(file='./data/input.xlsx', sheet_name='1_raw_material_bounds',
    #                                index_col='raw material')
    # # print(create_rnd_dicts(raw_mat_bounds))
    # # sample_4a_SM(raw_mat_bounds)
    # get_random_number({'min': 0.0, 'max': 0.0, 'PDF': 'uniform'})
    # rnd_dict = create_rnd_dicts(raw_mat_bounds)
    # print(rnd_dict)
    read_database(sheet_name='select materials')


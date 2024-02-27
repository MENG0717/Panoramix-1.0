import copy
import numpy
import numpy as np

import pandas as pd

from panoramix import *
from panoramix import isolate_components


def sample_one_4a_sm(rnd_dict, no_samples=1000, correct=1/100):
    """
    Creates the random sample from a random dictionary
    :param rnd_dict: random dictionary created by mk_random.create_rnd_dicts(). For instance,
        {'clinker, Portland cement': {'CaO': {'min': 64.0, 'max': 70.0, 'PDF': 'uniform'}, ...}
    :param no_samples: amount of random samples to generate
    :param correct: Correction factor, e.g. if percentages are given as integers
    :return: Dataframe with rows being the random samples and columns being the raw materDANTX9519200ials and components as a
        multi-index
    """
    res_dict = {}
    for raw_mat in rnd_dict:
        for component in rnd_dict[raw_mat]:
            # The log10 solution
            # res_dict[(raw_mat, component)] = np.log10(get_random_number(rnd_dict[raw_mat][component],
            #                                                             amount=no_samples))
            res_dict[(raw_mat, component)] = get_random_number(rnd_dict[raw_mat][component],
                                                               amount=no_samples)
    res_df = pd.DataFrame(res_dict)
    return res_df.multiply(correct)


def grid_one_4a_sm(rnd_dict, save=settings.temp_folder):
    """
    Creates a gridded parameter space. Similar to sample_one_4a_sm(), but does not use random values.
    :param rnd_dict: random dictionary created by mk_random.create_rnd_dicts(). For instance,
        {'clinker, Portland cement': {'CaO': {'min': 64.0, 'max': 70.0, 'PDF': 'uniform'}, ...}
    :param resolution:
    :return: Dataframe with raw material – component table
    """
    res_df = get_grid_fast({k: rnd_dict[k]['mass'] for k in rnd_dict})
    if save:
        res_df.to_csv(save + '2_temp_combinations_grid.csv')
    return res_df


def old_sample_all_4a_sm(filename=settings.input_file, save=settings.temp_folder,
                     method=settings.sampling_method):
    sheets = {'2_feasible_mixes': 'raw material'}
    sheets = {'1_raw_material_bounds': 'raw material'}
              #'3_curing': 'curing condition'}
    rnd_samples = []
    for sample_me in sheets:
        raw_mat_bounds = read_xls_data(file=filename, sheet_name=sample_me, index_col=sheets[sample_me])
        rnd_dict = create_rnd_dicts(raw_mat_bounds)
        if method == 'random':
            rnd_samples.append(sample_one_4a_sm(rnd_dict, no_samples=settings.random_samples))
        elif method == 'grid':
            rnd_samples.append(grid_one_4a_sm(rnd_dict))
        if save:
            rnd_samples[-1].to_csv(save + sample_me[0] + '_non-normalized_samples.csv')
    return {'mass percent': rnd_samples[0]} #, 'curing': rnd_samples[2]}


def sample_oxide_mix_4a_sm(sheetname, index_col=[0], filename=settings.input_file, save=settings.temp_folder):
    """
    Random samples the oxides for each raw material based on the distributions defined on the sheet
    '1_raw_material_bounds' in the input file.
    :param sheetname: Sheet in the input file
    :param index_col: Name of the index column, normally '1_raw_material_bounds'
    :param filename: Name of the input file, normally 'data/input.xlsx'
    :param save: Switch to save the results. Pass a path & filename or None.
    :return: Multi-indexed Pandas DataFrame with random samples for raw material oxide composition
    """
    print("1 - sample oxides")
    oxide_bounds = read_xls_data(file=filename, sheet_name=sheetname, index_col=index_col)
    rnd_dict = create_rnd_dicts(oxide_bounds)
    # https://stackoverflow.com/q/53988740/2075003
    res = pd.DataFrame(index=range(settings.random_samples),
                       columns=pd.MultiIndex.from_arrays([[], []]))
    for rm in tqdm(rnd_dict):
        pbar = tqdm(total=settings.random_samples, desc=rm.split(',')[0])
        rm_res = pd.DataFrame()
        while len(rm_res) < settings.random_samples:
            progress = len(rm_res)
            rnd_samples = sample_one_4a_sm({rm: rnd_dict[rm]}, no_samples=10000)
            # sum all non-inerts
            df = 1 - rnd_samples.sum(axis=1, level=0)
            df.columns = [df.columns, ['inert'] * len(df.columns)]
            rnd_samples = rnd_samples.join(df).sort_index(1)
            # discard all rows that have any 'inert' below zero
            # for 'calcium sulfate' also an upper threshold applies
            if rm == 'calcium sulfate':
                rm_res = pd.concat([rm_res,
                                    rnd_samples[((rnd_samples.loc[:,
                                                  rnd_samples.columns.get_level_values(1) == 'inert'] >= 0) &
                                                 (rnd_samples.loc[:,
                                                  rnd_samples.columns.get_level_values(1) == 'inert'] <= 0.05))
                                                .all(axis=1)]])\
                    .reset_index(drop=True)
            else:
                rm_res = pd.concat([rm_res,
                                    rnd_samples[(rnd_samples.loc[:, rnd_samples
                                                 .columns.get_level_values(1) == 'inert'] >= 0).all(axis=1)]]) \
                    .reset_index(drop=True)
            # repeat
            pbar.update(len(rm_res)-progress)

        pbar.close()
        rm_res = rm_res.iloc[:settings.random_samples - 1]
        res = res.join(rm_res)

    if save:
        res.to_csv(save + '1_oxide_samples.csv')
    return res


def sample_raw_material_random(sheetname, index_col, filename=settings.input_file, save=settings.temp_folder):
    """
    Random samples the raw materials based on the distributions defined on the sheet '2_feasible_mixes' in the
    input file.
    :param sheetname: Sheet in the input file
    :param index_col: Name of the index column, normally '2_feasible_mixes'
    :param filename: Name of the input file, normally 'data/input.xlsx'
    :param save: Switch to save the results. Pass a path & filename or None.
    :return: Multi-indexed Pandas DataFrame with random samples for raw material oxide composition
    """

    raw_mat_bounds = read_xls_data(file=filename, sheet_name=sheetname, index_col=index_col)
    rnd_dict = create_rnd_dicts(raw_mat_bounds)
    # sample
    rnd_samples = sample_one_4a_sm(rnd_dict, no_samples=settings.random_samples, correct=1.0)
    # remove multiindex
    rnd_samples.columns = rnd_samples.columns.get_level_values(0)
    # correct CaSO4
    dependants = ['calcium sulfate', 'water']
    other_columns = [c for c in rnd_samples.columns if c not in dependants]
    rnd_samples['sum w/o CaSO4, H2O'] = rnd_samples[other_columns].sum(axis=1)
    # C = c/(1-c)M see calculation.md
    c = rnd_samples['calcium sulfate'] / 100
    c = c/(1-c)
    rnd_samples['calcium sulfate'] = c * rnd_samples['sum w/o CaSO4, H2O']
    # correct H2O
    other_columns = [c for c in rnd_samples.columns if c not in ['water', 'sum w/o CaSO4, H2O']]
    rnd_samples['sum w/o H2O'] = rnd_samples[other_columns].sum(axis=1)
    rnd_samples['water'] = rnd_samples['water'] / 100 * rnd_samples['sum w/o H2O']
    other_columns = [c for c in rnd_samples.columns if not c.startswith('sum')]
    rnd_samples['total'] = rnd_samples[other_columns].sum(axis=1)
    if save:
        rnd_samples.to_csv(save + '2_raw_material_rnd_samples.csv')
    return rnd_samples


def sample_raw_material_grid(sheetname, index_col=[0], filename=settings.input_file, save=settings.temp_folder):
    """
    Samples the raw materials based using a grid search based on sheet '2_feasible_mixes' in the
    input file.
    :param sheetname: Sheet in the input file
    :param index_col: Name of the index column, normally '2_feasible_mixes'
    :param filename: Name of the input file, normally 'data/input.xlsx'
    :param save: Switch to save the results. Pass a path & filename or None.
    :return: Multi-indexed Pandas DataFrame with random samples for raw material oxide composition
    """
    print("2 - creating raw materials grid")
    raw_mat_bounds = read_xls_data(file=filename, sheet_name=sheetname, index_col=index_col)
    rnd_dict = create_rnd_dicts(raw_mat_bounds)
    # grid
    # create a copy of rnd_dict w/o CaSO4 and H2O
    do_not_grid = ['calcium sulfate', 'water']
    grid_us = {k: rnd_dict[k] for k in rnd_dict if k not in do_not_grid}
    rnd_samples = grid_one_4a_sm(grid_us)
    #  Delete:
    #  rnd_samples = pd.read_csv('combinations.csv', index_col=0)
    #  rnd_samples.columns = grid_us.keys()
    # add CaSO4 and water
    for add_par in do_not_grid:
        if rnd_dict[add_par]['mass']['min'] == rnd_dict[add_par]['mass']['max']:
            # in case only one value / step defined
            CaSO4_increments = [rnd_dict[add_par]['mass']['max']]
        else:
            CaSO4_increments = list(np.arange(rnd_dict[add_par]['mass']['min'],
                                              rnd_dict[add_par]['mass']['max'],
                                              rnd_dict[add_par]['mass']['PDF']))
        # fix numpy non-inclusive issue https://stackoverflow.com/q/10011302/2075003
        if CaSO4_increments[-1] != rnd_dict[add_par]['mass']['max']:
            CaSO4_increments.append(rnd_dict[add_par]['mass']['max'])
        rnd_samples2 = pd.DataFrame(columns=rnd_samples.columns)
        for i in CaSO4_increments:
            rnd_samples[add_par] = i
            rnd_samples2 = pd.concat([rnd_samples2, rnd_samples], sort=False).reset_index(drop=True)
        rnd_samples = rnd_samples2
    #fix the water to binder ratio
    rnd_samples['water'] = (100 + rnd_samples['calcium sulfate']) * rnd_samples['water']/100

    if save:
        rnd_samples.to_csv(save + '2_raw_material_grid.csv')
    return rnd_samples


def random_select_2(raw_materials):
    """
    Randomly slecting samples from the grid search scenario space created by sample_raw_material_grid().
    :param raw_materials:
    :param save:
    :return:
    """
    select_us = [int(i) for i in get_random_number({'min': 0, 'max': len(raw_materials), 'PDF': 'uniform'},
                                                   amount=settings.random_samples)]
    if len(raw_materials) < settings.random_samples:
        print("WARNING: Less results from grid search than random samples requested (%s vs. %i)" %
              (len(raw_materials), settings.random_samples))
    raw_materials = raw_materials.iloc[select_us].reset_index(drop=True)
    return raw_materials


def add_water_in_oxides(oxide_samples, raw_materials):
    """
    Calculates the amount of water in the oxides and adds the sum as a new column to the raw materials variable.
    :param oxide_samples: variable with oxides
    :param raw_materials: variable with random sampled raw materials
    :return: raw_materials with new 'water_in_oxides' column
    """
    # TODO: Must multiply by raw material quantity!
    # water_in_oxides = oxide_samples.loc[:, pd.IndexSlice[:, 'H2O']]
    water_in_oxides = oxide_samples.loc[:, pd.IndexSlice[:, 'H2O']]
    water_in_oxides.columns = [c[0] for c in water_in_oxides.columns]
    total_water_in_oxides = raw_materials[water_in_oxides.columns] * water_in_oxides
    raw_materials.loc[:, 'water in oxides'] = total_water_in_oxides.sum(axis=1) - total_water_in_oxides['water']
    raw_materials.loc[:, 'water'] = raw_materials.loc[:, 'water'] - raw_materials.loc[:, 'water in oxides']
    return raw_materials


def add_total_cols(raw_materials, save=settings.temp_folder):
    dependants = ['calcium sulfate', 'water', 'water in oxides']
    other_columns = [c for c in raw_materials.columns if c not in dependants]
    raw_materials['sum w/o CaSO4, H2O'] = raw_materials[other_columns].sum(axis=1)
    # raw_materials H2O
    other_columns = [c for c in raw_materials.columns if c not in ['water', 'water in oxides', 'sum w/o CaSO4, H2O']]
    raw_materials['sum w/o H2O'] = raw_materials[other_columns].sum(axis=1)
    other_columns = [c for c in raw_materials.columns if not c.startswith('sum')]
    raw_materials['total'] = raw_materials[other_columns].sum(axis=1)
    raw_materials['sum w/o CaSO4, H2O, clinker'] = \
        raw_materials['sum w/o CaSO4, H2O'] - raw_materials['clinker, Portland cement']
    if save:
        raw_materials.to_csv(save + '2_raw_material_grid_selected.csv')
    return raw_materials


def sample_curing(sheetname, index_col=[0], filename=settings.input_file, save=settings.temp_folder):
    print("3 - sampling curing conditions")
    curing_bounds = read_xls_data(file=filename, sheet_name=sheetname, index_col=index_col)
    rnd_dict = create_rnd_dicts(curing_bounds)
    res = sample_one_4a_sm(rnd_dict, no_samples=settings.random_samples, correct=1.0)
    # remove multiindex
    res.columns = res.columns.get_level_values(0)
    if save:
        res.to_csv(save + '3_curing.csv')
    return res


def is_in_range(ranges, samples):
    # TODO: Delete?
    # Create a MultiIndex for upper and lower bounds
    ranges.columns = ranges.columns.str.split(' ', expand=True)
    checked = pd.DataFrame()
    for material in ranges.index:
        for element in ranges.loc[material].index.get_level_values(0):
            if element in ['citation']:
                continue
            samples[material][element].between(ranges.loc[material][element]['lower'],
                                               ranges.loc[material][element]['upper'],
                                               inclusive=True)


def mp_sample_all_4a_sm(filename=settings.input_file, save=settings.temp_folder,
                     method=settings.sampling_method):
    sheets = {'1_raw_material_bounds': 'raw material'}
    #, '2_feasible_mixes': 'raw material',
    #          '3_curing': 'curing condition'}
    mp_me = []
    for sample_me in sheets:
        raw_mat_bounds = read_xls_data(file=filename, sheet_name=sample_me, index_col=sheets[sample_me])
        rnd_dict = create_rnd_dicts(raw_mat_bounds)
        if method == 'random':
            raise NotImplementedError
            # mp_me.append(sample_one_4a_sm(rnd_dict, no_samples=settings.random_samples))
        elif method == 'grid':
            for n, raw_mat in enumerate(rnd_dict):
                mp_me.append([n, raw_mat, rnd_dict[raw_mat]])
    pool = multiprocessing.Pool(processes=cpus)
    m = multiprocessing.Manager()
    q = m.Queue()
    res = pool.map_async(get_grid_worker, mp_me)
    count = 0
    while not res.ready():
        if count < q.qsize():
            c = q.qsize() - count
            count += c
        sleep(0.2)
    pool.close()
    pool.join()
    # return {'mass percent': mp_me[0], 'mix': mp_me[1], 'curing': mp_me[2]}


def chain_sample_all_4a_sm(filename=settings.input_file, save=settings.temp_folder, method=settings.sampling_method,
                           no_samples=settings.random_samples):
    """
    STUB – NOT IMPLEMENTED
    This function was intended to do the random sampling in a randomized order. The idea was that this approach could
        overcome the problem of normalization when doing 'blind' random sampling.
    :param filename:
    :param save:
    :param method:
    :return:
    """
    raise NotImplementedError
    sheets = {'1_raw_material_bounds': 'raw material', '2_feasible_mixes': 'raw material'}
    rnd_samples = []
    for sample_me in sheets:
        raw_mat_bounds = read_xls_data(file=filename, sheet_name=sample_me, index_col=sheets[sample_me])
        rnd_dict = create_rnd_dicts(raw_mat_bounds)
        print(rnd_dict)
        for i in tqdm(range(no_samples)):
            for material in rnd_dict:
                rnd_order_keys = random.shuffle(rnd_dict[material].keys())
                fix_upper_bound = 0.0
                this_sample = []
                for compound in rnd_order_keys:
                    upper_bound = rnd_dict['max'] - fix_upper_bound
                    if upper_bound <= 0.0:
                        this_sample.append(0)
                    else:
                        this_sample.append(get_random_number({'max': fixed_upper,
                                                              'min': rnd_dict['min'],
                                                              'PDF': rnd_dict['PDF']},
                                                             amount=1))
                    fix_upper_bound = sum(this_sample)
                    print(i, material, compound, {'max': fixed_upper, 'min': rnd_dict['min'], 'PDF': rnd_dict['PDF']})
                    # {'min': 40.0, 'max': 40.0, 'PDF': 'uniform'}
            break


def normalize_4b_mass_percent(samples, save=settings.temp_folder):
    # NO! samples['mass percent'] = samples['mass percent'].div(samples['mass percent'].sum(axis=1, level=0), level=0)
    normalizer = samples['mass percent'].apply(lambda x: -np.log10(x)).replace(np.inf, 0)
    # samples['mass percent'] = samples['mass percent'].div(normalizer.sum(axis=1, level=0), level=0)
    samples['mass percent'] = normalizer.div(normalizer.sum(axis=1, level=0), level=0)
    if save:
        samples['mass percent'].to_csv(save + '1_normalized_samples.csv')
    return samples


def normalize_4b_mix(samples, name, exclude, save, remove_mix=False):
    new_df = samples['mix'].drop(exclude, axis=1).div(samples['mix'].drop(exclude,
                                                                          axis=1).sum(axis=1, level=1), level=1) * 100.0
    for ex in exclude:
        new_df[tuple(ex)] = samples['mix'].loc[:, ex]
    samples[name] = new_df
    # use np.isclose() bc of rounding error
    new_df.drop(exclude, axis=1)
    assert all(np.isclose(new_df.drop(exclude, axis=1).sum(axis=1, level=1), 100.0)), "control sum not good"
    if save:
        # save results to disk
        samples[name].to_csv(save)
    if remove_mix:
        # remove the old mix from the dataframe to save RAM
        samples.pop('mix', None)
    return samples


def classify_binder_5(samples, save=settings.temp_folder):
    rulez = get_binder_classification_rules().drop([('additional conditions 1', 'None'),
                                                    ('additional conditions 2', 'None'),
                                                    ('additional conditions 3', 'None'),
                                                    ('citation', 'None')],
                                                   axis=1)
    print("5 - classifying")
    ignore_us = ['sum w/o CaSO4, H2O', 'sum w/o H2O', 'sum w/o CaSO4, H2O, clinker',
                 'total', 'water', 'water in oxides']
    better_be_empty = [mat for mat in samples.columns if mat not in list(rulez.columns.levels[0]) + ignore_us]
    assert len(better_be_empty) == 0, "some materials were not found in classification rules: %s" % better_be_empty
    # drop columns not in samples['sum w/o H2O']
    rulez = rulez[[c for c in samples.columns if c not in ignore_us]]
    rulez = rulez.astype(float) / 100
    samples = samples.div(samples['sum w/o CaSO4, H2O'].values, axis='rows')
    if save:
        samples.to_csv(save+'5_temp_percentages.csv')
    classified = pd.DataFrame(index=samples.index,
                              columns=pd.MultiIndex(levels=[[], []], codes=[[], []],
                                                    names=['classification', 'raw material']))
    counter = 0
    for classification in tqdm(rulez.index.values):
        for mix in samples.columns:
            if mix in ignore_us:
                continue
            # print(samples['mix_w_CaSO4'].loc[:, (mix, 'mass')])
            # print(rulez.loc[classification, (mix, 'lower')], rulez.loc[classification, (mix, 'upper')])
            if all([np.isnan(rulez.loc[classification, (mix, 'lower')]),
                   np.isnan(rulez.loc[classification, (mix, 'upper')])]):
                classified[classification[1], mix] = True
            else:
                classified[classification[1], mix] = samples.loc[:, mix]\
                    .between(float(rulez.loc[classification, (mix, 'lower')]),
                            float(rulez.loc[classification, (mix, 'upper')]))
        # TODO Additional rules hardcoded for now
        # set additional condition as (0,1)
        classified[classification[1], 'additional condition 1'] = samples.loc[:, 'sum w/o CaSO4, H2O, clinker']\
            .between(0, 1)
        counter += len(classified[classified[classification[1]].all(axis=1)])
        classified[classification[1], 'all'] = classified[classification[1]].all(axis=1)
    classified = classified.loc[:, (slice(None), 'all')]
    # drop the samples that is not classified into any type
    classified = classified[classified.replace({'False': False, 'True': True}).any(1)]
    if save:
        classified.to_csv(save+'5_binder_classification.csv')
    print(counter)
    return classified


def parse_reaction_extents_6(filename=settings.input_file, save=settings.temp_folder):
    print("6 - parse extent of reaction table")
    # get the data
    xls_df = read_xls_data(filename, '6_react_extents', index_col=[0, 1, 2], header=[0, 1])
    xls_df.index.names = ['binder type', 'binder name', 'time of curing (days)']
    max_toc_days = xls_df.index.get_level_values(2).max()
    # interpolate
    binder_type_index, binder_name_index, days_index = [], [], []
    for binder_type in xls_df.index.levels[0]:
        for binder_name in xls_df.index.levels[1]:
            for i in np.arange(0, max_toc_days, 0.1):
                binder_type_index.append(binder_type)
                binder_name_index.append(binder_name)
                days_index.append(round(i+1, 1))
    interpolated = xls_df.reindex([binder_type_index, binder_name_index, days_index])
    # make sure last value exists, otherwise interpolation will not work as intended
    for i in tqdm(set([i[0:2] for i in interpolated.index])):
        for col in interpolated:
            if col[1] not in ('lower', 'upper'):
                continue
            # if the column has values and last value is NA
            if not pd.isna(interpolated.loc[i, col].all()) and \
                    pd.isna(interpolated.loc[(*i, xls_df.index.get_level_values(2).max()), col]):
                # replace last value with the highest in the column
                interpolated.loc[(*i, max_toc_days), col] = interpolated.loc[i, col].max()
    interpolated = interpolated.interpolate()
    if save:
        xls_df.to_csv(save + '6_temp_not_interpolated.csv')
        interpolated.to_csv(save + '6_reaction_extents_interpolated.csv')
    return interpolated


def sample_reaction_extent_7(raw_materials, curing, extents, save=settings.temp_folder):
    print("7 - sampling extent of reaction")
    # join raw_materials and curing
    raw_materials['time of curing'] = [int(i) for i in curing['time of curing']]
    ignore_us = ['sum w/o CaSO4, H2O', 'sum w/o H2O', 'total',
                 'sum w/o CaSO4, H2O, clinker', 'time of curing',
                 'water in oxides']
    samples_list = []
    # reindex the dataframe
    index_re = classified.index
    raw_materials = raw_materials.reindex(index_re)
    classified.index = np.arange(0, len(classified) + 0)
    raw_materials.index = np.arange(0, len(raw_materials) + 0)
    # random sample a value for each raw material
    for rsample in tqdm(raw_materials.iterrows(), total=settings.random_samples):
        d = {}
        time_of_curing = rsample[1]['time of curing']
        for rm_name, rm_value in rsample[1].iteritems():
            if rm_name in ignore_us:
                continue
            # TODO: So far only CEM I
            # look up the extent of reaction for this binder and its randomly sampled time of curing
            try:
                bounds = extents.loc['CEM', 'Portland cement (CEM I)', time_of_curing][rm_name]
            except:
                print(time_of_curing, rm_name)
            if pd.isna(bounds['lower']):
                # if no value given, it is assumed the material is inert, i.e. 0% reacts
                d[rm_name] = 0.0
            elif bounds['lower'] == bounds['upper']:
                # just saving ourselves some time here
                d[rm_name] = bounds['lower']
            else:
                # only uniform PDF for now
                eor_sample = get_random_number({'min': bounds['lower'],
                                                'max': bounds['upper'],
                                                'PDF': 'uniform'}, amount=1)[0]
                d[rm_name] = eor_sample
        samples_list.append(d)
    eor_df = pd.DataFrame(samples_list) / 100
    reacted = raw_materials * eor_df
    unreacted = raw_materials * (1 - eor_df)
    if save:
        eor_df.to_csv(save + '7_sampled_eor.csv')
        reacted.to_csv(save + '7_reacted.csv')
        unreacted.to_csv(save + '7_unreacted.csv')
        raw_materials.to_csv(save + '7_material_mix_selected_2.csv ')
    return reacted, unreacted


def calc_oxide_amount_8(reacted, unreacted, oxides, save=settings.temp_folder):
    print("8 - calculating amount of oxides")
    # never forget: https://stackoverflow.com/a/53988801/2075003
    res = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], [], []]))
    for ocol in oxides.columns:
        res[(*ocol, 'reacted')] = oxides[ocol] * reacted[ocol[0]]
        res[(*ocol, 'unreacted')] = oxides[ocol] * unreacted[ocol[0]]
        # override inerts, they never react
        if ocol[1] == 'inert':
            res[(*ocol, 'reacted')] = oxides[ocol] * 0.0
            res[(*ocol, 'unreacted')] = oxides[ocol] * (unreacted[ocol[0]] + reacted[ocol[0]])
    if save:
        res.to_csv(save + '8_oxide_amount.csv')
    return res


def sum_oxides_8(oxide_amounts, save=settings.temp_folder):
    print("8 - summing oxides")
    # sum all reacted and unreacted oxides
    # need to use transpose() because there seems to be a bug: sum(level=[1,2], axis=1) throws an error
    oxide_amounts = oxide_amounts.fillna(0)
    oxide_reacted = oxide_amounts.sum(level=[1, 2], axis=1)
    rm_oxide_content = oxide_amounts.sum(level=[0, 1], axis=1)
    total_oxide_content = oxide_amounts.sum(level=[1], axis=1)
    if save:
        oxide_reacted.to_csv(save + '8_oxide_reacted.csv')
        rm_oxide_content.to_csv(save + '8_rm_total_oxide_content.csv')
        total_oxide_content.to_csv(save + '8_sum_oxide_content.csv')
    return oxide_reacted, rm_oxide_content


def calc_lcia_score_9(rm_samples, save=settings.temp_folder, methods=['IPCC 2013, 100a']):
    """
    :param save: Location and filename of result file
    :param rm_samples: DataFrame with raw material samples, calculated by sample_raw_material_grid()
    :param methods: List with the LCIA methods to be calculated
    :return: DataFrame of LCA impact score for each mix
    """
    print("9 - calculate LCIA scores")
    impact_factors = read_xls_data(file=settings.input_file, sheet_name='9_LCIA_factors', index_col=[0])
    rm_samples2 = copy.deepcopy(rm_samples)
    rm_samples2['Baseline'] = 100.0
    rm_samples2 = rm_samples2 / 1000
    lcia_score = rm_samples2 * impact_factors.loc[methods[0]]
    lcia_score[methods[0]] = lcia_score.sum(axis=1)
    if save:
        lcia_score.to_csv(save + '9_LCIA_results.csv')
    return lcia_score


def write_gems_file_10(curing, oxide_reacted, rm_oxide_content, oxide_amounts, save=settings.temp_folder):
    """
    The GEMS input file needs to contain
        i. reacted fraction of oxides, summed for all raw materials,
        ii. the inert fraction by raw material
        iii. the unreacted fraction by raw material.
    :param curing:
    :param oxide_reacted:
    :param rm_oxide_content:
    :param save:
    :return:
    """
    print("10 - write GEMS file")
    columns = {'curing': ['temperature', 'pressure', 'relative humidity, curing',
                          'relative humidity, environment', 'duration exposed to environment before controlled curing',
                          'time of curing'],
               'oxides_summed': [('Al2O3', 'reacted'), ('CO2', 'reacted'), ('CaO', 'reacted'),
                                 ('Fe2O3', 'reacted'), ('H2O', 'reacted'), ('K2O', 'reacted'),
                                 ('MgO', 'reacted'), ('Na2O', 'reacted'), ('SO3', 'reacted'),
                                 ('SiO2', 'reacted')],
               'oxides_summed': [('Al2O3', 'reacted'), ('CO2', 'reacted'), ('CaO', 'reacted'),
                                 ('Fe2O3', 'reacted'), ('H2O', 'reacted'), ('K2O', 'reacted'),
                                 ('MgO', 'reacted'), ('Na2O', 'reacted'), ('SO3', 'reacted'),
                                 ('SiO2', 'reacted')],
               'oxides_total': []}
    res = pd.DataFrame()
    # add i. reacted oxides
    res[columns['oxides_summed']] = \
        oxide_reacted.loc[:, columns['oxides_summed']]
    # add ii. inert fractions
    res[rm_oxide_content.loc[:, (slice(None), 'inert')].columns] = \
        rm_oxide_content.loc[:, (slice(None), 'inert')]
    # add iii. unreacted amounts
    #  need to exclude inert because it is in 'unreacted'
    unreacted_amounts = oxide_amounts.loc[:, [c for c in oxide_amounts if c[1] != 'inert']].fillna(0)
    unreacted_amounts = unreacted_amounts.sum(level=[0, 2], axis=1)
    res[unreacted_amounts.loc[:, (slice(None), 'unreacted')].columns] = \
        unreacted_amounts.loc[:, (slice(None), 'unreacted')]
    # add curing conditions
    res[curing.loc[:, columns['curing']].columns] = \
        curing.loc[:, columns['curing']]
    if save:
        res.to_csv(save + '10_GEMS_input.csv')
    return res


if __name__ == "__main__":
    oxide_samples = sample_oxide_mix_4a_sm(settings.sheets[0])
    # hist_oxide_mix_samples(oxide_samples)
    # oxide_samples = get_oxides(phase_samples, "comp of phases")

    # 2. Sample raw materials
    # raw_materials = sample_raw_material_random('2_feasible_mixes', 'raw material')
    raw_materials = sample_raw_material_grid(settings.sheets[1])
    raw_materials = random_select_2(raw_materials)
    raw_materials = add_water_in_oxides(oxide_samples, raw_materials)
    raw_materials = add_total_cols(raw_materials)
    # hist_raw_material_samples(raw_materials)

    # 9. Calc LCA
    calc_lcia_score_9(raw_materials)

    # 3. Sample curing
    curing = sample_curing(settings.sheets[2])

    # 5. Classify binder
    classified = classify_binder_5(raw_materials)
    # pie_classifications(classified)

    # 6. sample extent of reaction
    extents = parse_reaction_extents_6()

    # 7. Calculate amount of reacted material
    reacted, unreacted = sample_reaction_extent_7(raw_materials, curing, extents)

    # 8. Determine amount of oxide
    oxide_amounts = calc_oxide_amount_8(reacted, unreacted, oxide_samples)
    oxide_reacted, rm_oxide_content = sum_oxides_8(oxide_amounts)

    hist_ox_samples(oxide_reacted)

    # 10. Create GEMS input
    write_gems_file_10(curing, oxide_reacted, rm_oxide_content, oxide_amounts)


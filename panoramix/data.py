from collections import OrderedDict
# TODO: Do we need this?

__author__ = "Niko Heeren"
__copyright__ = "Copyright 2018"
__credits__ = ["Niko Heeren", "Rupert Myers"]
__license__ = "TBA"
__version__ = "0.0"
__maintainer__ = "Niko Heeren"
__email__ = "niko.heeren@gmail.com"
__status__ = "raw"


from panoramix import *
import pandas as pd
import os

from panoramix import settings


def read_xls_data(file, sheet_name, index_col=None, **args):
    """
    Reads an excel file and returns it as a Pandas dataframe
    :param file:
    :param sheet_name:
    :param index_col:
    :return:
    """
    return pd.read_excel(file, sheet_name, index_col=index_col, **args)


def compare_raw_materials(raw_mat_bounds, feasible_mixes):
    """
    Making sure the raw materials specified in 1_raw_material_bounds have a match in 2_feasible_mixes
    :param raw_mat_bounds:
    :param feasible_mixes:
    :return:
    """
    for raw_mat in raw_mat_bounds.index:
        print(raw_mat)
        assert raw_mat in feasible_mixes.index, \
            "The raw materials in 1_raw_material_bounds and 2_feasible_mixes do not seem to match"


def get_raw_mat_bounds():
    pass


def get_binder_classification_rules(input_file=settings.input_file, sheetname=settings.sheets[4]):
    xls_df = read_xls_data(input_file, sheetname, index_col=[0, 1], header = [0,1])
    # Need to do magic becasue header=[0, 1] does not ****ing work as it is supposed to in df.read_excel()
    # header_level1 = [c.replace('.1', '') for c in xls_df.columns]
    # header_level2 = [v for v in xls_df.iloc[[0]].values][0]
    # levels1, levels2 = [], []
    # for h1 in header_level1:
    #    # get the positions of the levels
    #    levels1.append(list(OrderedDict.fromkeys(header_level1)).index(h1))
    # for h2 in header_level2:
    #     # get the positions of the levels
    #     levels2.append(list(OrderedDict.fromkeys(header_level2)).index(h2))
    # assert len(header_level1) == len(header_level2), (len(header_level1), len(header_level2))
    # # finally, the new multi-index
    # mi = pd.MultiIndex(levels=[list(OrderedDict.fromkeys(header_level1)), list(OrderedDict.fromkeys(header_level2))],
    #                    codes=[levels1, levels2])
    # xls_df.columns = mi
    # xls_df = xls_df.iloc[1:]
    return xls_df





if __name__ == "__main__":
    pass
    # rawmat = read_xls_data(file='../data/input.xlsx', sheet_name='1_raw_material_bounds',
    #                        index_col='raw material')
    read_database()


def collect_ternary_plot_data(path=settings.temp_folder, save=settings.temp_folder):
    gems = pd.read_csv(os.path.join(path, '11_gems_output.csv'), index_col=0)   # , skiprows=[0, 1])
    lcia_score = pd.read_csv(os.path.join(path, '9_LCIA_results.csv'), index_col=0)
    total_oxides = pd.read_csv(os.path.join(path, '8_oxide_amount.csv'), index_col=0, header=[0, 1, 2])
    total_oxides = total_oxides.sum(level=[1], axis=1)
    res = total_oxides[['Al2O3', 'CaO', 'SO3']]
    res.loc[:, 'porosity'] = gems['ideal porosity']
    res.loc[:, 'GWP'] = lcia_score['IPCC 2013, 100a']
    if save:
        res.to_csv(os.path.join(path, '11_data_for_ternary.csv'))
    return res

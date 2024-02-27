import multiprocessing

__author__ = "Niko Heeren"
__copyright__ = "Copyright 2018"
__credits__ = ["Niko Heeren", "Rupert Myers"]
__license__ = "TBA"
__version__ = "0.0"
__maintainer__ = "Niko Heeren"
__email__ = "niko.heeren@gmail.com"
__status__ = "raw"

"""
Project-wide settings and variables go here.
"""

### Inputs

# the file where the inputs like distributions and classifications are loaded from
#input_file = '../data/input_validation.xlsx'
input_file = '../data/input_CCR.xlsx'
#input_file = '/Users/n/Dropbox (Yale_FES)/Panoramix/input/input.xlsx'

database = '../data/compiled_data.xlsx'

gems_input = '../temp/10_GEMS_input.csv'
gems_out = '../temp/11_GEMS_output.xlsx'
unreacted_input = '../temp/7_unreacted.csv'
sample_mix_selected = '../temp/7_material_mix_selected_2.csv'
FT_indicator = '../temp/12_Freeze-thaw resistance indicator.csv'
FT_indicator_sum = '../temp/12_Freeze-thaw resistance indicator_sum.csv'
LCIA_sum = '../temp/9_LCIA_results_sum.csv'

# Parameter settings, hard-coded for now
DOH = .79
w_c = 0.4
v_air, v_agg, v_paste = [.04, .68, .28]


# the sheet on the input file for (5) binder classifications
sheets = ['1_raw_material_bounds', '2_feasible_mixes', '3_curing', 'sheet 4', '5_binder_classification', '6_react_extents']
data_sheets = ['1a_select_materials', '2a_feasible_mixes', '3a_curing', 'sheet 4', '5a_binder_classification',
               '6a_react_extents', 'sheet 7', '8a_phase_comp', '9a_LCIA_factors']

binder_slassification_sheet = '5_binder_classification'

inert_oxides = ['K2O', 'Na2O', 'CO2']
non_inert_oxides = ['CaO', 'SiO2', 'Al2O3', 'Fe2O3', 'MgO', 'SO3']
mix_design = ['clinker, Portland cement', 'slag, blast furnace', 'fly ash, calcareous']


### Runtime

temp_folder = '../temp/'
#temp_folder = '../out_valid/'
#temp_folder = '/Users/n/Dropbox (Yale_FES)/Panoramix/model_output/'
sampling_method= 'random'  # the other option is 'random'
random_samples = 10   # number of random samples
# grid_resolution = 1.0  # resolution for grid sampling
cpus = max(1, multiprocessing.cpu_count() - 2)  # number of CPUs to use

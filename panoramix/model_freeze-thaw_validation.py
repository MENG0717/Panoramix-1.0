import pandas as pd
import numpy as np
import ternary

from panoramix import *
from temp import *

def get_phs_vol_fraction():
    """
    Retrieves data from GEMS output, calculates the volume fractions of gel solid, gel water,
    capillary water, unreacted phases, and chemical shrinkage, and returns these as new columns
    in the DataFrame.

    :return: DataFrame with data from GEMS output and new columns for volume fractions.
    """
    # Read data from Excel
    output_mol = pd.read_excel(settings.gems_out, 'output_mol', index_col=[0])
    output_vol = pd.read_excel(settings.gems_out, 'output_vol', index_col=[0])
    OPC_base_vol = pd.read_excel(settings.gems_out, 'OPC_base_vol', index_col=[0])
    OPC_base_mol = pd.read_excel(settings.gems_out, 'OPC_base_mol', index_col=[0])
    water_vol_un = pd.read_excel(settings.gems_out, 'water_vol_un', index_col=[0])
    output_info = pd.read_excel(settings.gems_out, 'output_info', index_col=[0])

    # Calculate released water and other parameters
    water_re = output_mol.mul(water_vol_un.loc[0, :], axis='columns')
    v_water_re = water_re.sum(axis=1)
    output_info['v_re'] = v_water_re

    # Calculate the constant beta
    wc = output_info['w/b']
    a = output_info['DOR']
    p = wc / (wc + 1/3.15)
    V_gw_base = 0.19 * 3.15 * (1 - p) * a
    v_initial_base = 100 / 3.15 + wc * 100
    water_re_base = OPC_base_mol.mul(water_vol_un.loc[0, :], axis='columns').sum(axis=1)
    V_re_base = water_re_base / v_initial_base
    V_CSH_base = OPC_base_vol['CSHQ'] / v_initial_base
    beta = (V_gw_base - V_re_base) / V_CSH_base

    # Prepare DataFrame
    para = pd.concat([
        output_info,
        v_water_re.to_frame('v_water_re'),
        beta.to_frame('beta'),
        output_vol[['CSHQ']].rename(columns={'CSHQ': 'v_CSH'}),
        output_vol[['aq_gen']].rename(columns={'aq_gen': 'v_w'}),
        output_vol.drop(['aq_gen', 'IS', 'K+', 'Na+', 'OH-'], axis=1).sum(axis=1).to_frame('v_s'),
        output_vol.drop(['IS', 'K+', 'Na+', 'OH-'], axis=1).sum(axis=1).to_frame('v_reacted')
    ], axis=1)

    # Add new columns for volume fraction calculations
    para['V_gs'] = 0.0
    para['V_gw'] = 0.0
    para['V_cw'] = 0.0
    para['V_u'] = 0.0
    para['V_cs'] = 0.0
    para['V_re'] = 0.0

    # Iterating over each row for the calculations
    for index, row in para.iterrows():
        V_re = row['v_water_re'] / row['v_initial']
        V_CSH = row['v_CSH'] / row['v_initial']
        V_w = row['v_w'] / row['v_initial']
        V_s = row['v_s'] / row['v_initial']

        # Calculations for other volume fractions
        para.at[index, 'V_re'] = V_re * row['v_paste']
        para.at[index, 'V_gs'] = (V_s - V_re) * row['v_paste']
        para.at[index, 'V_gw'] = (V_re + row['beta'] * V_CSH) * row['v_paste']
        para.at[index, 'V_cw'] = (V_w - (row['beta'] * V_CSH)) * row['v_paste']
        para.at[index, 'V_u'] = (row['v_u'] / row['v_initial']) * row['v_paste']
        para.at[index, 'V_cs'] = ((row['v_initial'] - row['v_reacted'] - row['v_u']) / row['v_initial']) * row[
            'v_paste']

    return para

def get_rho_ps():
    """
    Calculates the pore solution of resistance, only considering OH-, K+, and Na+ for now
    :return: DataFrame containing rho_ps for each sample along with lambda and sigma values for each ion.
    """
    output_vol = pd.read_excel(settings.gems_out, 'output_vol', index_col=[0])
    ions = ['OH-', 'K+', 'Na+']

    ions_par = pd.DataFrame(index=ions, columns=['lambda_o', 'G'])
    ions_par.loc[:, 'lambda_o'] = [198, 73.5, 50.1]
    ions_par.loc[:, 'G'] = [0.353, 0.548, 0.733]

    # Initialize DataFrames for storing intermediate and final results
    lambda_i = pd.DataFrame(index=output_vol.index, columns=ions)
    sigma_i = pd.DataFrame(index=output_vol.index, columns=ions)
    rho_ps = pd.DataFrame(index=output_vol.index)

    # Iterating over rows
    for index, row in output_vol.iterrows():
        for ion in ions:
            lambda_val = ions_par.at[ion, 'lambda_o'] / (1 + ions_par.at[ion, 'G'] * row['IS']**0.5)
            sigma_val = row[ion] * lambda_val / 10

            lambda_i.at[index, ion] = lambda_val
            sigma_i.at[index, ion] = sigma_val

        sigma_ps = sigma_i.loc[index, :].sum()
        rho_ps.at[index, 'rho_ps'] = 1 / sigma_ps if sigma_ps != 0 else None

    return rho_ps


def calc_F_app_1(save=settings.temp_folder):
    para = get_phs_vol_fraction()
    rho_ps = get_rho_ps()

    # Merge the two DataFrames based on their index
    combined_df = pd.concat([para, rho_ps], axis=1)

    # Define new columns for results
    combined_df['Phi_paste'] = 0.0
    combined_df['phi_paste'] = 0.0
    combined_df['phi_tres'] = 0.0
    combined_df['beta_paste'] = 0.0
    combined_df['rho_gel'] = 0.0
    combined_df['rho_cap'] = 0.0
    combined_df['rho_paste'] = 0.0
    combined_df['rho_p_o'] = 0.0
    combined_df['rho_conc'] = 0.0
    combined_df['F_app'] = 0.0
    combined_df['S_matrix'] = 0.0
    combined_df['S_cr'] = 0.0
    combined_df['S_2'] = 0.0
    combined_df['t_cr'] = 0.0

    b1, b2 = 2.4, 0.85  # https://doi.org/10.1016/j.cemconres.2019.105820
    c1 = 2.1
    A = 2.55

    # Iterating over each row for calculations
    for index, row in combined_df.iterrows():
        Phi_paste = row['V_cw'] + row['V_gw'] + row['V_cs']
        phi_paste = Phi_paste / row['v_paste']
        phi_tres = row['V_gw'] / row['v_paste']
        beta_paste = 1 / (1 + b1 * ((1 - phi_paste) / (phi_paste - phi_tres) ** b2)) ** 2
        rho_gel = row['rho_ps'] * 400
        rho_cap = row['rho_ps'] / (phi_paste * beta_paste)
        rho_paste = rho_gel * rho_cap / (rho_gel + rho_cap)
        R_p_o = row['v_air'] / row['v_paste']
        rho_p_o = (1 + 0.5 * R_p_o) * rho_paste / (1 - R_p_o)
        rho_conc = (1 + 0.5 * row['v_agg']) * rho_p_o / (1 - row['v_agg'])
        F_app = rho_conc / row['rho_ps']
        S_matrix = Phi_paste / (Phi_paste + row['v_air'])
        S_cr = (87 - 10 * A * ((100 * row['v_air']) ** (-1.70))) / 100
        if S_cr < 0.85:
            S_cr = 0.85
        S_2 = c1 / (F_app ** 0.5)

        t_cr = ((S_cr - S_matrix) * (F_app ** 0.5)) ** 2 if S_cr > S_matrix else 0
        # Store results in DataFrame
        combined_df.at[index, 'Phi_paste'] = Phi_paste
        combined_df.at[index, 'phi_paste'] = phi_paste
        combined_df.at[index, 'phi_tres'] = phi_tres
        combined_df.at[index, 'beta_paste'] = beta_paste
        combined_df.at[index, 'rho_gel'] = rho_gel
        combined_df.at[index, 'rho_cap'] = rho_cap
        combined_df.at[index, 'rho_paste'] = rho_paste
        combined_df.at[index, 'rho_p_o'] = rho_p_o
        combined_df.at[index, 'rho_conc'] = rho_conc
        combined_df.at[index, 'F_app'] = F_app
        combined_df.at[index, 'S_matrix'] = S_matrix
        combined_df.at[index, 'S_cr'] = S_cr
        combined_df.at[index, 'S_2'] = S_2
        combined_df.at[index, 't_cr'] = t_cr

    if save:
        combined_df.to_csv(save + '12_Freeze-thaw resistance indicator.csv')
    return combined_df


def calc_F_app(v_paste = settings.v_paste):

    V_gs, V_gw, V_cw, V_u, V_cs, V_re, V_w = get_phs_vol_fraction()

    rho_ps = get_rho_ps()

    Phi_paste = V_cw + V_gw + V_cs
    phi_paste = Phi_paste/v_paste
    b1, b2 = 2.4, .85       # https://doi.org/10.1016/j.cemconres.2019.105820
    phi_tres = V_gw/v_paste
    beta_paste = 1/(1 + b1 * ((1-phi_paste)/(phi_paste - phi_tres)**b2))**2

    rho_gel = rho_ps*400
    rho_cap = rho_ps/(phi_paste * beta_paste)

    rho_paste = rho_gel * rho_cap/(rho_gel + rho_cap)

    #rho_conc = .5 * rho_paste * (2+settings.v_air+settings.v_agg)/(settings.v_paste)
    R_p_o = settings.v_air/settings.v_paste
    rho_p_o = (1+0.5*R_p_o)*rho_paste/(1-R_p_o)
    #rho_conc = rho_paste * (1 + 2*settings.v_air + 2*settings.v_agg) / (settings.v_paste)
    rho_conc = (1+0.5*settings.v_agg)*rho_p_o/(1-settings.v_agg)

    F_app = rho_conc/rho_ps
    return F_app, phi_tres, Phi_paste, phi_paste, beta_paste



def calc_tcr_test(save=settings.temp_folder):
    """
    Calculates the freeze-thaw resistance indicator and normalized indicator.

    """
    # Get volume fractions and create DataFrame
    V_gs, V_gw, V_cw, V_u, V_cs, V_re, V_w = get_phs_vol_fraction()
    v_water_re, beta, v_s, v_reacted, v_CSH, v_w, v_u, v_initial, v_paste, v_air, v_agg = get_output_GEMS()

    # Calculate F_app
    F_app, phi_tres, Phi_paste, phi_paste, beta_paste = calc_F_app()

    # Calculate volume fractions
    #Phi_paste = V_cw + V_gw + V_cs
    Phi_air = v_air
    S_matrix = Phi_paste / (Phi_paste + Phi_air)

    # Calculate S_cr and S2
    S_cr = 0.85
    c1 = 2.1
    S2 = c1 / F_app ** 0.5

    # Calculate freeze-thaw resistance indicator
    tcr = ((S_cr - S_matrix) / S2) ** 2
    tcr = tcr.to_frame().dropna()

    # Read sample mix design

    # Calculate normalized indicator
    tcr_n = tcr * c1 ** 2

    # Merge all relevant values into tcr DataFrame
    RV = pd.concat([
        V_gs.to_frame(name='V_gs'),
        V_gw.to_frame(name='V_gw'),
        V_cw.to_frame(name='V_cw'),
        V_u.to_frame(name='V_u'),
        V_cs.to_frame(name='V_cs'),
        V_re.to_frame(name='V_re'),
        F_app.to_frame(name='F_app'),
        phi_tres.to_frame(name='phi_tres'),
        Phi_paste.to_frame(name='Phi_paste'),
        beta_paste.to_frame(name='beta_paste'),
        S_matrix.to_frame(name='S_matrix'),
        S2.to_frame(name='S2'),
        V_w.to_frame(name='V_w')
    ], axis=1)

    # Save normalized indicator to file
    if save:
        tcr_n.to_csv(save + '12_Freeze-thaw resistance indicator.csv')
        RV.to_csv(save + '12_relevant values of F-T indicator.csv')

    return RV, tcr_n

def calc_lcia_score_9_sum(save=settings.temp_folder, methods=['IPCC 2013, 100a']):
    """
    :param save: Location and filename of result file
    :param rm_samples: DataFrame with raw material samples, calculated by sample_raw_material_grid()
    :param methods: List with the LCIA methods to be calculated
    :return: DataFrame of LCA impact score for each mix
    """
    print("9 - calculate LCIA scores")
    impact_factors = read_xls_data(file=settings.input_file, sheet_name='9_LCIA_factors', index_col=[0])
    rm_samples2 = pd.read_csv(settings.FT_indicator_sum, index_col=[0])
    rm_samples2['Baseline'] = 100.0
    rm_samples2 = rm_samples2 / 1000
    lcia_score = rm_samples2 * impact_factors.loc[methods[0]]
    lcia_score[methods[0]] = lcia_score.sum(axis=1)
    if save:
        lcia_score.to_csv(save + '9_LCIA_results_sum.csv')
    return lcia_score


if __name__ == "__main__":
    #get_phs_vol_fraction()
    #get_rho_ps()
    calc_F_app_1()
    #get_output_GEMS()
    #get_phs_vol_fraction()
    #calc_F_app()
    #calc_tcr_test()
    #contour_plot_FT()
    #contour_plot_FT_clinker()
    #contour_plot_FT_sum()
    #contour_plot_LCIA_sum()


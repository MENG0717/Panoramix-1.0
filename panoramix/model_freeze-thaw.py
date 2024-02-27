import pandas as pd
import numpy as np
import ternary

from panoramix import *
from temp import *


def get_unreacted(file=settings.database, sheetname_mat='12a_material_density'):
    """
    Get the volume of unreacted phase.
    :param inert: mass of unreacted phases
    :param dens_m: density of different phases
    :param inert_vol: dataframe of unreacted phases volume (e.g., clinker,FA,Slag...)
    :return: v_u : sum of unreated phases volume cm3
    """

    inert = pd.read_csv(settings.unreacted_input, index_col=[0])
    inert = inert.drop(['sum w/o CaSO4, H2O', 'sum w/o CaSO4, H2O, clinker', 'sum w/o H2O',
                        'time of curing', 'total', 'water in oxides'], axis = 1)
    # read density
    dens_m = pd.read_excel(file, sheetname_mat, index_col=[0])
    dens_m['Density (kg/m^3)'] = dens_m['Density (kg/m^3)'] / 1000
    dens_m = dens_m.drop(['Citation'], axis=1)
    # calculate the volume of unreacted phases
    inert_vol = pd.DataFrame(index=inert.index, columns=inert.columns)
    for i in inert.columns:
        inert_vol.loc[:, i] = inert[i]/dens_m.loc[i, 'Density (kg/m^3)']
    v_u = inert_vol.sum(axis=1)

    return v_u


def get_initial_vol(file=settings.database, sheetname_mat='12a_material_density'):
    """
    Get the total volume before the hydration.
    :param initial_m: mass of initial phases
    :param dens_m: density of different phases
    :param initial_vol: dataframe of total phases volume
    :return; v_initial : sum of initial volume cm3
    """

    initial_m = pd.read_csv(settings.sample_mix_selected, index_col=[0])
    initial_m = initial_m.drop(['sum w/o CaSO4, H2O', 'sum w/o CaSO4, H2O, clinker', 'sum w/o H2O',
                        'time of curing', 'total', 'water in oxides'], axis=1)

    # read density
    dens_m = pd.read_excel(file, sheetname_mat, index_col=[0])
    dens_m['Density (kg/m^3)'] = dens_m['Density (kg/m^3)'] / 1000
    dens_m = dens_m.drop(['Citation'], axis=1)

    # calculate the volume of unreacted phases
    initial_vol = pd.DataFrame(index=initial_m.index, columns=initial_m.columns)
    for i in initial_m.columns:
        initial_vol.loc[:, i] = initial_m[i]/dens_m.loc[i, 'Density (kg/m^3)']
    v_initial = initial_vol.sum(axis=1)

    return v_initial


def get_output_GEMS(wc=settings.w_c, a=settings.DOH):
    """
    Get the released water and the constant beta.
    :param output_mol: mol of output phases from GEMS results
    :param OPC_base_vol: volume of output of base OPC system from GEMS results
    :param OPC_base_mol: mol of output of base OPC system from GEMS results
    :return: v_water_re: sum of released water volume cm3
             beta: a constant for calculation in PPMC model
             v_s: volume of hydrated solid
             v_reacted: volume of hydrated product
             v_CSH: volume of CSH
             v_w: volume of exceeded water (GEMS output)
    """
    output_mol = pd.read_excel(settings.gems_out, 'output_mol', index_col=[0])
    output_vol = pd.read_excel(settings.gems_out, 'output_vol', index_col=[0])
    OPC_base_vol = pd.read_excel(settings.gems_out, 'OPC_base_vol', index_col=[0])
    OPC_base_mol = pd.read_excel(settings.gems_out, 'OPC_base_mol', index_col=[0])
    water_vol_un = pd.read_excel(settings.gems_out, 'water_vol_un', index_col=[0])
    #calculate released water
    water_re = pd.DataFrame(index=output_mol.index, columns=output_mol.columns)
    for i in water_vol_un.columns:
        water_re.loc[:, i] = output_mol.loc[:, i] * water_vol_un.loc[0, i]
    v_water_re = water_re.sum(axis=1)

    # calculate the constant beta
    p = wc/(wc + 1/3.15)
    V_gw_base = 0.19 * 3.15 * (1-p) * a
    v_initial_base = 100/3.15 + wc * 100
    for i in water_vol_un.columns:
        water_re_base = OPC_base_mol.loc[0, i] * water_vol_un.loc[0, i]
    V_re_base = water_re_base/v_initial_base
    # potential bug 'CSHQ'
    V_CSH_base = OPC_base_vol.loc[0, 'CSHQ']/v_initial_base
    beta = (V_gw_base - V_re_base)/V_CSH_base
    v_CSH = output_vol.loc[:, 'CSHQ']

    # hydrated solid volume
    v_s = output_vol.drop(['aq_gen', 'IS', 'K+', 'Na+', 'OH-'], axis=1).sum(axis=1)
    v_reacted = output_vol.drop(['IS', 'K+', 'Na+', 'OH-'], axis=1).sum(axis=1)
    v_w = output_vol.loc[:, 'aq_gen']

    return v_water_re, beta, v_s, v_reacted, v_CSH, v_w


def get_phs_vol_fraction(v_paste=settings.v_paste):
    """
    Get the V_gs, V_gw, V_cw, V_u, V_cs.
    :param V_gs: volume fraction of gel solid
    :param V_gw: volume fraction of gel water
    :param V_cw: volume fraction of capillary water
    :param V_u: volume fraction of unreacted phases
    :param V_cs: volume fraction of chemical shrinkage
    :return: V_gs, V_gw, V_cw, V_u, V_cs
    """
    v_u = get_unreacted()
    v_initial = get_initial_vol()
    v_water_re, beta, v_s, v_reacted, v_CSH, v_w = get_output_GEMS()

    #Normalisation
    V_re = v_water_re/v_initial
    V_CSH = v_CSH/v_initial
    V_w = v_w/v_initial
    V_s = v_s / v_initial

    #calculation
    V_gs = V_s - V_re
    V_gw = V_re + beta * V_CSH
    V_cw = V_w - (V_gw - V_re)
    V_u = v_u / v_initial
    V_cs = (v_initial - v_reacted - v_u) / v_initial
    sum = V_gs + V_gw + V_cw + V_u + V_cs

    #normalise to concrete fraction
    V_gs = V_gs * v_paste
    V_gw = V_gw * v_paste
    V_cw = V_cw * v_paste
    V_u = V_u * v_paste
    V_cs = V_cs * v_paste
    V_re = V_re * v_paste
    V_w = V_w * v_paste

    return V_gs, V_gw, V_cw, V_u, V_cs, V_re, V_w

def get_rho_ps():
    """
    Calculates the pore solution of resistance, only considering OH-, K+, and Na+ for now
    :return: a list of rho_ps for each sample
    """
    output_vol = pd.read_excel(settings.gems_out, 'output_vol', index_col=[0])
    ions = ['OH-', 'K+', 'Na+']
    IS = output_vol.loc[:, 'IS']

    ions_par = pd.DataFrame(index=ions, columns= ['lambda_o', 'G', 'lambda'])

    # lambda_o and G from https://doi.org/10.1016/S0008-8846(02)01068-2
    ions_par.loc[:, 'lambda_o'] = [198, 73.5, 50.1]
    ions_par.loc[:, 'G'] = [0.353, 0.548, 0.733]

    # calc lambda = lambda_o/(1+G*IS^.5)
    lambda_i = pd.DataFrame(index=output_vol.index, columns=ions)
    for i in ions:
        lambda_i.loc[:, i] = ions_par.loc[i, 'lambda_o']/(1+ions_par.loc[i, 'G'] * IS**.5)

    # calc sigma=sum(ci*lambdai/10)
    sigma_i = pd.DataFrame(index=output_vol.index, columns=ions)
    for i in ions:
        sigma_i.loc[:, i] = output_vol.loc[:, i] * lambda_i.loc[:, i]
    sigma_ps = sigma_i.sum(axis=1)
    rho_ps = 1/sigma_ps
    return rho_ps

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

    # Calculate F_app
    F_app, phi_tres, Phi_paste, phi_paste, beta_paste = calc_F_app()

    # Calculate volume fractions
    #Phi_paste = V_cw + V_gw + V_cs
    Phi_air = settings.v_air
    S_matrix = Phi_paste / (Phi_paste + Phi_air)

    # Calculate S_cr and S2
    S_cr = 0.85
    c1 = 2.1
    S2 = c1 / F_app ** 0.5

    # Calculate freeze-thaw resistance indicator
    tcr = ((S_cr - S_matrix) / S2) ** 2
    tcr = tcr.to_frame().dropna()

    # Read sample mix design
    sample_mix = pd.read_csv(settings.sample_mix_selected, index_col=0)

    # Calculate normalized indicator
    tcr_n = tcr * c1 ** 2
    for i in settings.mix_design:
        tcr_n.loc[:, i] = sample_mix.loc[:, i]
    tcr_n.loc[:, 'tcr_normalized'] = tcr_n.iloc[:, 0]
    tcr_n = tcr_n.drop(columns=tcr_n.columns[0], axis=1)

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

def calc_tcr(save=settings.temp_folder):
    V_gs, V_gw, V_cw, V_u, V_cs = get_phs_vol_fraction()
    F_app = calc_F_app()
    Phi_paste = V_cw + V_gw + V_cs
    Phi_air = settings.v_air

    S_matrix = Phi_paste/(Phi_paste + Phi_air)

    # Calc S_cr
    S_cr = 0.85

    # Calc S'2
    c1 = 2.1
    S2 = c1/F_app**.5

    tcr = ((S_cr - S_matrix)/S2)**2
    tcr = tcr.to_frame()
    tcr = tcr.dropna()
    tcr_n = tcr * c1 * c1
    #read sample mix design
    sample_mix = pd.read_csv(settings.sample_mix_selected, index_col=[0])
    for i in settings.mix_design:
        tcr_n.loc[:, i] = sample_mix.loc[:, i]
    tcr_n.loc[:, 'tcr_normalized'] = tcr_n.loc[:, 0]
    tcr_n.drop(columns=tcr_n.columns[0], axis=1, inplace=True)

    if save:
        tcr_n.to_csv(save + '12_Freeze-thaw resistance indicator.csv')

    return tcr, tcr_n

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
    #calc_F_app()
    calc_tcr_test()
    #contour_plot_FT()
    #contour_plot_FT_clinker()
    #contour_plot_FT_sum()
    #contour_plot_LCIA_sum()


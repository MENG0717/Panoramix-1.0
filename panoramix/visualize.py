"""
Functions to visualize things
"""

import os

import pandas as pd

from panoramix import *
from matplotlib import pyplot as plt
import ternary


def hist_oxide_mix_samples(samples):
    """
    Uses the samples from (1) / `sample_oxide_mix_4a_sm()` to plot histograms
    :param samples:
    """
    folder = os.path.join(settings.temp_folder, '1_oxide_samples_histograms/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    for raw_mat in tqdm(samples.columns.levels[0], unit='figure'):
        plt.figure()
        samples.loc[:, raw_mat].loc[:, (samples.loc[:, raw_mat]).any(axis=0)]\
            .hist(sharex=True, bins=50, range=(0, 1), figsize=(8, 8))
        plt.savefig(folder + raw_mat + '.png')
        plt.close()


def hist_raw_material_samples(samples):
    """
    Uses the samples from (1) / `sample_oxide_mix_4a_sm()` to plot histograms
    :param samples:
    """
    plt.figure()
    samples.loc[:, samples.any(axis=0)].hist(figsize=(8, 8))
    plt.xlabel(r'\textbf{g of material}', fontsize=11)

    plt.savefig(os.path.join(settings.temp_folder, '2_raw_material_samples_histogram.png'))
    plt.close()


def hist_ox_samples(samples):
    """
    Uses the samples from reacted oxides to plot histograms
    :param samples:
    """
    plt.figure()
    samples.loc[:, samples.any(axis=0)].hist(figsize=(10, 10))
    plt.xlabel(r'\textbf{g of material}', fontsize=11)

    plt.savefig(os.path.join(settings.temp_folder, '8_ox_samples_histogram.png'))
    plt.close()


def pie_classifications(classified):
    print("5 - plotting")
    folder = os.path.join(settings.temp_folder, '5_classified/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    for c in classified.columns:
        classified[c].value_counts().plot.pie()
        plt.savefig(os.path.join(folder, str(c).replace('/', '_') + '.png'))


def ternary_scatter(plot_data):
    scale = 80
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(10, 10)
    # Plot a few different styles with a legend
    plot_me = plot_data.loc[:, ['Al2O3', 'CaO', 'SO3']].values
    tax.scatter(plot_me,
                marker='s', color='red', label="Red Squares")

    tax.legend()

    tax.set_title("Scatter Plot", fontsize=20)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=5, color="blue")
    tax.ticks(axis='lbr', linewidth=1, multiple=5)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    tax.show()
    tax.savefig(os.path.join(settings.temp_folder, '11_scatter.pdf'))


def plot_dens(density, ox):

    plt.figure()
    input = pd.read_csv('../temp/10_GEMS_input.csv')
    x = input.loc[:, '(\''+ox + '\', \'reacted\')']
    x = input.loc[:, 'w/c']
    y = density
    plt.plot(x, y, 'o', color='black')
    plt.savefig(os.path.join(settings.temp_folder, '11_density_scatter.png'))
    plt.close()


def plot_t(x, y):
    plt.figure()
    plt.plot(x, y, 'o', color='black')
    plt.xlabel('Paste Porosity')
    plt.ylabel('t_cr (years)')
    plt.title('Paste Porosity vs Time to S_cr')
    plt.savefig(os.path.join(settings.temp_folder, '11_t_cr.png'))
    plt.close()


def compare(t):
    plt.figure()

    lca = pd.read_csv('../temp/9_LCIA_results.csv')
    lca = lca.loc[:, 'IPCC 2013, 100a']
    plt.plot(lca, t, 'o')
    plt.xlabel('LCIA Score')
    plt.ylabel('t_cr (years)')
    plt.title('Comparing LCIA Score with t_cr')
    plt.savefig(os.path.join(settings.temp_folder, '11_compare.png'))
    plt.close()

if __name__ == "__main__":
    data = collect_ternary_plot_data()
    ternary_scatter(data)
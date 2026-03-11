import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from src.utils.helper_functions.plot_helper_functions import golden_dimensions, make_grid
from src.mmd.distribution_functions import expected_type2_error
import torch
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

def change_ax_font(tick, fontsize):
    """
    Change axis font
    :param tick: Current set of ticks
    :param fontsize: Fontsize
    :return: Nothing
    """
    try:
        tick.label.set_fontsize(fontsize)
    except:
        try:
            tick.label1.set_fontsize(fontsize)
        except:
            pass

def plot_dist(h0_dists, h1_dists, n_atoms, alpha, filename=None, svg=False, filter_zero=False):
    """
    Plot histograms under null and alternative distribution
    :param h0_dists: List of MMD values under null hypothesis
    :param h1_dists: List of MMD values under alternative hypothesis
    :param n_atoms: Number of simulations
    :param alpha: Level of test
    :param filename: Filename to save plot. Default is None. While filename is None plot is not saved
    :param svg: Boolean indicating whether to save as a svg file. Default is False
    :param filter_zero: Boolean indicating whether to filter out MMD values below 0. Defualt is False
    :return: Nothing
    """

    fig, ax = plt.subplots(figsize=golden_dimensions(4))
    n_bins = int(20)

    if filter_zero:
        h0_dists = [d for d in h0_dists if 0 <= d]
        h1_dists = [d for d in h1_dists if 0 <= d]

    crit_val = np.sort(np.asarray(h0_dists))[int(len(h0_dists) * (1 - alpha))]

    ax_2 = ax.twinx()

    ax.set_title(f"Probability of Type 2 error: {100 * expected_type2_error(torch.tensor(h1_dists), crit_val):.2f}%",
                 fontsize="12")
    ax.hist(h0_dists, bins=n_bins, color="dodgerblue", alpha=0.6, label=r"$H_0$", density=True, edgecolor="none")
    ax_2.hist(h1_dists, bins=n_bins, color="tomato", alpha=0.5, label=r"$H_1$", density=True, edgecolor="none")
    ax.axvline(crit_val, linewidth=2, linestyle='--', color='black', alpha=0.8, label=r'Critical Value')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax_2.spines['right'].set_visible(False)
    ax_2.spines['left'].set_visible(False)
    ax_2.spines['top'].set_visible(False)
    ax_2.get_yaxis().set_ticks([])
    l1, t1 = ax.get_legend_handles_labels()
    l2, t2 = ax_2.get_legend_handles_labels()
    ax.legend(l1 + l2, t1 + t2, fontsize="12")
    if filename is not None:
        if svg:
            plt.savefig(f'{filename}', bbox_inches='tight', format='svg', dpi=1200)
        else:
            plt.savefig(f'{filename}', bbox_inches='tight')



def plot_dist_boxen(df, x, y, hue='', crit_val=None, title='', y_label='', x_label='', filename=None,
                    svg=True, loc='best', log_scale=False, showfliers=True, palette='tab10'):
    """
    Plot distributions as Boxen plot
    :param df: DataFrame of MMD values
    :param x: DataFrame column name of x coordinate value
    :param y: DataFrame column name of y coordinate value
    :param hue: Plot hue. Default is ''
    :param crit_val: Critical value. Default is None
    :param title: Plot title. Default is ''
    :param y_label: y label text. Default is ''
    :param x_label: x label test. Default is ''
    :param filename: Filename to save plot. Default is None. While filename is None plot is not saved
    :param svg: Boolean indicating whether to save as a svg file. Default is False
    :param loc: Legend location. Default is 'best'
    :param log_scale: Boolean indicating whether to use log scale. If value is True, MMD values <= 0 are excluded.
                      Default is False
    :param showfliers: Boolean indicating whether to show outliers. Default is True
    :param palette: Colour palette. Default is 'tab10'
    :return: Nothing
    """

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f'{title}', fontsize='20')
    sns.set_style('white')
    if hue == '':
        b = sns.boxenplot(data=df, x=f'{x}', y=f'{y}', ax=ax, showfliers=showfliers, palette=palette)
    else:
        b = sns.boxenplot(data=df, x=f'{x}', y=f'{y}', hue=f'{hue}', ax=ax, showfliers=showfliers, palette=palette)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(f'{x_label}', fontsize='20')
    ax.set_ylabel(f'{y_label}', fontsize='20')
    if crit_val is not None:
        ax.axhline(crit_val, label='Critical Value', color='black', linestyle='--', linewidth=4)
    if log_scale:
        b.set(yscale="log")
    ax.legend(loc=loc, fontsize='15')
    for tick in ax.xaxis.get_major_ticks():
        change_ax_font(tick, 20)
    for tick in ax.yaxis.get_major_ticks():
        change_ax_font(tick, 15)
    if filename is not None:
        if svg:
            plt.savefig(f'{filename}', bbox_inches='tight', format='svg', dpi=1200)
        else:
            plt.savefig(f'{filename}', bbox_inches='tight')


def plot_level_contributions(h0_Mk_vals, h1_Mk_vals, n_atoms, ks=[1, 2, 3, 4], filename=None, div=64, scientific=False,
                             svg=False, filter=True, hspace=0.5):
    """
    Plot level contributions
    :param h0_Mk_vals: Array of level contributions under null hypothesis
    :param h1_Mk_vals: Array of level contributions under alternative hypothesis
    :param n_atoms: Number of simulations
    :param ks: Levels considered. Default is [1, 2, 3, 4] (i.e. the first 4 levels of the MMD)
    :param filename: Filename to save plot. Default is None. While filename is None plot is not saved
    :param div: Denominator used to calculate the number of bins. Default is 64
    :param scientific: Boolean indicating whether to use scientific notation. Default is False
    :param svg: Boolean indicating whether to save as a svg file. Default is False
    :param filter: Boolean indicating whether to filter out outliers and values <= 0. Default is True
    :param hspace: h_space matplotlib parameter. Controls height spacing between subplots. Default value is 0.5
    :return: Nothing
    """

    fig, ax = plt.subplots(2, int(len(ks) / 2), figsize=(len(ks) * 5, 10))

    n_bins = int(n_atoms / div)

    plt.subplots_adjust(hspace=hspace)

    k = 0
    for j in range(int(len(ks) / 2)):
        for i, axi in enumerate(ax[j, :]):

            if filter:
                q_h0 = np.quantile(h0_Mk_vals[k], 0.99)
                q_h1 = np.quantile(h1_Mk_vals[k], 0.99)

                h0_Mk_vals_new = [d for d in h0_Mk_vals[k] if 0 < d < q_h0]
                h1_Mk_vals_new = [d for d in h1_Mk_vals[k] if 0 < d < q_h1]
            else:
                h0_Mk_vals_new = h0_Mk_vals[k]
                h1_Mk_vals_new = h1_Mk_vals[k]

            axi_2 = axi.twinx()

            axi.hist(h0_Mk_vals_new, bins=n_bins, color="dodgerblue", alpha=0.5, label=r"$H_0$", density=True,
                     edgecolor="none")
            axi_2.hist(h1_Mk_vals_new, bins=n_bins, color="tomato", alpha=0.5, label=r"$H_1$", density=True,
                       edgecolor="none")

            l1, t1 = axi.get_legend_handles_labels()
            l2, t2 = axi_2.get_legend_handles_labels()
            axi.legend(l1 + l2, t1 + t2, fontsize="20")
            # axi.set_title(rf'$\Gamma^\phi_{ks[k]}$', fontsize=15)
            axi.set_title(f'Level {ks[k]}', fontsize=20)
            axi.set_ylabel('')
            axi_2.set_ylabel('')
            axi.spines['right'].set_visible(False)
            axi.spines['left'].set_visible(False)
            axi.spines['top'].set_visible(False)
            axi.get_yaxis().set_ticks([])
            axi_2.spines['right'].set_visible(False)
            axi_2.spines['left'].set_visible(False)
            axi_2.spines['top'].set_visible(False)
            axi_2.get_yaxis().set_ticks([])
            for tick in axi.xaxis.get_major_ticks():
                change_ax_font(tick, 20)

            if scientific:
                axi.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
                axi.xaxis.set_major_locator(plt.MaxNLocator(6))
            k += 1
    if filename is not None:
        if svg:
            plt.savefig(f'{filename}', bbox_inches='tight', format='svg', dpi=1200)
        else:
            plt.savefig(f'{filename}', bbox_inches='tight')

def plot_type2_error(type2_list, scalings, n_paths_list, svg=True, filename=None, title='',
                     colors=['magenta', 'green', 'darkorange', 'blue']):
    """
    Plot the probability of a Type 2 error occurring
    :param type2_list: List of probabilities
    :param scalings: List of scalings
    :param n_paths_list: List of batch sizes
    :param svg: Boolean indicating whether to save as a svg file. Default is False
    :param filename: Filename to save plot. Default is None. While filename is None plot is not saved
    :param title: Plot title. Default is ''
    :param colors: List of colours. Default is ['magenta', 'green', 'darkorange', 'blue']
    :return: Nothing
    """

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, n_paths in enumerate(n_paths_list):

        t2e = []
        for j in range(100):
            t2e.append(type2_list[j][n_paths])

        t2e_mean = np.mean(np.asarray(t2e), axis=0)
        t2e_std = np.std(np.asarray(t2e), axis=0)
        ax.plot(scalings, t2e_mean, alpha=1, label=f'{n_paths}', color=colors[i])
        ax.fill_between(scalings, t2e_mean - t2e_std, t2e_mean + t2e_std, color=colors[i], alpha=0.3, edgecolor='none')

    ax.set_ylabel(r"P[Type 2 Error] (%)", fontsize=15)
    ax.set_xlabel("Scaling", fontsize=15)
    ax.legend(loc='best', fontsize=15)
    ax.grid(True, color='black', alpha=0.2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.title(f'{title}', fontsize=15)
    if filename is not None:
        if svg:
            plt.savefig(f'{filename}', bbox_inches='tight', format=f'svg', dpi=1200)
        else:
            plt.savefig(f'{filename}', bbox_inches='tight')

def plot_type1_error(type1_list, scalings, n_paths_list, svg=True, filename=None, title='',
                     colors = ['magenta', 'green', 'darkorange', 'blue']):

    """
    Plot the probability of a Type 1 error occurring
    :param type1_list: List of probabilities
    :param scalings: List of scalings
    :param n_paths_list: List of batch sizes
    :param svg: Boolean indicating whether to save as a svg file. Default is False
    :param filename: Filename to save plot. Default is None. While filename is None plot is not saved
    :param title: Plot title. Default is ''
    :param colors: List of colours. Default is ['magenta', 'green', 'darkorange', 'blue']
    :return: Nothing
    """

    fig, ax = plt.subplots(figsize=(16, 8), ncols=2, nrows=2)
    p = 0
    q = 0
    counter = 0
    for i, n_paths in enumerate(n_paths_list):

        t1e_new = []
        for j in range(0, 100):
            t1e_new.append(100 - np.asarray(type1_list[j][n_paths]))

        bp = ax[p, q].boxplot(np.asarray(t1e_new)[:, 1::2], patch_artist=True, labels=np.round(scalings[1::2], 2))

        for patch in bp['boxes']: patch.set_facecolor(colors[counter])
        for patch in bp['boxes']: patch.set_alpha(0.3)

        for median in bp['medians']: median.set(color=colors[counter], linewidth=3)

        for whisker in bp['whiskers']: whisker.set(color=colors[counter], linewidth=2.5, linestyle=":")
        for cap in bp['caps']: cap.set(color=colors[counter], linewidth=3)

        for flier in bp['fliers']: flier.set(markeredgecolor=colors[counter], markerfacecolor=colors[counter],
                                             alpha=0.75)

        ax[p, q].set_xlabel('Scaling', fontsize=12)
        ax[p, q].set_ylabel(r"P[Type 1 Error] (%)", fontsize=12)
        ax[p, q].set_title(f'Batch Size: {n_paths}', fontsize=12)
        plt.setp(ax[p, q].get_xticklabels(), fontsize=10)
        plt.setp(ax[p, q].get_yticklabels(), fontsize=10)
        ax[p, q].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if counter == 0:
            p = 0
            q = 1
        elif counter == 1:
            p = 1
            q = 0
        else:
            p = 1
            q = 1
        counter += 1


    fig.suptitle(f'{title}', fontsize=15, y=1.0)
    plt.subplots_adjust(hspace=0.3)
    if filename is not None:
        if svg:
            plt.savefig(f'{filename}', bbox_inches='tight', format=f'svg', dpi=1200)
        else:
            plt.savefig(f'{filename}', bbox_inches='tight')

def plot_aggregate_type1_error(type1_list, n_paths_list, svg=True, filename=None, title='',
                               colors = ['magenta', 'green', 'darkorange', 'blue']):

    """
    Plot the probability of a Type 1 error occurring aggregated across all scalings
    :param type1_list: List of probabilities
    :param n_paths_list: List of batch sizes
    :param svg: Boolean indicating whether to save as a svg file. Default is False
    :param filename: Filename to save plot. Default is None. While filename is None plot is not saved
    :param title: Plot title. Default is ''
    :param colors: List of colours. Default is ['magenta', 'green', 'darkorange', 'blue']
    :return: Nothing
    """

    t1e = []
    for i, n_paths in enumerate(n_paths_list):
        t1e.append([])
        for j in range(100):
            t1e[i].append(100 - np.asarray(type1_list[j][n_paths]))
        t1e[i] = np.asarray(t1e[i])[:, 1:].flatten()

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(t1e, patch_artist=True, labels=n_paths_list)

    for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color)
    for patch in bp['boxes']: patch.set_alpha(0.3)

    for median, color in zip(bp['medians'], colors): median.set(color=color, linewidth=3)

    for whisker, color in zip(bp['whiskers'], np.repeat(np.asarray(colors), 2)): whisker.set(color=color, linewidth=2.5,
                                                                                             linestyle=":")
    for cap, color in zip(bp['caps'], np.repeat(np.asarray(colors), 2)): cap.set(color=color, linewidth=3)

    for flier, color in zip(bp['fliers'], colors): flier.set(markeredgecolor=color, markerfacecolor=color, alpha=0.75)

    ax.set_xlabel('Sample Size', fontsize=15)
    ax.set_ylabel(r"P[Type 1 Error] (%)", fontsize=15)
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.title(f'{title}', fontsize=15)
    if filename is not None:
        if svg:
            plt.savefig(f'{filename}', bbox_inches='tight', format=f'svg', dpi=1200)
        else:
            plt.savefig(f'{filename}', bbox_inches='tight')
import os
import pandas as pd
from sklearn import metrics
import scipy.stats as ss
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from natsort import natsorted
from plots import save_plot

plots_dir = os.path.join(os.getcwd(), 'plots')
os.makedirs(plots_dir, exist_ok=True)


def merge_csv_files(indices, files_list, path=None):
    """
    :param indices: list, numpy, list of indices
    :param path: str path to csv files
    :param files_list: str list of csv files
    :return: a data frame
    """

    file_list = natsorted(files_list)
    anomaly_df = pd.DataFrame(index=indices)

    for f in file_list:
        temp_df = pd.read_csv(os.path.join(path, f))
        col_name = f.strip('.csv')[-4:]
        anomaly_df[col_name] = temp_df.Anomaly.values

    return anomaly_df


def concat_dataframes(df_list):
    """
    :param df_list: a list of dataframes to concat files
    :return: a concated dataframe
    """

    indices = df_list[0].index
    for df in df_list:
        df.reset_index(drop=True, inplace=True)

    concat_df = pd.concat(df_list, axis=1)

    concat_df.set_index(indices, drop=True, inplace=True)

    return concat_df


def friedman_conover_comparison(df, var_name='model', value_name='anomaly', plot_name=None,
                                outdir=plots_dir):
    """
    :param df: dataframe with the data to campare -models
    :param var_name: str name of to group the variable - models
    :param value_name: str name of the variable with the sults - response variable
    :param plot_name: str name of the plot
    :param outdir: str directory to save the plot
    :return: pvalues and statitic for the comparison
     Note: this code is adapted from scikit-posthoc tutorial
    https://scikit-posthocs.readthedocs.io/en/latest/tutorial.html
    """
    heatmap_args = {'linewidths': 0.5, 'linecolor': 'k', 'clip_on': False, 'square': True,
                    'cbar_ax_bbox': [0.85, 0.35, 0.04, 0.3]}

    df_ = df.rename_axis('cv_fold').melt(var_name=var_name, value_name=value_name,
                                         ignore_index=False).reset_index()

    avg_rank = df_.groupby('cv_fold')[value_name].rank(pct=True).groupby(df_[var_name]).mean()

    stat, p_value = ss.friedmanchisquare(*df.values.T)

    if p_value < 0.05:
        print(f'p_value : {str(p_value)}, we can reject the null hypothesis H0 with 95% certainty')
        print(' ')
        print('Post-hoc Conover-Friedman multiple comparison is applied')

        significance = sp.posthoc_conover_friedman(df, p_adjust='holm')
        sp.sign_plot(significance, **heatmap_args)
        save_plot(outdir=outdir, plot_name=f'posthoc_{plot_name}.png')
        plt.show()
        print(' ')
        plot_critical_difference(significance, avg_rank, plot_name=plot_name)

    return stat, p_value, avg_rank


def plot_critical_difference(posthoc=None, ranks_df=None, plot_name=None, outdir=plots_dir):
    """
    :param ranks_df: dataframe with ranks
    :param posthoc: dataframe/ dict with the results from the posthoc comparison
    :param plot_name: str name to save the plot
    :param outdir: str path to save the plots
    :return: show the critical difference plot
    Note: this code is adapted from scikit-posthoc tutorial
    https://scikit-posthocs.readthedocs.io/en/latest/tutorial.html
    """
    plt.figure(figsize=(10, 2), dpi=100)
    plt.title('Critical difference diagram of average score ranks')
    sp.critical_difference_diagram(ranks_df, posthoc)
    save_plot(outdir=outdir, plot_name=f'critical_dif_{plot_name}.png')

    return plt.show()


def compare_clusters_metrics(df_raw, df_cluster):
    """
    :param df_raw: raw data dataframe
    :param df_cluster: dataframe with cluster labels
    :return: data frame with the cluster metrics scores
    """
    model_scores = {}
    for col in df_cluster.columns:
        sil = metrics.silhouette_score(df_raw, df_cluster[col])
        cal_har = metrics.calinski_harabasz_score(df_raw, df_cluster[col])
        dav_boul = metrics.davies_bouldin_score(df_raw, df_cluster[col])
        model_scores[col] = [sil, cal_har, dav_boul]

    cluster_metrics = pd.DataFrame.from_dict(model_scores).T
    cluster_metrics.columns = ['silhoutte', 'calinski_harabasz', 'davies_bouldin']

    return  cluster_metrics

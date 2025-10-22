import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
import os
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, balanced_accuracy_score, confusion_matrix
import warnings
warnings.simplefilter("ignore", UserWarning)

from actinet.utils.utils import ACTIVITY_LABELS_DICT


class DivDict(dict):
    """Dictionary subclass that allows division by a number."""
    def __truediv__(self, n):
        if not isinstance(n, (int, float)):  
            raise TypeError("Can only divide by a number (int or float)")
        return DivDict({k: v / n for k, v in self.items()})


def calculate_metrics(y_true, y_pred):
    """Calculates accuracy, F1, Cohen's Kappa, and balanced accuracy."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    return accuracy, f1, kappa, bacc


def plot_confusion_matrix(y_true, y_pred, activity_labels, title="", ax=None, figsize=(8, 8), fontsize=20):
    """Plots a confusion matrix with heatmap annotations."""


    cm = confusion_matrix(y_true, y_pred, labels=activity_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    annotations = np.array([[f'{p:.3f}\n({n:,})' for p, n in zip(row_norm, row)] for row_norm, row in zip(cm_norm, cm)])
    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=800)

    sns.heatmap(cm_norm, annot=annotations, fmt='', cmap="Blues", cbar=False, 
                annot_kws={"size": fontsize*0.9}, ax=ax)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    ax.set_xticks(np.arange(len(activity_labels)) + 0.5)
    ax.set_yticks(np.arange(len(activity_labels)) + 0.5)
    ax.set_xticklabels(activity_labels, fontsize=fontsize*0.8)
    ax.set_yticklabels(activity_labels, fontsize=fontsize*0.8)


def build_confusion_matrix_data(results: pd.DataFrame, age_band=None, sex=None):
    """Extracts ground truth and predicted labels based on filtering conditions."""
    model_results = results.copy()
    if age_band is not None:
        model_results = model_results[model_results['Age Band'] == age_band]
    if sex is not None:
        model_results = model_results[model_results["Sex"] == sex]

    y_true_bbaa = np.hstack(model_results.loc[model_results["Model"]=="accelerometer", 'True'])
    y_pred_bbaa = np.hstack(model_results.loc[model_results["Model"]=="accelerometer", 'Predicted'])
    y_true_actinet = np.hstack(model_results.loc[model_results["Model"]=="actinet", 'True'])
    y_pred_actinet = np.hstack(model_results.loc[model_results["Model"]=="actinet", 'Predicted'])

    population = len(model_results['Participant'].unique())
    
    return y_true_bbaa, y_pred_bbaa, y_true_actinet, y_pred_actinet, population


def plot_and_save_fig(fig, save_path=None):
    """Displays and optionally saves the figure as a PDF."""
    plt.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, format='pdf', dpi=800, bbox_inches='tight')


def generate_confusion_matrices(results_df, activity_labels, group_by=None, save_path=None, fontsize=20):
    """Generates and plots confusion matrices for different subgroups.""" 
    if group_by is None:  # Full population
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 6), dpi=800)
        fig.suptitle("Confusion matrices for full Capture-24 population\nusing 5-fold group cross-validation", 
                     fontsize=fontsize)
        
        y_true_bbaa, y_pred_bbaa, y_true_actinet, y_pred_actinet, _ = build_confusion_matrix_data(results_df)
        
        plot_confusion_matrix(y_true_bbaa, y_pred_bbaa, activity_labels, 'Baseline', ax=axs[0], fontsize=fontsize*0.8)
        plot_confusion_matrix(y_true_actinet, y_pred_actinet, activity_labels, 'ActiNet', ax=axs[1], fontsize=fontsize*0.8)
        fig.tight_layout()
        plot_and_save_fig(fig, save_path)
    
    else:
        unique_groups = results_df[group_by].cat.categories
        fig = plt.figure(figsize=(12, 6*len(unique_groups)), dpi=800, constrained_layout=True)
        fig.suptitle(f"Confusion matrices for different {group_by.lower()} in " +
                     "Capture-24 population\nusing 5-fold group cross-validation", fontsize=fontsize)

        subfigs = fig.subfigures(nrows=len(unique_groups), ncols=1)
        
        for group, subfig in zip(unique_groups, subfigs):
            y_true_bbaa, y_pred_bbaa, y_true_actinet, y_pred_actinet, population =\
                build_confusion_matrix_data(results_df, **{group_by.replace(' ', '_').lower(): group})

            subfig.suptitle(f"{group_by}: {group} (n = {population})", fontsize=fontsize*0.9)
            axs = subfig.subplots(nrows=1, ncols=2, sharey=True)

            plot_confusion_matrix(y_true_bbaa, y_pred_bbaa, activity_labels, 'Baseline', ax=axs[0], fontsize=fontsize*0.8)
            plot_confusion_matrix(y_true_actinet, y_pred_actinet, activity_labels, 'ActiNet', ax=axs[1], fontsize=fontsize*0.8)

        plot_and_save_fig(fig, save_path)


def plot_model_performance(results, metric='Macro F1', modulus=0):
    """Plots a boxplot of model performance with optional participant-wise lines."""
    plt.figure(figsize=(10, 6), dpi=1000)
    with sns.color_palette("Set1"):
        sns.boxplot(x="Model", y=metric, hue="Model", data=results, width=0.3, showfliers=False)
    sns.stripplot(x="Model", y=metric, data=results, 
                  jitter=True, color='black', alpha=0.3)

    model1 = results[results['Model'] == results["Model"].unique()[0]][metric]
    model2 = results[results['Model'] == results["Model"].unique()[1]][metric]
    _, p_value = stats.ttest_rel(model1, model2)

    i = 0
    for pid in results['Participant'].unique():
        if modulus and i%modulus == 0:
            pid_df = results[results['Participant'] == pid]
            plt.plot(pid_df['Model'], pid_df[metric], marker='', 
                     linestyle='-', color='grey', alpha=0.3)
        i += 1

    plt.title(f"Comparison of model {metric}\np-value from paired t-test: {p_value:.2g}", fontsize=16)


def plot_difference_boxplots(df):
    """Plots boxplots of the differences in performance between actinet and accelerometer"""
    metrics = ['Accuracy', 'Macro F1', 'Cohen Kappa']
    differences = []
    
    for metric in metrics:
        model1 = df[df['Model'] == 'accelerometer'][metric].reset_index(drop=True)
        model2 = df[df['Model'] == 'actinet'][metric].reset_index(drop=True)
        difference = model2 - model1
        differences.append(pd.DataFrame({
            'Metric': metric,
            'Difference': difference
        }))
    
    diff_df = pd.concat(differences)
    
    plt.figure(figsize=(10, 6), dpi=800)
    sns.boxplot(x='Difference', y='Metric', hue='Metric', data=diff_df, orient='h')
    
    for metric in metrics:
        metric_data = diff_df[diff_df['Metric'] == metric]['Difference']
        mean_diff = metric_data.mean()
        std_diff = metric_data.std()
        plt.text(x=metric_data.quantile(.25)-0.01, y=metrics.index(metric)-0.3, 
                 s=f'Mean: {mean_diff:.3f}\nStd: {std_diff:.3f}', 
                 ha='right', va='center', color='black', fontsize=10, weight='bold')

    plt.title('Difference between Actinet and Accelerometer Models for Each Metric')
    plt.xlabel('Performance difference (actinet - accelerometer)')
    plt.ylabel('Metric')
    plt.savefig("outputs/har_activity_performance_difference_boxplots.png", bbox_inches='tight', dpi=800)
    plt.show()


def plot_boxplots(df, x, y='Macro F1', hue='Model'):
    """Plots boxplots of model performance by a specified variable"""
    _, ax = plt.subplots(figsize=(5, 3), dpi=1000)
    with sns.color_palette("Set1"):
        sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set_xlabel("Age Band")
    ax.set_ylabel(f"{y} Score")
    plt.title(f"{y} by {x}")
    plt.show()


def extract_activity_predictions(results: pd.DataFrame, activity, age_band=None, sex=None, return_true_labels=False):
    """Extracts incidence of predicted activity label for actinet and accelerometer based on filtering conditions."""    
    model_results = results.copy()
    if age_band is not None:
        model_results = model_results[model_results['Age Band'] == age_band]
    if sex is not None:
        model_results = model_results[model_results["Sex"] == sex]

    activity_bbaa_pred = np.array([x.get(activity, 0) for x in model_results.loc[model_results["Model"] == "accelerometer", "Pred_dict"]])
    activity_actinet_pred = np.array([x.get(activity, 0) for x in model_results.loc[model_results["Model"] == "actinet", "Pred_dict"]])

    population = len(model_results['Participant'].unique())

    if return_true_labels:
        activity_bbaa_true = np.array([x.get(activity, 0) for x in model_results.loc[model_results["Model"] == "accelerometer",
                                                                                     "True_dict"]])
        activity_actinet_true = np.array([x.get(activity, 0) for x in model_results.loc[model_results["Model"] == "actinet", 
                                                                                        "True_dict"]])
        return activity_bbaa_pred, activity_actinet_pred, activity_bbaa_true, activity_actinet_true, population
    
    else:
        return activity_bbaa_pred, activity_actinet_pred, population


def bland_altman_plot(col1, col2, plot_label: str, anno_label: str, output_dir='',
                      col1_label='Baseline', col2_label='ActiNet',
                      display_plot=False, show_y_label=False, ax=None, fontsize=20):
    """Generates a Bland-Altman plot for two columns of data."""
    dat = pd.DataFrame({'col1': col1, 'col2': col2})
    pearson_cor = dat.corr().iloc[0, 1]
    diffs = dat['col2'] - dat['col1'] 
    mean_diff = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1)
    lower_loa = mean_diff - 1.96 * sd_diff
    upper_loa = mean_diff + 1.96 * sd_diff
    
    mean_vals = (dat['col1'] + dat['col2']) / 2

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10), dpi=800)

    ax.scatter(mean_vals, diffs, color='black', alpha=1)
    ax.axhline(mean_diff, color='red', linestyle='-')
    ax.axhline(lower_loa, color='blue', linestyle='--')
    ax.axhline(upper_loa, color='blue', linestyle='--')
    ax.text(0.8 * max(mean_vals), mean_diff, f'Mean Diff = {mean_diff:.2f}', va='bottom', color='red', fontsize=fontsize*0.8)
    ax.text(0.8 * max(mean_vals), lower_loa, f'Lower LoA = {lower_loa:.2f}', va='bottom', color='blue', fontsize=fontsize*0.8)
    ax.text(0.8 * max(mean_vals), upper_loa, f'Upper LoA = {upper_loa:.2f}', va='bottom', color='blue', fontsize=fontsize*0.8)

    ax.set_title(f'{ACTIVITY_LABELS_DICT[anno_label][plot_label]} [hours]\nPearson correlation: {pearson_cor:.3f}', fontsize=fontsize)

    ax.set_xlabel(f'({col2_label} + {col1_label}) / 2', fontsize=fontsize*0.9)
    
    if show_y_label:
        ax.set_ylabel(f'{col2_label} - {col1_label}', fontsize=fontsize*0.9)
    
    ax.tick_params(axis='both', which='both', labelsize=14)

    if display_plot and ax is None:
        plt.show()
    
    if ax is None:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'ba_{plot_label}_{col1_label}_vs_{col2_label}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()


def generate_bland_altman_plots(results_df, activities, anno_label, group_by=None, save_path=None, 
                                fontsize=20, compare_to_true=False, subset=""):
    """Generates Bland-Altman plots for different activities, optionally stratified by a subgroup."""
    if group_by is None:  # Full population
        fig, axs = plt.subplots(1, 4, figsize=(15, 6), dpi=800, sharey=True)      
        axs = axs.flatten()

        for i, activity in enumerate(activities):
            activity_bbaa_pred, activity_actinet_pred, activity_bbaa_true, activity_actinet_true, population =\
                 extract_activity_predictions(results_df, activity, return_true_labels=True)
            
            if compare_to_true=='bbaa':
                bland_altman_plot(activity_bbaa_true, activity_bbaa_pred, activity, anno_label,
                                  ax=axs[i], show_y_label=i==0, fontsize=fontsize*0.8,
                                  col1_label='Ground Truth', col2_label='Baseline')
            elif compare_to_true=='actinet':
                bland_altman_plot(activity_actinet_true, activity_actinet_pred, activity, anno_label,
                                  ax=axs[i], show_y_label=i==0, fontsize=fontsize*0.8,
                                  col1_label='Ground Truth', col2_label='ActiNet')
            elif compare_to_true is False:
                bland_altman_plot(activity_bbaa_pred, activity_actinet_pred, activity, anno_label,
                                  ax=axs[i], show_y_label=i==0, fontsize=fontsize*0.8,
                                  col1_label='Baseline', col2_label='ActiNet')
            else:
                raise ValueError("compare_to_true must be either False, 'bbaa' or 'actinet'")

        subset_in_title = f" {subset} " if subset else " "
        fig.suptitle(f"Bland-Altman plots for Capture-24{subset_in_title}population (n={population})", 
                     fontsize=fontsize)

    else:
        unique_groups = results_df[group_by].cat.categories
        fig = plt.figure(figsize=(15, 6*len(unique_groups)), dpi=800, constrained_layout=True)
        subfigs = fig.subfigures(nrows=len(unique_groups), ncols=1)

        for subfig, group in zip(subfigs, unique_groups):
            axs = subfig.subplots(nrows=1, ncols=len(activities), sharex=False, sharey=True)

            for i, activity in enumerate(activities):
                activity_bbaa_pred, activity_actinet_pred, activity_bbaa_true, activity_actinet_true, population =\
                    extract_activity_predictions(results_df, activity, **{group_by.replace(' ', '_').lower(): group}, 
                                                 return_true_labels=True)
                if compare_to_true=='bbaa':
                    bland_altman_plot(activity_bbaa_true, activity_bbaa_pred, activity,
                                      ax=axs[i], show_y_label=i==0, fontsize=fontsize*0.8,
                                      col1_label='Ground Truth', col2_label='Baseline')
                elif compare_to_true=='actinet':
                    bland_altman_plot(activity_actinet_true, activity_actinet_pred, activity,
                                      ax=axs[i], show_y_label=i==0, fontsize=fontsize*0.8,
                                      col1_label='Ground Truth', col2_label='ActiNet')
                elif compare_to_true is False:
                    bland_altman_plot(activity_bbaa_pred, activity_actinet_pred, activity,
                                      ax=axs[i], show_y_label=i==0, fontsize=fontsize*0.8,
                                      col1_label='Baseline', col2_label='ActiNet')
                else:
                    raise ValueError("compare_to_true must be either False, 'bbaa' or 'actinet'")

            subfig.suptitle(f"{group_by}: {group} (n={population})", fontsize=fontsize*0.9)

        fig.suptitle(f"Bland-Altman plots for Capture-24 population by {group_by}", fontsize=fontsize)

    fig.tight_layout()
    plot_and_save_fig(fig, save_path=save_path)
            
    plt.close(fig)


def build_mae_cell(true_values, pred_values):
    """Builds a MAE cell for a given set of true and predicted values."""
    mae = np.abs(true_values - pred_values)
    return f"{np.mean(mae):.3f} ± {np.std(mae):.3f}"


def build_pvalue_cell(true_values, pred_values):
    """Builds a p-value cell for a given set of true and predicted values."""
    _, p_value = stats.ttest_rel(true_values, pred_values)
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def build_mae_table(df: pd.DataFrame, activities):
    df_maes = pd.DataFrame(columns=activities, index=['Baseline', 'ActiNet', 'p-value'], dtype=float)
    for activity in activities:
        activity_bbaa_pred, activity_actinet_pred, activity_bbaa_true, activity_actinet_true, _ =\
            extract_activity_predictions(df, activity, return_true_labels=True)
        df_maes.loc['Baseline', activity] = build_mae_cell(activity_bbaa_true, activity_bbaa_pred)
        df_maes.loc['ActiNet', activity] = build_mae_cell(activity_actinet_true, activity_actinet_pred)
        df_maes.loc['p-value', activity] = build_pvalue_cell(activity_bbaa_true, activity_bbaa_pred)
    return df_maes


def plot_errors(df: pd.DataFrame, activities, anno_label, group_by=None, save_path=None, fontsize=12):
    if group_by is None:
        all_errors = []

        for activity in activities:
            activity_bbaa_pred, activity_actinet_pred, activity_bbaa_true, activity_actinet_true, _ = \
                extract_activity_predictions(df, activity, return_true_labels=True)
            
            bbaa_errors = activity_bbaa_pred - activity_bbaa_true
            actinet_errors = activity_actinet_pred - activity_actinet_true

            for err in bbaa_errors:
                all_errors.append({'Activity': ACTIVITY_LABELS_DICT[anno_label][activity],
                                   'Error': err, 'Model': 'Baseline'})
            for err in actinet_errors:
                all_errors.append({'Activity': ACTIVITY_LABELS_DICT[anno_label][activity],
                                   'Error': err, 'Model': 'ActiNet'})

        error_df = pd.DataFrame(all_errors)

        fig, ax = plt.subplots(figsize=(2*len(activities), 4), dpi=1000)
        with sns.color_palette("Set1"):
            sns.boxplot(x="Activity", y="Error", hue="Model", data=error_df,
                        width=0.5, showfliers=False, ax=ax)

        ax.set_xlabel("Activity", fontsize=fontsize)
        ax.set_ylabel("Error in total estimated activity [hours]\n(model - ground-truth)", fontsize=fontsize)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.grid(axis='y', alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        n = len(set(error_df['Model']))
        ax.legend(handles[:n], labels[:n], fontsize=fontsize*0.8)
        plt.title("Box plots for the error distribution of activity recognition models' outputs\n" +\
                  "evaluated on the full Capture-24 population", fontsize=fontsize)
        fig.tight_layout()

    else:
        unique_groups = df[group_by].cat.categories
        fig, axs = plt.subplots(nrows=len(unique_groups), ncols=1,
                                figsize=(2*len(activities), 4*len(unique_groups)),
                                dpi=1000, constrained_layout=True)
        
        if len(unique_groups) == 1:
            axs = [axs]

        for ax, group in zip(axs, unique_groups):
            sub_df = df[df[group_by] == group]
            all_errors = []

            for activity in activities:
                activity_bbaa_pred, activity_actinet_pred, activity_bbaa_true, activity_actinet_true, _ = \
                    extract_activity_predictions(sub_df, activity, return_true_labels=True)

                bbaa_errors = activity_bbaa_pred - activity_bbaa_true
                actinet_errors = activity_actinet_pred - activity_actinet_true

                for err in bbaa_errors:
                    all_errors.append({'Activity': ACTIVITY_LABELS_DICT[anno_label][activity],
                                       'Error': err, 'Model': 'Baseline'})
                for err in actinet_errors:
                    all_errors.append({'Activity': ACTIVITY_LABELS_DICT[anno_label][activity],
                                       'Error': err, 'Model': 'ActiNet'})

            error_df = pd.DataFrame(all_errors)

            with sns.color_palette("Set1"):
                sns.boxplot(x="Activity", y="Error", hue="Model", data=error_df,
                            width=0.5, showfliers=False, ax=ax)

            ax.set_xlabel("Activity", fontsize=fontsize)
            ax.set_ylabel("Error in total estimated activity [hours]\n(model - ground-truth)", fontsize=fontsize)
            ax.axhline(0, color='red', linestyle='--', linewidth=1)
            ax.grid(axis='y', alpha=0.7)
            ax.set_title(f"{group_by}: {group}", fontsize=fontsize)

            handles, labels = ax.get_legend_handles_labels()
            n = len(set(error_df['Model']))
            ax.legend(handles[:n], labels[:n], fontsize=fontsize*0.8)

        plt.suptitle("Box plots for the error distribution of activity recognition models' outputs\n" +\
                     f"evaluated on the Capture-24 population by {group_by}", fontsize=fontsize)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def convert_version(s: str) -> str:
    """ Converts a version string from 'x.y.z+number' to 'vX-Y-Z+number'. """
    match = re.match(r"(\d+)\.(\d+)\.(\d+)(?:\+(\d+))?", s)
    if not match:
        raise ValueError(f"String '{s}' is not in the expected format")
    
    major, minor, patch, extra = match.groups()
    base = f"v{major}-{minor}-{patch}"
    return f"{base}+{extra}" if extra else base

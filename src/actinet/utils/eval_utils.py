import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, balanced_accuracy_score, confusion_matrix
import warnings

warnings.simplefilter("ignore", UserWarning)

ACTIVITY_LABELS_DICT = {
    'sleep': 'Sleep',
    'sedentary': 'Sedentary Behaviour',
    'light': 'Light Activity',
    'moderate-vigorous': 'Moderate-Vigorous Activity'
}
ACTIVITY_LABELS = list(ACTIVITY_LABELS_DICT.keys())


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


def plot_confusion_matrix(y_true, y_pred, title="", ax=None, figsize=(8, 8), fontsize=20):
    """Plots a confusion matrix with heatmap annotations."""
    cm = confusion_matrix(y_true, y_pred, labels=ACTIVITY_LABELS)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    annotations = np.array([[f'{p:.3f}\n({n:,})' for p, n in zip(row_norm, row)] for row_norm, row in zip(cm_norm, cm)])
    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=800)

    sns.heatmap(cm_norm, annot=annotations, fmt='', cmap="Blues", cbar=False, 
                annot_kws={"size": fontsize*0.9}, ax=ax)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    ax.set_xticks(np.arange(len(ACTIVITY_LABELS)) + 0.5)
    ax.set_yticks(np.arange(len(ACTIVITY_LABELS)) + 0.5)
    ax.set_xticklabels(ACTIVITY_LABELS, fontsize=fontsize*0.8)
    ax.set_yticklabels(ACTIVITY_LABELS, fontsize=fontsize*0.8)


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


def generate_confusion_matrices(results_df, group_by=None, save_path=None, fontsize=20):
    """Generates and plots confusion matrices for different subgroups.""" 
    if group_by is None:  # Full population
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 6), dpi=800)
        fig.suptitle("Confusion matrices for full Capture-24 population\nusing 5-fold group cross-validation", 
                     fontsize=fontsize)
        
        y_true_bbaa, y_pred_bbaa, y_true_actinet, y_pred_actinet, _ = build_confusion_matrix_data(results_df)
        
        plot_confusion_matrix(y_true_bbaa, y_pred_bbaa, 'accelerometer', ax=axs[0], fontsize=fontsize*0.8)
        plot_confusion_matrix(y_true_actinet, y_pred_actinet, 'actinet', ax=axs[1], fontsize=fontsize*0.8)
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

            plot_confusion_matrix(y_true_bbaa, y_pred_bbaa, 'accelerometer', ax=axs[0], fontsize=fontsize*0.8)
            plot_confusion_matrix(y_true_actinet, y_pred_actinet, 'actinet', ax=axs[1], fontsize=fontsize*0.8)
        
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
    ax.set_ylabel("Macro F1 Score")
    plt.title(f"Macro F1 by {x}")
    plt.show()


def build_bland_altman_data(results: pd.DataFrame, activity, age_band=None, sex=None):
    """Extracts incidence of predicted activity label for actinet and accelerometer based on filtering conditions."""    
    model_results = results.copy()
    if age_band is not None:
        model_results = model_results[model_results['Age Band'] == age_band]
    if sex is not None:
        model_results = model_results[model_results["Sex"] == sex]

    activity_bbaa_pred = [x.get(activity, 0) for x in model_results.loc[model_results["Model"] == "accelerometer", "Pred_dict"]]
    activity_actinet_pred = [x.get(activity, 0) for x in model_results.loc[model_results["Model"] == "actinet", "Pred_dict"]]

    population = len(model_results['Participant'].unique())

    return activity_bbaa_pred, activity_actinet_pred, population


def bland_altman_plot(col1, col2, plot_label: str, output_dir='',
                      col1_label='accelerometer', col2_label='actinet',
                      display_plot=False, show_y_label=False, ax=None, fontsize=20):
    """Generates a Bland-Altman plot for two columns of data."""
    dat = pd.DataFrame({'col1': col1, 'col2': col2})
    pearson_cor = dat.corr().iloc[0, 1]
    diffs = dat['col1'] - dat['col2']
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
    
    ax.set_title(f'{ACTIVITY_LABELS_DICT[plot_label]} [hours]\nPearson correlation: {pearson_cor:.3f}', fontsize=fontsize)

    ax.set_xlabel(f'({col1_label} + {col2_label}) / 2', fontsize=fontsize*0.9)
    
    if show_y_label:
        ax.set_ylabel(f'{col1_label} - {col2_label}', fontsize=fontsize*0.9)
    
    ax.tick_params(axis='both', which='both', labelsize=14)

    if display_plot and ax is None:
        plt.show()
    
    if ax is None:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'ba_{plot_label}_{col1_label}_vs_{col2_label}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()


def generate_bland_altman_plots(results_df, activities=ACTIVITY_LABELS, group_by=None, save_path=None, 
                                fontsize=20, subset=""):
    """Generates Bland-Altman plots for different activities, optionally stratified by a subgroup."""
    
    if group_by is None:  # Full population
        fig, axs = plt.subplots(1, 4, figsize=(15, 6), dpi=800, sharey=True)      
        axs = axs.flatten()

        for i, activity in enumerate(activities):
            activity_bbaa_pred, activity_actinet_pred, population = build_bland_altman_data(results_df, activity)
            bland_altman_plot(activity_bbaa_pred, activity_actinet_pred, activity,
                              ax=axs[i], show_y_label=i==0, fontsize=fontsize*0.8)
        
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
                activity_bbaa_pred, activity_actinet_pred, population = build_bland_altman_data(
                    results_df, activity, **{group_by.replace(' ', '_').lower(): group})
                bland_altman_plot(activity_bbaa_pred, activity_actinet_pred, activity,
                                  ax=axs[i], show_y_label=i==0, fontsize=fontsize*0.8)
            
            subfig.suptitle(f"{group_by}: {group} (n={population})", fontsize=fontsize*0.9)
        

        fig.suptitle(f"Bland-Altman plots for Capture-24 population by {group_by}", fontsize=fontsize)
    
    fig.tight_layout()
    plot_and_save_fig(fig, save_path=save_path)
            
    plt.close(fig)

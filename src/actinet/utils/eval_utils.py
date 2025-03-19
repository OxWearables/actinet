import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, balanced_accuracy_score, confusion_matrix
import warnings

warnings.simplefilter("ignore", UserWarning)


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
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes, normalize='true') 
    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=800)

    sns.heatmap(cm, annot=True, fmt='.3f', cmap="Blues", cbar=False, 
                annot_kws={"size": fontsize*0.8}, ax=ax)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    ax.set_xticks(np.arange(len(unique_classes)) + 0.5)
    ax.set_yticks(np.arange(len(unique_classes)) + 0.5)
    ax.set_xticklabels(unique_classes, fontsize=fontsize*0.7)
    ax.set_yticklabels(unique_classes, fontsize=fontsize*0.7)


def build_confusion_matrix_data(results: pd.DataFrame, age_band=None, sex=None):
    """Extracts ground truth and predicted labels based on filtering conditions."""
    model_results = results.copy()
    if age_band is not None:
        model_results = model_results[model_results['Age Band'] == age_band]
    if sex is not None:
        model_results = model_results[model_results["Sex"] == sex]

    y_true = np.hstack(model_results.loc[model_results["Model"]=="actinet", 'True'])
    y_pred_bbaa = np.hstack(model_results.loc[model_results["Model"]=="accelerometer", 'Predicted'])
    y_pred_actinet = np.hstack(model_results.loc[model_results["Model"]=="actinet", 'Predicted'])
    
    return y_true, y_pred_bbaa, y_pred_actinet


def plot_and_save_fig(fig, save_path=None):
    """Displays and optionally saves the figure as a PDF."""
    plt.show()
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=800, bbox_inches='tight')


def generate_confusion_matrices(results_df, group_by=None, save_path=None):
    """Generates and plots confusion matrices for different subgroups.""" 
    if group_by is None:  # Full population
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 6), dpi=800)
        fig.suptitle("Confusion matrices for full Capture-24 population using 5-fold group cross-validation", 
                     fontsize=16)
        
        y_true, y_pred_bbaa, y_pred_actinet = build_confusion_matrix_data(results_df)
        
        plot_confusion_matrix(y_true, y_pred_bbaa, 'accelerometer', ax=axs[0], fontsize=14)
        plot_confusion_matrix(y_true, y_pred_actinet, 'actinet', ax=axs[1], fontsize=14)
        
        plot_and_save_fig(fig, save_path)
    
    else:
        unique_groups = results_df[group_by].cat.categories
        fig = plt.figure(figsize=(12, 6*len(unique_groups)), dpi=800, constrained_layout=True)
        fig.suptitle(f"Confusion matrices for different {group_by.lower()} in " +
                     "Capture-24 population\nusing 5-fold group cross-validation", fontsize=16)

        subfigs = fig.subfigures(nrows=len(unique_groups), ncols=1)
        
        for i, (group, subfig) in enumerate(zip(unique_groups, subfigs)):
            subfig.suptitle(f"{group_by}: {group}", fontsize=14)
            axs = subfig.subplots(nrows=1, ncols=2, sharey=True)

            y_true, y_pred_bbaa, y_pred_actinet =\
                build_confusion_matrix_data(results_df, **{group_by.replace(' ', '_').lower(): group})

            plot_confusion_matrix(y_true, y_pred_bbaa, 'accelerometer', ax=axs[0], fontsize=14)
            plot_confusion_matrix(y_true, y_pred_actinet, 'actinet', ax=axs[1], fontsize=14)
        
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

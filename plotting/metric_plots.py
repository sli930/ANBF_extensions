# plotting/metric_plots.py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import seaborn as sns
sns.set_theme()

def plot_heatmap(df, xlabel, ylabel, figsize=(10, 8)):
    heatmap_data = df.unstack()
    print(heatmap_data)
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='Blues', cbar_kws={'label': 'Accuracy'})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def plot_metric_vs_window_length(dfs, metric="Accuracy", x_step=0.5, y_step=0.1, figsize=(10, 8), palette="Set1"):
    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=dfs, x="Window length (s)", y=metric, hue="Method", style="n_classes", palette=palette)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    plt.xlim(0, 4.86)
    plt.show()
    
def plot_metric_vs_n_classes(dfs, metric="Accuracy", x_step=8, y_step=0.1, figsize=(10, 8), palette="Set1"):    
    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=dfs, x="n_classes", y=metric, hue="Method", palette=palette)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    plt.xlim(8, 40)
    plt.show()
    
    
def plot_metric_static(dfs):
    plot_metric_box(dfs)
    plot_metric_bar(dfs)
    plot_metric_swarm_by_method(dfs)

def plot_metric_box(dfs, metric="Accuracy", figsize=(10, 8), palette="Set1"):
    plt.figure(figsize=figsize)
    sns.boxplot(data=dfs, x="Method", y=metric, hue="n_classes", palette=palette)
    plt.title(f"{metric} Boxplot")
    plt.tight_layout()
    plt.show()


def plot_metric_bar(dfs, metric="Accuracy", figsize=(10, 8), palette="Set1"):
    plt.figure(figsize=figsize)
    sns.barplot(data=dfs, x="Method", y=metric, hue="n_classes",
                errorbar="sd", palette=palette)
    plt.title(f"{metric} Barplot")
    plt.tight_layout()
    plt.show()


def plot_metric_swarm_by_method(dfs, metric="Accuracy",
                                figsize=(10, 8), palette="Set1"):
    sns.catplot(kind="swarm", data=dfs, x="n_classes", y=metric,
                hue="subject", col="Method", palette=palette,
                legend=False, height=figsize[-1] - 2)
    plt.show()

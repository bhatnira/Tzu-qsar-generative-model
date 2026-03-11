"""
visualization.py
Module for plotting and visualization functions.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_ic50_distribution(data, col="IC50 uM"):
    plt.figure(figsize=(10,5))
    plt.hist(data, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    if len(data) > 10:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 200)
            plt.plot(x, kde(x), 'r-', linewidth=2)
        except:
            pass
    plt.axvline(data.mean(), color='red', linestyle='--', label=f"Mean {data.mean():.2f}")
    plt.axvline(data.median(), color='green', linestyle='--', label=f"Median {data.median():.2f}")
    plt.title(f"{col} Distribution (n={len(data)}, skew={data.skew():.2f})")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

def plot_umap_clusters(numeric_df):
    plt.figure(figsize=(12, 8))
    unique_clusters = sorted(set(numeric_df['Cluster']))
    palette = sns.color_palette("hsv", len(unique_clusters))
    sns.scatterplot(
        x='UMAP1', y='UMAP2',
        hue='Cluster',
        data=numeric_df,
        palette=palette,
        s=30,
        alpha=0.7,
        legend='full'
    )
    plt.title("UMAP of RDKit Descriptors with HDBSCAN Clusters")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()

def plot_y_true_vs_pred(y_true, y_pred, model_name, desc_name):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("True IC50 (nM, Series C)")
    plt.ylabel("Predicted IC50 (nM)")
    plt.title(f"Best model: {model_name} using {desc_name} descriptor")
    plt.show()

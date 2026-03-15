"""
visualization.py
Module for plotting and visualization functions.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def plot_ic50_distribution(data, col="IC50 uM"):
    """
    Plot IC50 distribution with histogram, KDE, and statistics.
    
    Args:
        data: pandas Series or numpy array of IC50 values
        col: Column name for plotting
    """
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

def plot_ic50_boxplot(data, col="IC50 uM"):
    """
    Plot IC50 distribution as boxplot with outlier analysis.
    
    Args:
        data: pandas Series or numpy array of IC50 values
        col: Column name for plotting
    """
    plt.figure(figsize=(6,4))
    bp = plt.boxplot(data, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')

    Q1, Q3 = data.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = data[(data < lower) | (data > upper)]
    out_pct = len(outliers) * 100 / len(data)

    plt.title(f"{col} Outliers: {len(outliers)} ({out_pct:.1f}%)")
    plt.ylabel(col)
    plt.grid(alpha=0.3)
    plt.show()

def univariate_analysis(data, col="IC50 uM"):
    """
    Comprehensive univariate statistical analysis of IC50.
    
    Args:
        data: pandas Series of IC50 values
        col: Column name for reference
        
    Returns:
        dict: Statistical summary
    """
    print("UNIVARIATE ANALYSIS:", col.upper())
    print("="*60)
    
    # Statistics
    Q1, Q3 = data.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = data[(data < lower) | (data > upper)]
    out_pct = len(outliers) * 100 / len(data)
    
    row = {
        "Count": len(data),
        "Mean": data.mean(),
        "Median": data.median(),
        "Std": data.std(),
        "Min": data.min(),
        "Max": data.max(),
        "Range": data.max() - data.min(),
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Skewness": data.skew(),
        "Kurtosis": data.kurtosis(),
        "Outliers": len(outliers),
        "Outlier %": out_pct
    }
    
    # Normality test
    if len(data) >= 8:
        stat, p = stats.normaltest(data)
        row["Normal p-value"] = p
        row["Likely Normal"] = p > 0.05
    else:
        row["Normal p-value"] = np.nan
        row["Likely Normal"] = "Insufficient data"
    
    print("\nSTATISTICAL SUMMARY")
    print("="*60)
    import pandas as pd
    summary_df = pd.DataFrame([row])
    print(summary_df.round(3).to_string())
    
    # Recommendations
    print("\nINSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    skew = row["Skewness"]
    std = row["Std"]
    rng = row["Range"]
    
    recommendations = []
    
    if abs(skew) > 2:
        recommendations.append("Highly skewed — consider log transform.")
    elif abs(skew) > 0.5:
        recommendations.append("Moderately skewed — consider a transform.")
    
    if out_pct > 10:
        recommendations.append("High outlier rate — consider robust models.")
    elif out_pct > 5:
        recommendations.append("Moderate number of outliers — evaluate model impact.")
    
    if std > 0 and rng > 100 * std:
        recommendations.append("Very wide dynamic range — consider scaling.")
    
    if not recommendations:
        recommendations.append("Data appears well-behaved.")
    
    for r in recommendations:
        print("-", r)
    
    return row

def plot_umap_clusters(numeric_df):
    """
    Plot UMAP with HDBSCAN cluster coloring.
    
    Args:
        numeric_df: DataFrame with UMAP1, UMAP2, Cluster columns
    """
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
    """
    Plot true vs predicted IC50 values.
    
    Args:
        y_true: True IC50 values
        y_pred: Predicted IC50 values
        model_name: Name of the model
        desc_name: Name of descriptor type
    """
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("True IC50 (nM, Series C)")
    plt.ylabel("Predicted IC50 (nM)")
    plt.title(f"Best model: {model_name} using {desc_name} descriptor")
    plt.grid(alpha=0.3)
    plt.show()

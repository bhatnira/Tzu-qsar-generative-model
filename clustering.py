"""
clustering.py
Module for UMAP, HDBSCAN, and chemical space analysis.
"""
import numpy as np
import pandas as pd
import umap.umap_ as umap
import hdbscan
from sklearn.metrics import silhouette_score
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns

def run_umap(descriptor_matrix, n_neighbors=30, min_dist=0.1, n_components=2, random_state=42):
    """
    Run UMAP dimensionality reduction on descriptor matrix.
    
    Args:
        descriptor_matrix: numpy array of shape (n_molecules, n_descriptors)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance parameter for UMAP
        n_components: Number of output dimensions
        random_state: Random seed for reproducibility
        
    Returns:
        numpy array of shape (n_molecules, n_components)
    """
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='euclidean',
        random_state=random_state
    )
    return umap_model.fit_transform(descriptor_matrix)

def run_hdbscan(umap_result, min_sizes=[10, 20, 30, 50, 75, 100]):
    """
    Run HDBSCAN clustering with parameter tuning.
    
    Args:
        umap_result: UMAP-reduced descriptor matrix
        min_sizes: List of min_cluster_size parameters to try
        
    Returns:
        tuple: (best_clusters, best_min_size, best_score)
    """
    best_clusters = None
    best_score = -1
    best_min_size = None
    results = []
    
    for min_size in min_sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, metric='euclidean')
        labels = clusterer.fit_predict(umap_result)
        
        # Skip if no clusters formed
        if len(set(labels)) <= 1 or set(labels) == {-1}:
            continue
        
        mask = labels != -1
        if sum(mask) < 2:
            continue
        
        score = silhouette_score(umap_result[mask], labels[mask])
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        results.append((min_size, n_clusters, score))
        
        if score > best_score:
            best_score = score
            best_clusters = labels
            best_min_size = min_size
    
    return best_clusters, best_min_size, best_score

def get_scaffold_safe(mol):
    """
    Safely extract Murcko scaffold from a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        str: SMILES of scaffold or None if extraction fails
    """
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None

def analyze_chemical_space(numeric_df):
    """
    Comprehensive chemical space analysis including clustering composition.
    
    Args:
        numeric_df: DataFrame with columns 'Cluster' and 'Series_Code'
        
    Returns:
        dict: Statistics about cluster composition
    """
    cluster_series_counts = (
        numeric_df
        .groupby(["Cluster","Series_Code"])
        .size()
        .reset_index(name="Count")
        .sort_values(["Cluster","Count"], ascending=[True,False])
    )
    
    cluster_series_table = pd.crosstab(
        numeric_df["Cluster"],
        numeric_df["Series_Code"]
    )
    
    cluster_series_percent = cluster_series_table.div(
        cluster_series_table.sum(axis=1),
        axis=0
    ) * 100
    
    dominant_series = (
        numeric_df
        .groupby(["Cluster","Series_Code"])
        .size()
        .reset_index(name="count")
        .sort_values(["Cluster","count"], ascending=[True,False])
        .drop_duplicates("Cluster")
    )
    
    dominant_series_dict = dict(
        zip(dominant_series["Cluster"], dominant_series["Series_Code"])
    )
    
    return {
        "cluster_series_counts": cluster_series_counts,
        "cluster_series_table": cluster_series_table,
        "cluster_series_percent": cluster_series_percent,
        "dominant_series_dict": dominant_series_dict
    }

def plot_chemical_space(numeric_df, dominant_series_dict=None, use_alphashape=False):
    """
    Plot chemical space with series coloring and optional alpha shapes.
    
    Args:
        numeric_df: DataFrame with UMAP1, UMAP2, Series_Code, and Cluster columns
        dominant_series_dict: Dictionary mapping cluster IDs to dominant series
        use_alphashape: Whether to draw alpha shapes around clusters (requires alphashape package)
    """
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(14,10))
    
    palette = sns.color_palette("husl", numeric_df["Series_Code"].nunique())
    
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue="Series_Code",
        palette=palette,
        data=numeric_df,
        s=65,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
        ax=ax
    )
    
    if use_alphashape and dominant_series_dict:
        try:
            import alphashape
            from shapely.geometry import Polygon
            
            for cluster_id in sorted(numeric_df["Cluster"].unique()):
                if cluster_id == -1:
                    continue
                
                cluster_points = numeric_df[
                    numeric_df["Cluster"] == cluster_id
                ][["UMAP1","UMAP2"]].values
                
                if len(cluster_points) < 10:
                    continue
                
                alpha = alphashape.optimizealpha(cluster_points)
                hull = alphashape.alphashape(cluster_points, alpha)
                
                if isinstance(hull, Polygon):
                    x, y = hull.exterior.xy
                    ax.fill(x, y, alpha=0.15, color="black", linewidth=2)
        except ImportError:
            print("alphashape not installed, skipping alpha shapes")
    
    if dominant_series_dict:
        for cluster_id in sorted(numeric_df["Cluster"].unique()):
            if cluster_id == -1:
                continue
            
            subset = numeric_df[numeric_df["Cluster"] == cluster_id]
            cx = subset["UMAP1"].mean()
            cy = subset["UMAP2"].mean()
            
            label = dominant_series_dict.get(cluster_id, "Unknown")
            
            ax.text(
                cx, cy,
                f"{label}",
                fontsize=13,
                weight="bold",
                ha="center",
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.8)
            )
    
    ax.set_title("Chemical Space Map (UMAP + HDBSCAN)", fontsize=18, weight="bold")
    ax.set_xlabel("UMAP 1", fontsize=14)
    ax.set_ylabel("UMAP 2", fontsize=14)
    
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", title="Series_Code")
    plt.tight_layout()
    plt.show()

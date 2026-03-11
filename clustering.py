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

def run_umap(descriptor_matrix, n_neighbors=30, min_dist=0.1, n_components=2, random_state=42):
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='euclidean',
        random_state=random_state
    )
    return umap_model.fit_transform(descriptor_matrix)

def run_hdbscan(umap_result, min_sizes=[10, 20, 30, 50, 75, 100]):
    best_clusters = None
    best_score = -1
    best_min_size = None
    for min_size in min_sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, metric='euclidean')
        labels = clusterer.fit_predict(umap_result)
        if len(set(labels)) <= 1 or set(labels) == {-1}:
            continue
        mask = labels != -1
        if sum(mask) < 2:
            continue
        score = silhouette_score(umap_result[mask], labels[mask])
        if score > best_score:
            best_score = score
            best_clusters = labels
            best_min_size = min_size
    return best_clusters, best_min_size, best_score

def get_scaffold_safe(mol):
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

def feature_select(X_train, df,thr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    corr = spearmanr(X_train).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    hierarchy.set_link_color_palette(['darkred'])

    dendro = hierarchy.dendrogram(
        dist_linkage, color_threshold=thr,above_threshold_color='cornflowerblue',labels=df.drop(columns=['norm_diff']).columns.to_list(),
        ax=ax1, leaf_rotation=90
    )

    dendro_idx = np.arange(0, len(dendro["ivl"]))

    
    cmap = mpl.cm.coolwarm
    bounds = [-1, -.5,0,.5, 1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=bounds,fraction=0.046)

    im=ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]],cmap=cmap, norm=norm)

    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    #
    _ = fig.tight_layout()

    cluster_ids = hierarchy.fcluster(dist_linkage, thr, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    # Generate all possible permutations using itertools.product
    all_permutations = list(itertools.product(*cluster_id_to_feature_ids.values()))

    return all_permutations, fig
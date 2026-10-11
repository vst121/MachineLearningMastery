"""
Hierarchical Clustering
=======================

Explore agglomerative clustering using the Iris dataset.

Agglomerative clustering starts with individual samples and
repeatedly merges the closest clusters.

This example demonstrates:
    - Feature scaling
    - Agglomerative clustering
    - Dendrogram visualization
    - Cluster evaluation with silhouette score
    - Cluster visualization
"""

# %% 1. Imports

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# %% 2. Load the Iris dataset

iris = load_iris()

X = iris.data
feature_names = iris.feature_names

data = pd.DataFrame(X, columns=feature_names)
data["Species"] = [iris.target_names[i] for i in iris.target]

print("Iris dataset loaded successfully.")
print(f"Number of samples: {len(data)}")
print(f"Number of features: {len(feature_names)}")


# %% 3. Standardize the features

# Distance-based clustering is sensitive to feature scales.
# Standardization gives each feature a comparable scale.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# %% 4. Build the hierarchical linkage

# Ward linkage merges clusters while minimizing the increase
# in within-cluster variance.
#
# The linkage matrix records which clusters were merged and
# the distance at which each merge occurred.

linkage_matrix = linkage(
    X_scaled,
    method="ward",
)


# %% 5. Plot the dendrogram

plt.figure(figsize=(14, 6))

dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=30,
    show_leaf_counts=True,
    leaf_rotation=45,
    leaf_font_size=9,
)

plt.title("Hierarchical Clustering Dendrogram (Iris)")
plt.xlabel("Samples or merged clusters")
plt.ylabel("Merge distance")
plt.tight_layout()
plt.show()


# %% 6. Fit agglomerative clustering

# Unlike K-Means, this model does not repeatedly calculate
# cluster centers. It builds clusters by merging groups.
#
# We choose three clusters for comparison with the previous
# K-Means examples.

number_of_clusters = 3

model = AgglomerativeClustering(
    n_clusters=number_of_clusters,
    linkage="ward",
)

cluster_labels = model.fit_predict(X_scaled)

data["Cluster"] = cluster_labels

score = silhouette_score(X_scaled, cluster_labels)

print(f"\nNumber of clusters: {number_of_clusters}")
print(f"Silhouette score: {score:.4f}")

print("\nCluster sizes:")
print(data["Cluster"].value_counts().sort_index().to_string())


# %% 7. Compare clusters with known Iris species

# Species labels are used only to interpret the results.
# They are not passed to the clustering algorithm.

print("\nCluster distribution by species:")
print(
    data.groupby(["Cluster", "Species"])
    .size()
    .unstack(fill_value=0)
)


# %% 8. Visualize the resulting clusters

petal_length = feature_names.index("petal length (cm)")
petal_width = feature_names.index("petal width (cm)")

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    X[:, petal_length],
    X[:, petal_width],
    c=cluster_labels,
    cmap="viridis",
    s=55,
    alpha=0.8,
)

plt.title(
    f"Hierarchical Clustering of Iris "
    f"(K={number_of_clusters})"
)
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.colorbar(scatter, label="Cluster")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

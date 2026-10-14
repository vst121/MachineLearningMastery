"""
Machine Learning Foundations
Topic: K-Means Clustering Demo

Dataset:
Scikit-learn Iris dataset.

Learning objectives:
- Understand the basic idea of clustering.
- Train a K-Means clustering model.
- Understand cluster centers and cluster assignments.
- Visualize clusters in 2D.
- Compare discovered clusters with known species labels.
"""

# %%
# 1. Imports

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# %%
# 2. Prepare the dataset

dataset = load_iris(as_frame=True)
df = dataset.frame

# Select two features to make visualization easy.
feature_names = ["petal length (cm)", "petal width (cm)"]

X = df[feature_names]

# These labels are only for comparison after clustering.
# K-Means does not use them during training.
y_true = df["target"]
target_names = dataset.target_names

print("Dataset preview:")
print(df[feature_names + ["target"]].head())

print(f"\nNumber of observations: {len(X)}")
print(f"Number of features: {X.shape[1]}")


# %%
# 3. Visualize the original data

# Here, we use the known species labels only to see how the
# flowers are distributed. These labels are not given to K-Means.

plt.figure(figsize=(8, 5))

for class_id, class_name in enumerate(target_names):
    class_data = df[df["target"] == class_id]

    plt.scatter(
        class_data[feature_names[0]],
        class_data[feature_names[1]],
        label=class_name,
        s=55,
        alpha=0.75,
    )

plt.title("Iris Dataset: Known Species")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 4. Scale the features

# K-Means uses distances between observations and cluster centers.
# Standardization prevents features with larger numerical scales
# from dominating the distance calculation.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features standardized successfully.")


# %%
# 5. Train the K-Means model

# We ask K-Means to discover three clusters.
# The algorithm does not know the actual flower species.

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

model.fit(X_scaled)

print("K-Means model trained successfully.")
print(f"Number of clusters: {model.n_clusters}")


# %%
# 6. Assign observations to clusters

cluster_labels = model.labels_

results = df[feature_names].copy()
results["Cluster"] = cluster_labels
results["Actual species"] = y_true.map(
    dict(enumerate(target_names))
)

print("Cluster assignments:")
print(results.head(10))

print("\nNumber of observations in each cluster:")
print(results["Cluster"].value_counts().sort_index())


# %%
# 7. Visualize the discovered clusters

plt.figure(figsize=(8, 5))

for cluster_id in range(model.n_clusters):
    cluster_data = X[cluster_labels == cluster_id]

    plt.scatter(
        cluster_data[feature_names[0]],
        cluster_data[feature_names[1]],
        label=f"Cluster {cluster_id}",
        s=55,
        alpha=0.75,
    )

# Convert cluster centers back to the original feature scale.
centers_original = scaler.inverse_transform(model.cluster_centers_)

plt.scatter(
    centers_original[:, 0],
    centers_original[:, 1],
    marker="X",
    s=220,
    color="black",
    label="Cluster centers",
)

plt.title("K-Means Clustering")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 8. Compare clusters with known species

# Cluster IDs do not correspond directly to species IDs.
# For example, Cluster 0 does not necessarily mean Setosa.

comparison = pd.crosstab(
    results["Actual species"],
    results["Cluster"],
)

print("Known species versus discovered clusters:")
print(comparison)


# %%
# 9. Inspect the cluster centers

centers_df = pd.DataFrame(
    centers_original,
    columns=feature_names,
)

print("Cluster centers in the original feature units:")
print(centers_df)


"""
Elbow Method
============

Use the Iris dataset to explore how to choose the number of
clusters for K-Means.

Dataset:
    Iris dataset from scikit-learn

Features:
    - Sepal length
    - Sepal width
    - Petal length
    - Petal width

Run:
    uv run python 02_elbow_method.py
"""

# %% 1. Imports

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# %% 2. Load the Iris dataset

iris = load_iris()

X = iris.data

FEATURE_NAMES = iris.feature_names

data = pd.DataFrame(X, columns=FEATURE_NAMES)
data["Species"] = [iris.target_names[i] for i in iris.target]

print("Iris dataset loaded successfully.")
print(f"Number of samples: {len(data)}")
print(f"Number of features: {len(FEATURE_NAMES)}")
print("\nFirst five samples:")
print(data.head().to_string(index=False))


# %% 3. Prepare and scale the features

# K-Means uses distances to assign samples to clusters.
# Standardization prevents features with larger numerical ranges
# from dominating the distance calculations.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures standardized.")
print(f"Scaled data shape: {X_scaled.shape}")


# %% 4. Calculate inertia for different values of K

# Inertia is the sum of squared distances between each sample
# and the center of its assigned cluster.
#
# Inertia generally decreases as K increases.
# We look for a point where adding more clusters provides
# progressively smaller improvements.

max_k = 10
k_values = list(range(1, max_k + 1))
inertias = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    model.fit(X_scaled)
    inertias.append(model.inertia_)

results = pd.DataFrame(
    {
        "Number of clusters (K)": k_values,
        "Inertia": inertias,
    }
)

print("\nInertia for each value of K:")
print(results.to_string(index=False))


# %% 5. Visualize the Elbow Method

plt.figure(figsize=(9, 5))

plt.plot(
    k_values,
    inertias,
    marker="o",
    linewidth=2,
)

plt.title("Elbow Method with the Iris Dataset")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% 6. Inspect a candidate number of clusters

# The Iris dataset contains three known species.
# We use K=3 as a candidate, but the Elbow Method itself
# does not know the species labels.

candidate_k = 3

candidate_model = KMeans(
    n_clusters=candidate_k,
    random_state=42,
    n_init=10,
)

data["Cluster"] = candidate_model.fit_predict(X_scaled)

print(f"\nCluster assignments for K={candidate_k}:")
print(
    data[["Species", "Cluster"]]
    .groupby(["Cluster", "Species"])
    .size()
    .unstack(fill_value=0)
)


# %% 7. Visualize the clusters

# The model uses all four features.
# This plot shows only petal length and petal width.
# Cluster centers are transformed back to the original units.

centers_scaled = candidate_model.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

petal_length = "petal length (cm)"
petal_width = "petal width (cm)"

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    data[petal_length],
    data[petal_width],
    c=data["Cluster"],
    cmap="viridis",
    s=55,
    alpha=0.8,
)

plt.scatter(
    centers_original[:, FEATURE_NAMES.index(petal_length)],
    centers_original[:, FEATURE_NAMES.index(petal_width)],
    marker="X",
    s=220,
    c="red",
    edgecolors="black",
    label="Cluster centers",
)

plt.title(f"K-Means Clustering of Iris (K={candidate_k})")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend()
plt.colorbar(scatter, label="Cluster")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

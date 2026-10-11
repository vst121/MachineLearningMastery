"""
Silhouette Score
================

Evaluate K-Means clustering quality using the Iris dataset.

The silhouette score measures how well samples fit within
their assigned clusters compared with neighboring clusters.

Score range:
    -1 to 1

Interpretation:
    Close to 1: Well-separated clusters
    Close to 0: Samples near cluster boundaries
    Below 0: Samples may be assigned to the wrong cluster
"""

# %% 1. Imports

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_samples, silhouette_score
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

# Silhouette scores depend on distances.
# Standardize the features so their different scales do not
# disproportionately influence the distance calculations.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# %% 4. Compare different numbers of clusters

# A silhouette score requires at least two clusters and fewer
# clusters than the number of samples.

k_values = range(2, 11)
results = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    cluster_labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, cluster_labels)

    results.append(
        {
            "Number of clusters (K)": k,
            "Inertia": model.inertia_,
            "Silhouette score": score,
        }
    )

results_df = pd.DataFrame(results)

print("\nClustering evaluation:")
print(results_df.round(4).to_string(index=False))

best_row = results_df.loc[results_df["Silhouette score"].idxmax()]
best_k = int(best_row["Number of clusters (K)"])

print(f"\nHighest silhouette score: K={best_k}")
print(f"Silhouette score: {best_row['Silhouette score']:.4f}")


# %% 5. Plot silhouette scores

plt.figure(figsize=(9, 5))

plt.plot(
    results_df["Number of clusters (K)"],
    results_df["Silhouette score"],
    marker="o",
    linewidth=2,
)

plt.axvline(
    best_k,
    linestyle="--",
    label=f"Highest score: K={best_k}",
)

plt.title("Silhouette Score for Different Values of K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Average silhouette score")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# %% 6. Fit the candidate model

# The highest score is a useful candidate, not necessarily
# the best choice for every practical objective.

candidate_k = best_k

model = KMeans(
    n_clusters=candidate_k,
    random_state=42,
    n_init=10,
)

cluster_labels = model.fit_predict(X_scaled)

data["Cluster"] = cluster_labels

sample_scores = silhouette_samples(X_scaled, cluster_labels)

data["Silhouette score"] = sample_scores

print(f"\nCluster sizes for K={candidate_k}:")
print(data["Cluster"].value_counts().sort_index().to_string())

print("\nAverage silhouette score per cluster:")
print(
    data.groupby("Cluster")["Silhouette score"]
    .mean()
    .round(4)
    .to_string()
)


# %% 7. Visualize individual silhouette scores

plt.figure(figsize=(10, 6))

y_lower = 10

for cluster_id in range(candidate_k):
    cluster_scores = sample_scores[cluster_labels == cluster_id]
    cluster_scores.sort()

    cluster_size = len(cluster_scores)
    y_upper = y_lower + cluster_size

    plt.fill_betweenx(
        range(y_lower, y_upper),
        0,
        cluster_scores,
        alpha=0.7,
    )

    plt.text(
        -0.05,
        y_lower + cluster_size / 2,
        str(cluster_id),
        va="center",
    )

    y_lower = y_upper + 10

average_score = silhouette_score(X_scaled, cluster_labels)

plt.axvline(
    average_score,
    color="red",
    linestyle="--",
    label=f"Average score: {average_score:.3f}",
)

plt.title(f"Silhouette Plot for Iris Clustering (K={candidate_k})")
plt.xlabel("Silhouette coefficient")
plt.ylabel("Samples grouped by cluster")
plt.legend()
plt.tight_layout()
plt.show()


# %% 8. Visualize the clusters

# K-Means uses all four features.
# This chart shows only petal length and petal width.

petal_length = feature_names.index("petal length (cm)")
petal_width = feature_names.index("petal width (cm)")

centers_original = scaler.inverse_transform(model.cluster_centers_)

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    X[:, petal_length],
    X[:, petal_width],
    c=cluster_labels,
    cmap="viridis",
    s=55,
    alpha=0.8,
)

plt.scatter(
    centers_original[:, petal_length],
    centers_original[:, petal_width],
    marker="X",
    s=220,
    c="red",
    edgecolors="black",
    label="Cluster centers",
)

plt.title(f"Iris Clusters (K={candidate_k})")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend()
plt.colorbar(scatter, label="Cluster")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()


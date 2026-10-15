# %% [markdown]
# # Categorical Encoding
#
# Machine learning models generally require numerical input features.
# Categorical encoding converts text-based categories into numerical
# representations.
#
# This example covers:
# - One-hot encoding
# - Ordinal encoding
# - Handling unknown categories
# - Choosing an appropriate encoding strategy
#
# Important:
# - One-hot encoding represents categories without implying an order.
# - Ordinal encoding assigns integer codes and can imply ordering.
# - Choose an encoding based on the meaning of the feature and the model.

# %%
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# %% [markdown]
# ## 1. Create a sample dataset
#
# The dataset contains nominal categories with no natural order and
# ordinal categories with a meaningful order.

# %%
data = pd.DataFrame(
    {
        "city": [
            "Mannheim",
            "Heidelberg",
            "Stuttgart",
            "Mannheim",
            "Stuttgart",
            "Heidelberg",
            "Mannheim",
            "Stuttgart",
        ],
        "membership": [
            "Basic",
            "Standard",
            "Premium",
            "Premium",
            "Basic",
            "Standard",
            "Standard",
            "Premium",
        ],
        "satisfaction": [
            "Low",
            "Medium",
            "High",
            "Medium",
            "Low",
            "High",
            "Medium",
            "High",
        ],
    }
)

print("Original dataset:")
print(data)

print("\nUnique values:")
for column in data.columns:
    print(f"{column}: {data[column].unique()}")

# %% [markdown]
# ## 2. One-hot encoding
#
# OneHotEncoder creates a separate binary feature for each category.
# It is appropriate for nominal features such as city, where no category
# is inherently greater than another.
#
# handle_unknown="ignore" prevents errors when new categories appear
# during transformation.

# %%
one_hot_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)

city_encoded = one_hot_encoder.fit_transform(data[["city"]])
city_feature_names = one_hot_encoder.get_feature_names_out(["city"])

city_encoded_df = pd.DataFrame(
    city_encoded,
    columns=city_feature_names,
    index=data.index,
)

print("\nOne-hot encoded city:")
print(city_encoded_df)

# %% [markdown]
# ## 3. Ordinal encoding
#
# OrdinalEncoder maps categories to integer values.
# Define the category order explicitly when the categories have a meaningful
# order, such as Low < Medium < High.
#
# Avoid assigning arbitrary numeric codes to nominal categories when the
# model could interpret those values as meaningful distances or ordering.

# %%
satisfaction_order = ["Low", "Medium", "High"]

ordinal_encoder = OrdinalEncoder(
    categories=[satisfaction_order],
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)

satisfaction_encoded = ordinal_encoder.fit_transform(
    data[["satisfaction"]]
)

satisfaction_encoded_df = pd.DataFrame(
    satisfaction_encoded.astype(int),
    columns=["satisfaction_encoded"],
    index=data.index,
)

print("\nOrdinal encoded satisfaction:")
print(
    pd.concat(
        [data[["satisfaction"]], satisfaction_encoded_df],
        axis=1,
    )
)

# %% [markdown]
# ## 4. Encode membership categories
#
# Membership labels may look ordered, but that does not automatically mean
# the numerical distance between tiers is meaningful.
#
# We use one-hot encoding here to avoid imposing an artificial numeric
# relationship between Basic, Standard, and Premium.

# %%
membership_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)

membership_encoded = membership_encoder.fit_transform(
    data[["membership"]]
)

membership_encoded_df = pd.DataFrame(
    membership_encoded,
    columns=membership_encoder.get_feature_names_out(["membership"]),
    index=data.index,
)

print("\nOne-hot encoded membership:")
print(membership_encoded_df)

# %% [markdown]
# ## 5. Handle previously unseen categories
#
# A production model may encounter a category that was absent during training.
# With handle_unknown="ignore", OneHotEncoder produces zeros for all encoded
# columns belonging to the unknown feature.
#
# OrdinalEncoder can instead assign a dedicated unknown value, such as -1.

# %%
new_cities = pd.DataFrame(
    {
        "city": [
            "Mannheim",
            "Berlin",
        ]
    }
)

new_city_encoded = one_hot_encoder.transform(new_cities)

new_city_encoded_df = pd.DataFrame(
    new_city_encoded,
    columns=city_feature_names,
    index=new_cities.index,
)

print("\nTransforming known and unknown cities:")
print(new_cities)
print(new_city_encoded_df)

print("\nEncoded representation for the unknown city:")
print(new_city_encoded_df.iloc[1].to_dict())

# %% [markdown]
# ## 6. Compare the representations
#
# One-hot encoding increases the number of features.
# Ordinal encoding typically uses one column but assigns integer codes.
#
# The correct approach depends on the semantics of the data and the
# assumptions of the model.

# %%
comparison = pd.DataFrame(
    {
        "Original city": data["city"],
        "One-hot encoded columns": [
            ", ".join(
                city_encoded_df.columns[city_encoded_df.iloc[i].to_numpy() == 1]
            )
            for i in range(len(data))
        ],
        "Original satisfaction": data["satisfaction"],
        "Ordinal value": satisfaction_encoded_df["satisfaction_encoded"],
    }
)

print("\nEncoding comparison:")
print(comparison)

print("\nFeature counts:")
print(f"Original categorical columns: {data.shape[1]}")
print(f"One-hot city columns: {city_encoded_df.shape[1]}")
print(f"Ordinal satisfaction columns: {satisfaction_encoded_df.shape[1]}")

# %% [markdown]
# ## Key takeaways
#
# - OneHotEncoder is suitable for nominal categories without a natural order.
# - OrdinalEncoder is appropriate when a meaningful order exists and is
#   explicitly defined.
# - Integer encoding of nominal categories can introduce unintended order.
# - handle_unknown="ignore" helps OneHotEncoder handle unseen categories.
# - OrdinalEncoder supports an explicit unknown-category value.
# - Fit encoders on training data only in predictive workflows.
# - Use ColumnTransformer and Pipeline to integrate encoding safely.

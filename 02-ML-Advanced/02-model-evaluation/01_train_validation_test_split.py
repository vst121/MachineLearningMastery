# %% 1. Imports
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# %% 2. Load the dataset
data = load_breast_cancer()

X = data.data
y = data.target

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Classes: {data.target_names}")
print(f"Class counts: {np.bincount(y)}")

# %% 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\nRandom split")
print(f"Training samples: {len(y_train)}")
print(f"Test samples:     {len(y_test)}")
print(f"Training class counts: {np.bincount(y_train)}")
print(f"Test class counts:     {np.bincount(y_test)}")

# %% 4. Split with stratification
X_train_strat, X_test_strat, y_train_strat, y_test_strat = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
)

print("\nStratified split")
print(f"Training class counts: {np.bincount(y_train_strat)}")
print(f"Test class counts:     {np.bincount(y_test_strat)}")

# %% 5. Compare class proportions
def show_class_proportions(name, labels):
    proportions = np.bincount(labels) / len(labels)

    print(f"\n{name}")
    for class_index, class_name in enumerate(data.target_names):
        print(
            f"{class_name}: "
            f"{proportions[class_index]:.3f}"
        )


show_class_proportions("Original dataset", y)
show_class_proportions("Random training set", y_train)
show_class_proportions("Random test set", y_test)
show_class_proportions("Stratified training set", y_train_strat)
show_class_proportions("Stratified test set", y_test_strat)

# %% 6. Create training, validation, and test sets
# Target proportions: 60% training, 20% validation, 20% test.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

print("\nTraining / validation / test split")
print(f"Training samples:   {len(y_train)}")
print(f"Validation samples: {len(y_val)}")
print(f"Test samples:       {len(y_test)}")

print(f"Training proportions:   {np.bincount(y_train) / len(y_train)}")
print(f"Validation proportions: {np.bincount(y_val) / len(y_val)}")
print(f"Test proportions:       {np.bincount(y_test) / len(y_test)}")

# %% 7. Verify reproducibility
X_train_repeat, X_test_repeat, y_train_repeat, y_test_repeat = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
)

print("\nReproducibility checks")
print(
    "Same training features:",
    np.array_equal(X_train_strat, X_train_repeat),
)
print(
    "Same test features:",
    np.array_equal(X_test_strat, X_test_repeat),
)
print(
    "Same training labels:",
    np.array_equal(y_train_strat, y_train_repeat),
)
print(
    "Same test labels:",
    np.array_equal(y_test_strat, y_test_repeat),
)

import pandas as pd


# Load datasets
harmful = pd.read_csv(
    "data/raw/jbb_harmful.csv"
)

benign = pd.read_csv(
    "data/raw/jbb_benign.csv"
)


print("=" * 70)
print("HARMFUL DATASET")
print("=" * 70)

print("Shape:", harmful.shape)

print("\nColumns:")
print(harmful.columns.tolist())

print("\nFirst 3 records:")
print(
    harmful[
        [
            "Index",
            "Goal",
            "Target",
            "Behavior",
            "Category",
            "Source",
            "label"
        ]
    ].head(3).to_string(index=False)
)


print("\n\n")
print("=" * 70)
print("BENIGN DATASET")
print("=" * 70)

print("Shape:", benign.shape)

print("\nColumns:")
print(benign.columns.tolist())

print("\nFirst 3 records:")
print(
    benign[
        [
            "Index",
            "Goal",
            "Target",
            "Behavior",
            "Category",
            "Source",
            "label"
        ]
    ].head(3).to_string(index=False)
)


print("\n\n")
print("=" * 70)
print("CATEGORY DISTRIBUTION")
print("=" * 70)

print("\nHarmful categories:")

print(
    harmful["Category"].value_counts()
)


print("\n\nBenign categories:")

print(
    benign["Category"].value_counts()
)


print("\n\n")
print("=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

combined = pd.concat(
    [harmful, benign],
    ignore_index=True
)

print(
    combined["label"].value_counts()
)
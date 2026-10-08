import pandas as pd
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Load the two JBB datasets
# ============================================================

harmful = pd.read_csv(
    RAW_DIR / "jbb_harmful.csv"
)

benign = pd.read_csv(
    RAW_DIR / "jbb_benign.csv"
)


# ============================================================
# 3. Keep only the columns required for our research
# ============================================================

columns = [
    "Index",
    "Goal",
    "Behavior",
    "Category",
    "Source",
    "label"
]


harmful = harmful[columns].copy()
benign = benign[columns].copy()


# ============================================================
# 4. Rename columns
# ============================================================

harmful = harmful.rename(
    columns={
        "Index": "pair_id",
        "Goal": "prompt",
        "label": "safety_label"
    }
)

benign = benign.rename(
    columns={
        "Index": "pair_id",
        "Goal": "prompt",
        "label": "safety_label"
    }
)


# ============================================================
# 5. Combine harmful + benign
# ============================================================

base_dataset = pd.concat(
    [
        harmful,
        benign
    ],
    ignore_index=True
)


# ============================================================
# 6. Basic cleaning
# ============================================================

base_dataset["prompt"] = (
    base_dataset["prompt"]
    .fillna("")
    .astype(str)
    .str.strip()
)


base_dataset["Behavior"] = (
    base_dataset["Behavior"]
    .fillna("")
    .astype(str)
    .str.strip()
)


base_dataset["Category"] = (
    base_dataset["Category"]
    .fillna("")
    .astype(str)
    .str.strip()
)


base_dataset["Source"] = (
    base_dataset["Source"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# 7. Remove empty prompts
# ============================================================

base_dataset = base_dataset[
    base_dataset["prompt"] != ""
].copy()


# ============================================================
# 8. Add a unique sample ID
# ============================================================

base_dataset.insert(
    0,
    "sample_id",
    range(len(base_dataset))
)


# ============================================================
# 9. Reorder columns
# ============================================================

base_dataset = base_dataset[
    [
        "sample_id",
        "pair_id",
        "prompt",
        "safety_label",
        "Behavior",
        "Category",
        "Source"
    ]
]


# ============================================================
# 10. Save
# ============================================================

output_file = (
    PROCESSED_DIR /
    "base_prompts.csv"
)


base_dataset.to_csv(
    output_file,
    index=False
)


# ============================================================
# 11. Print validation information
# ============================================================

print("=" * 70)
print("BASE DATASET CREATED")
print("=" * 70)

print()

print("Total samples:", len(base_dataset))

print(
    "Harmful samples:",
    (base_dataset["safety_label"] == 1).sum()
)

print(
    "Benign samples:",
    (base_dataset["safety_label"] == 0).sum()
)

print()

print("Missing values:")

print(
    base_dataset.isnull().sum()
)

print()

print("Category distribution:")

print(
    base_dataset["Category"].value_counts()
)

print()

print("Label distribution:")

print(
    base_dataset["safety_label"].value_counts()
)

print()

print("Saved to:")

print(output_file)
import pandas as pd


# ============================================================
# Load emoji dataset
# ============================================================

df = pd.read_csv(
    "data/processed/emoji_prompts.csv"
)


print("=" * 70)
print("EMOJI DATASET INSPECTION")
print("=" * 70)

print()

print("Total rows:", len(df))

print()


# ============================================================
# Show dataset columns
# ============================================================

print("Columns:")

print(df.columns.tolist())

print()


# ============================================================
# Show one complete sample
# ============================================================

sample_id = 0

sample = df[
    df["sample_id"] == sample_id
]


print("=" * 70)
print("SAMPLE", sample_id)
print("=" * 70)


for _, row in sample.iterrows():

    print()

    print("Variant:", row["attack_type"])

    print("Emoji:", row["emoji"])

    print("Emoji count:", row["emoji_count"])

    print("Emoji density:", row["emoji_density"])

    print()

    print("Prompt:")

    print(row["prompt"])

    print("-" * 70)


# ============================================================
# Check attack types
# ============================================================

print()
print("=" * 70)
print("ATTACK TYPE COUNTS")
print("=" * 70)

print(
    df["attack_type"].value_counts()
)


# ============================================================
# Check duplicate prompts
# ============================================================

print()
print("=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicate_count = df["prompt"].duplicated().sum()

print(
    "Duplicate prompt rows:",
    duplicate_count
)


# ============================================================
# Check emoji counts
# ============================================================

print()
print("=" * 70)
print("EMOJI COUNT DISTRIBUTION")
print("=" * 70)

print(
    df["emoji_count"].value_counts()
)


# ============================================================
# Final message
# ============================================================

print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
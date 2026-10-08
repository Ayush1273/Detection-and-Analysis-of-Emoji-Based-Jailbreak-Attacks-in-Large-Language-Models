from datasets import load_dataset
from pathlib import Path


# Create output directory
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)


print("Downloading JBB-Behaviors dataset...")
print()


# Load harmful split
print("Loading harmful behaviors...")

harmful = load_dataset(
    "JailbreakBench/JBB-Behaviors",
    "behaviors",
    split="harmful"
)


# Load benign split
print("Loading benign behaviors...")

benign = load_dataset(
    "JailbreakBench/JBB-Behaviors",
    "behaviors",
    split="benign"
)


# Convert to pandas
harmful_df = harmful.to_pandas()
benign_df = benign.to_pandas()


# Add our research labels
harmful_df["label"] = 1
benign_df["label"] = 0


# Save datasets
harmful_file = output_dir / "jbb_harmful.csv"
benign_file = output_dir / "jbb_benign.csv"


harmful_df.to_csv(
    harmful_file,
    index=False
)

benign_df.to_csv(
    benign_file,
    index=False
)


print()
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print()
print("Harmful samples:", len(harmful_df))
print("Benign samples :", len(benign_df))

print()
print("Columns:")
print(harmful_df.columns.tolist())

print()
print("Files created:")

print(harmful_file)
print(benign_file)
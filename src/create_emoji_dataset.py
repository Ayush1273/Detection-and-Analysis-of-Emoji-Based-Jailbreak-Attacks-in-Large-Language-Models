import pandas as pd
from pathlib import Path


# ============================================================
# 1. Paths
# ============================================================

INPUT_FILE = Path(
    "data/processed/base_prompts.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)

OUTPUT_FILE = OUTPUT_DIR / "emoji_prompts.csv"


# ============================================================
# 2. Load base dataset
# ============================================================

print("Loading base dataset...")

df = pd.read_csv(
    INPUT_FILE
)


print("Base samples:", len(df))


# ============================================================
# 3. Controlled emoji set
# ============================================================

# These are neutral emojis.
# We deliberately use a fixed set so that the experiment
# remains reproducible.

EMOJIS = [
    "😀",
    "🔍",
    "⚠️",
    "📝",
    "🔒",
    "🧠",
    "💡",
    "📌"
]


# ============================================================
# 4. Transformation functions
# ============================================================

def add_prefix(prompt, emoji):
    """
    Add one emoji before the prompt.
    """
    return f"{emoji} {prompt}"


def add_suffix(prompt, emoji):
    """
    Add one emoji after the prompt.
    """
    return f"{prompt} {emoji}"


def add_middle(prompt, emoji):
    """
    Insert one emoji approximately in the middle
    of the prompt.
    """

    words = prompt.split()

    if len(words) <= 2:
        middle = 1
    else:
        middle = len(words) // 2

    transformed = (
        words[:middle]
        + [emoji]
        + words[middle:]
    )

    return " ".join(transformed)


# ============================================================
# 5. Generate dataset
# ============================================================

rows = []


for _, row in df.iterrows():

    sample_id = row["sample_id"]
    pair_id = row["pair_id"]
    prompt = row["prompt"]

    safety_label = row["safety_label"]
    behavior = row["Behavior"]
    category = row["Category"]
    source = row["Source"]


    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    rows.append({
        "sample_id": sample_id,
        "pair_id": pair_id,
        "variant_id": f"{sample_id}_original",
        "original_prompt": prompt,
        "prompt": prompt,
        "safety_label": safety_label,
        "attack_type": "original",
        "emoji": "",
        "emoji_count": 0,
        "behavior": behavior,
        "category": category,
        "source": source
    })


    # Select one deterministic emoji.
    # Using sample_id makes the experiment reproducible.

    emoji = EMOJIS[
        int(sample_id) % len(EMOJIS)
    ]


    # --------------------------------------------------------
    # Prefix
    # --------------------------------------------------------

    transformed = add_prefix(
        prompt,
        emoji
    )

    rows.append({
        "sample_id": sample_id,
        "pair_id": pair_id,
        "variant_id": f"{sample_id}_prefix",
        "original_prompt": prompt,
        "prompt": transformed,
        "safety_label": safety_label,
        "attack_type": "prefix",
        "emoji": emoji,
        "emoji_count": 1,
        "behavior": behavior,
        "category": category,
        "source": source
    })


    # --------------------------------------------------------
    # Suffix
    # --------------------------------------------------------

    transformed = add_suffix(
        prompt,
        emoji
    )

    rows.append({
        "sample_id": sample_id,
        "pair_id": pair_id,
        "variant_id": f"{sample_id}_suffix",
        "original_prompt": prompt,
        "prompt": transformed,
        "safety_label": safety_label,
        "attack_type": "suffix",
        "emoji": emoji,
        "emoji_count": 1,
        "behavior": behavior,
        "category": category,
        "source": source
    })


    # --------------------------------------------------------
    # Middle insertion
    # --------------------------------------------------------

    transformed = add_middle(
        prompt,
        emoji
    )

    rows.append({
        "sample_id": sample_id,
        "pair_id": pair_id,
        "variant_id": f"{sample_id}_insertion",
        "original_prompt": prompt,
        "prompt": transformed,
        "safety_label": safety_label,
        "attack_type": "insertion",
        "emoji": emoji,
        "emoji_count": 1,
        "behavior": behavior,
        "category": category,
        "source": source
    })


# ============================================================
# 6. Create DataFrame
# ============================================================

emoji_df = pd.DataFrame(rows)


# ============================================================
# 7. Calculate emoji density
# ============================================================

emoji_df["word_count"] = (
    emoji_df["prompt"]
    .str.split()
    .str.len()
)


emoji_df["emoji_density"] = (
    emoji_df["emoji_count"]
    / emoji_df["word_count"].replace(0, 1)
)


# ============================================================
# 8. Reorder columns
# ============================================================

emoji_df = emoji_df[
    [
        "sample_id",
        "pair_id",
        "variant_id",
        "original_prompt",
        "prompt",
        "safety_label",
        "attack_type",
        "emoji",
        "emoji_count",
        "emoji_density",
        "behavior",
        "category",
        "source"
    ]
]


# ============================================================
# 9. Save
# ============================================================

emoji_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. Validation
# ============================================================

print()
print("=" * 70)
print("EMOJI DATASET CREATED")
print("=" * 70)

print()

print(
    "Total rows:",
    len(emoji_df)
)

print()

print(
    "Expected rows:",
    len(df) * 4
)

print()

print("Attack type distribution:")

print(
    emoji_df["attack_type"].value_counts()
)

print()

print("Safety label distribution:")

print(
    emoji_df["safety_label"].value_counts()
)

print()

print("Emoji distribution:")

print(
    emoji_df["emoji"].value_counts()
)

print()

print("Missing values:")

print(
    emoji_df.isnull().sum()
)

print()

print("Saved to:")

print(OUTPUT_FILE)
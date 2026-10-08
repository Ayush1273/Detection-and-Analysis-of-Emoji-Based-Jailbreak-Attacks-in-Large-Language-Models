import string
import unicodedata
from pathlib import Path
import pandas as pd

from src.utils.paths import EMOJI_PROMPTS_CSV, EMOJI_FEATURES_CSV


def get_emoji_characters(text: str) -> list[str]:
    """
    Extract characters that are classified as Unicode symbols ('S' category:
    So = Symbol, Other; Sk = Modifier; Sm = Math; Sc = Currency).
    """
    emojis = []
    for char in str(text):
        category = unicodedata.category(char)
        if category.startswith("S"):
            emojis.append(char)
    return emojis


def get_emoji_position(row: pd.Series) -> str:
    """
    Determine where the emoji occurs in the prompt.
    """
    attack_type = row["attack_type"]
    if attack_type == "original":
        return "none"
    if attack_type == "prefix":
        return "prefix"
    if attack_type == "suffix":
        return "suffix"
    if attack_type == "insertion":
        return "middle"
    return "unknown"


def get_unicode_values(symbols: list[str]) -> list[str]:
    """
    Return Unicode code points for detected symbols (e.g. U+1F600).
    """
    return [f"U+{ord(s):04X}" for s in symbols]


def get_unicode_categories(symbols: list[str]) -> list[str]:
    """
    Return Unicode general categories for detected symbols.
    """
    return [unicodedata.category(s) for s in symbols]


def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the complete feature extraction pipeline to the emoji prompts dataset.
    Preserves existing columns and appends textual, positional, and Unicode features.
    """
    df = df.copy()

    # Add unique_pair_id (safety_label + "_" + pair_id)
    df["unique_pair_id"] = (
        df["safety_label"].astype(str) + "_" + df["pair_id"].astype(str)
    )

    # 1. Text-level features
    df["text_length"] = df["prompt"].astype(str).str.len()
    df["word_count"] = df["prompt"].astype(str).str.split().str.len()
    df["space_count"] = df["prompt"].apply(lambda x: str(x).count(" "))
    df["punctuation_count"] = df["prompt"].apply(
        lambda x: sum(1 for c in str(x) if c in string.punctuation)
    )

    # 2. Emoji / Symbol features
    df["detected_symbols"] = df["prompt"].apply(get_emoji_characters)
    df["detected_symbol_count"] = df["detected_symbols"].apply(len)
    df["unique_symbol_count"] = df["detected_symbols"].apply(lambda x: len(set(x)))

    # 3. Position and Unicode features
    df["emoji_position"] = df.apply(get_emoji_position, axis=1)
    df["unicode_codepoints"] = df["detected_symbols"].apply(get_unicode_values)
    df["unicode_categories"] = df["detected_symbols"].apply(get_unicode_categories)

    # Reorder columns logically
    ordered_columns = [
        "sample_id",
        "pair_id",
        "unique_pair_id",
        "variant_id",
        "original_prompt",
        "prompt",
        "safety_label",
        "attack_type",
        "emoji",
        "emoji_count",
        "emoji_density",
        "emoji_position",
        "text_length",
        "word_count",
        "space_count",
        "punctuation_count",
        "detected_symbols",
        "detected_symbol_count",
        "unique_symbol_count",
        "unicode_codepoints",
        "unicode_categories",
        "behavior",
        "category",
        "source"
    ]

    # Keep all available columns matching ordered_columns
    final_cols = [c for c in ordered_columns if c in df.columns]
    # Add any remaining columns if any
    for col in df.columns:
        if col not in final_cols:
            final_cols.append(col)

    return df[final_cols]


def materialize_feature_dataset(
    input_csv: Path = EMOJI_PROMPTS_CSV,
    output_csv: Path = EMOJI_FEATURES_CSV
) -> pd.DataFrame:
    """
    Reads the base emoji prompts, runs feature extraction, and saves to CSV.
    """
    print(f"Loading input dataset from: {input_csv}")
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    print(f"Running feature extraction on {len(df)} samples...")
    featured_df = extract_all_features(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Feature dataset successfully saved to: {output_csv}")
    print(f"Dataset shape: {featured_df.shape} (Rows: {len(featured_df)}, Columns: {len(featured_df.columns)})")
    return featured_df


if __name__ == "__main__":
    materialize_feature_dataset()

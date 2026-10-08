import pandas as pd
from pathlib import Path
from src.utils.paths import EMOJI_FEATURES_CSV, LLM_EVAL_DATA_DIR


def create_pilot_40_dataset(
    input_csv: Path = EMOJI_FEATURES_CSV,
    output_csv: Path = LLM_EVAL_DATA_DIR / "pilot_40_prompts.csv",
    num_harmful_behaviors: int = 10
) -> pd.DataFrame:
    """
    Extracts the pilot evaluation subset:
    Takes the first `num_harmful_behaviors` unique harmful behaviors (safety_label == 1),
    and exports all 4 variants (original, prefix, suffix, insertion) for each behavior.
    Resulting in exactly num_harmful_behaviors * 4 = 40 prompts.
    """
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    # Filter to harmful behaviors
    harmful_df = df[df["safety_label"] == 1].copy()

    # Get distinct unique_pair_ids
    unique_pairs = harmful_df["unique_pair_id"].drop_duplicates().tolist()
    selected_pairs = unique_pairs[:num_harmful_behaviors]

    # Filter to all 4 variants for these selected pairs
    pilot_df = harmful_df[harmful_df["unique_pair_id"].isin(selected_pairs)].copy()

    # Sort deterministically by pair_id and attack_type
    attack_order = {"original": 0, "prefix": 1, "suffix": 2, "insertion": 3}
    pilot_df["order"] = pilot_df["attack_type"].map(attack_order)
    pilot_df = pilot_df.sort_values(by=["pair_id", "order"]).drop(columns=["order"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pilot_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("PILOT DATASET CREATED FOR REMOTE EVALUATION")
    print("=" * 70)
    print(f"File path: {output_csv}")
    print(f"Total samples: {len(pilot_df)} (Target: {num_harmful_behaviors * 4})")
    print(f"Behaviors selected: {pilot_df['behavior'].nunique()}")
    print("Variant breakdown:")
    print(pilot_df["attack_type"].value_counts().to_string())
    print("=" * 70)

    return pilot_df


if __name__ == "__main__":
    create_pilot_40_dataset()

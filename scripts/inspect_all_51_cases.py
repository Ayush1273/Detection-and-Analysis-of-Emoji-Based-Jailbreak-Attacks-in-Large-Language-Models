import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/processed/llm_evaluation/qwen_harmful_review_cases.csv')
print(f"Total review cases: {len(df)}")

categories = df['category'].unique()

for cat in categories:
    cat_df = df[df['category'] == cat]
    print("\n" + "#" * 80)
    print(f"CATEGORY: {cat} (Count: {len(cat_df)})")
    print("#" * 80)
    
    pairs = cat_df['unique_pair_id'].unique()
    for pid in pairs:
        pair_df = cat_df[cat_df['unique_pair_id'] == pid]
        print(f"\n--- PAIR: {pid} (Cases: {len(pair_df)}) ---")
        for idx, r in pair_df.iterrows():
            print(f"\n[Variant: {r['variant_id']}] | Attack: {r['attack_type']} | Auto Behavior: {r['response_behavior']}")
            print(f"Prompt: {r['prompt']}")
            print(f"Refusal Evidence: {r['refusal_evidence']}")
            print(f"Harmful Evidence: {r['harmful_content_evidence']}")
            print(f"Response Snippet:\n{r['response'][:450]}...")
            if len(r['response']) > 450:
                print(f"[Tail snippet]: ...{r['response'][-200:]}")

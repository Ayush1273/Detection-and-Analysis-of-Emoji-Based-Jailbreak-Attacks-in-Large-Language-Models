import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/processed/llm_evaluation/qwen_full_800_results.csv')
benign_refusals = df[(df['safety_label'] == 0) & (df['response_behavior'] == 'REFUSAL')].copy()
print(f"Total benign automated refusals: {len(benign_refusals)}")

for idx, r in benign_refusals.iterrows():
    print("=" * 80)
    print(f"[{r['variant_id']}] | Pair: {r['unique_pair_id']} | Cat: {r['category']} | Attack: {r['attack_type']}")
    print(f"Prompt: {r['prompt']}")
    print(f"Refusal Evidence: {r['refusal_evidence']}")
    print(f"Response Snippet:\n{r['response'][:350]}...")
    if len(r['response']) > 350:
        print(f"[Tail]: ...{r['response'][-150:]}")

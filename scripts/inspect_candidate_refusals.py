import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/processed/llm_evaluation/qwen_full_800_results.csv')
b_ref = df[(df['safety_label'] == 0) & (df['response_behavior'] == 'REFUSAL')].copy()

# List of cases that look like they might have actual refusals or hedging
candidate_refusals = [
    '100_suffix', '114_suffix', '118_prefix', '119_insertion', '121_suffix',
    '122_original', '122_prefix', '126_original', '128_suffix', '128_insertion',
    '129_insertion', '130_suffix', '136_insertion', '137_prefix', '138_suffix',
    '142_insertion', '145_insertion', '160_original', '164_insertion', '170_suffix',
    '190_insertion'
]

for vid in candidate_refusals:
    r = b_ref[b_ref['variant_id'] == vid].iloc[0]
    print("=" * 80)
    print(f"[{r['variant_id']}] | Cat: {r['category']} | Attack: {r['attack_type']}")
    print(f"Prompt: {r['prompt']}")
    print(f"Full Response:\n{r['response']}\n")

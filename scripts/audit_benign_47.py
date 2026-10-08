import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/processed/llm_evaluation/qwen_full_800_results.csv')
b_ref = df[(df['safety_label'] == 0) & (df['response_behavior'] == 'REFUSAL')].copy()

print(f"Total automated benign refusals: {len(b_ref)}")
print("-" * 140)
print(f"{'#':<3} | {'Variant ID':<15} | {'Category':<20} | {'Attack':<10} | {'Prompt':<45} | {'Response Head':<45}")
print("-" * 140)

i = 1
for idx, r in b_ref.iterrows():
    p = r['prompt'][:42].replace('\n', ' ')
    resp = r['response'][:42].replace('\n', ' ')
    cat = r['category'][:18]
    print(f"{i:<3} | {r['variant_id']:<15} | {cat:<20} | {r['attack_type']:<10} | {p:<45} | {resp:<45}")
    i += 1
print("-" * 140)

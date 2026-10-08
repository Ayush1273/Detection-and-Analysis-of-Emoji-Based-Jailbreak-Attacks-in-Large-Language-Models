import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/processed/llm_evaluation/qwen_full_800_results.csv')
b_red = df[(df['safety_label'] == 0) & (df['response_behavior'] == 'SAFE_REDIRECTION')].copy()

print(f"Total automated benign SAFE_REDIRECTION: {len(b_red)}")
print("-" * 120)
for idx, r in b_red.iterrows():
    p = r['prompt'][:50].replace('\n', ' ')
    resp_head = r['response'][:120].replace('\n', ' ')
    print(f"[{r['variant_id']}] ({r['attack_type']}, {r['category']}) | Prompt: {p} | Head: {resp_head}")

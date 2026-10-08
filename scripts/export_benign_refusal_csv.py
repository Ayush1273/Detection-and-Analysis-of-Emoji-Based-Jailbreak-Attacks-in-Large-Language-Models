import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from compute_benign_full_metrics import b

b47 = b[b['response_behavior'] == 'REFUSAL'].copy()

# Add confidence and clean columns
b47['confidence'] = 'HIGH'

cols = [
    'variant_id', 'unique_pair_id', 'category', 'attack_type', 
    'prompt', 'response', 'response_behavior', 'human_label', 
    'confidence', 'human_reason'
]
b47_out = b47[cols].rename(columns={
    'unique_pair_id': 'pair_id',
    'response_behavior': 'automated_response_behavior',
    'human_reason': 'reason'
})

b47_out.to_csv('data/processed/llm_evaluation/qwen_benign_refusal_cases_validated.csv', index=False)
print(f"Exported {len(b47_out)} validated benign refusal cases to data/processed/llm_evaluation/qwen_benign_refusal_cases_validated.csv")

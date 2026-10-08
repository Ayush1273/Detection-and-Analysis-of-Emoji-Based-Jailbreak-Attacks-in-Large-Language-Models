import pandas as pd

df = pd.read_csv('data/processed/llm_evaluation/qwen_harmful_review_cases.csv')
print(f"Total harmful review cases: {len(df)}")
print("-" * 120)
print(f"{'Variant ID':<15} | {'Category':<22} | {'Attack Type':<10} | {'Behavior':<18} | {'Prompt Snippet':<40}")
print("-" * 120)

for idx, r in df.iloc[:15].iterrows():
    p_clean = r['prompt'].encode('ascii', 'backslashreplace').decode()[:38].replace('\n', ' ')
    cat_clean = r['category'][:22]
    print(f"{r['variant_id']:<15} | {cat_clean:<22} | {r['attack_type']:<10} | {r['response_behavior']:<18} | {p_clean:<40}")
print("-" * 120)

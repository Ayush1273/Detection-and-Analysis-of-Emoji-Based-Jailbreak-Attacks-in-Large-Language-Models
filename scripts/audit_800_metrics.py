import pandas as pd
import json

df = pd.read_csv('data/processed/llm_evaluation/qwen_full_800_results.csv')

print('=== 1. BASIC COUNTS & INTEGRITY ===')
print('Rows:', len(df), 'Columns:', len(df.columns))
print('Unique variant_ids:', df['variant_id'].nunique())
print('Unique prompt pairs:', df['unique_pair_id'].nunique())
print('Empty prompts:', (df['prompt'].str.strip() == '').sum())
print('Empty responses:', (df['response'].str.strip() == '').sum())
print('Generation errors (error notna):', df['error'].notna().sum())

print('\n=== 2. SAFETY LABELS & CATEGORIES ===')
print('Safety label distribution:')
print(df['safety_label'].value_counts())
print('\nAttack type distribution:')
print(df['attack_type'].value_counts())
print('\nCategory distribution:')
print(df['category'].value_counts())

print('\n=== 3. RESPONSE BEHAVIOR OVERALL & SPLIT ===')
print('Overall Response Behavior:')
print(df['response_behavior'].value_counts())
print(df['response_behavior'].value_counts(normalize=True) * 100)

print('\nResponse Behavior by Safety Label:')
print(pd.crosstab(df['safety_label'], df['response_behavior'], margins=True))
print(pd.crosstab(df['safety_label'], df['response_behavior'], normalize='index') * 100)

print('\n=== 4. ATTACK TYPE BREAKDOWN (HARMFUL ONLY) ===')
df_harmful = df[df['safety_label'] == 1]
print(pd.crosstab(df_harmful['attack_type'], df_harmful['response_behavior'], margins=True))
print((pd.crosstab(df_harmful['attack_type'], df_harmful['response_behavior'], normalize='index') * 100).round(2))

print('\n=== 5. ATTACK TYPE BREAKDOWN (BENIGN ONLY) ===')
df_benign = df[df['safety_label'] == 0]
print(pd.crosstab(df_benign['attack_type'], df_benign['response_behavior'], margins=True))
print((pd.crosstab(df_benign['attack_type'], df_benign['response_behavior'], normalize='index') * 100).round(2))

print('\n=== 6. PAIR TRANSITIONS IN HARMFUL PROMPTS ===')
harmful_pairs = df_harmful.groupby('unique_pair_id')
transitions = []

for pid, grp in harmful_pairs:
    orig = grp[grp['attack_type'] == 'original']
    if orig.empty:
        continue
    orig_b = orig.iloc[0]['response_behavior']
    for at in ['prefix', 'suffix', 'insertion']:
        v = grp[grp['attack_type'] == at]
        if not v.empty:
            vb = v.iloc[0]['response_behavior']
            transitions.append({
                'pair_id': pid,
                'category': grp.iloc[0]['category'],
                'original': orig_b,
                'variant_type': at,
                'variant': vb,
                'is_same': (orig_b == vb)
            })

tdf = pd.DataFrame(transitions)
print('Total harmful variant transitions evaluated:', len(tdf))
print('Identical behavior count:', tdf['is_same'].sum(), f"({tdf['is_same'].mean()*100:.2f}%)")
print('\nTransition Matrix (Original vs Variant):')
print(pd.crosstab(tdf['original'], tdf['variant'], margins=True))

diffs = tdf[~tdf['is_same']]
print(f'\nTotal discrepant transitions: {len(diffs)}')
print(diffs.to_string())

print('\n=== 7. LATENCY AND LENGTH STATISTICS ===')
for name, sub_df in [('All 800', df), ('Benign (400)', df_benign), ('Harmful (400)', df_harmful)]:
    print(f'--- {name} ---')
    print('Latency (s): mean={:.2f}, std={:.2f}, median={:.2f}, min={:.2f}, max={:.2f}'.format(
        sub_df['generation_time_seconds'].mean(),
        sub_df['generation_time_seconds'].std(),
        sub_df['generation_time_seconds'].median(),
        sub_df['generation_time_seconds'].min(),
        sub_df['generation_time_seconds'].max()
    ))
    resp_chars = sub_df['response'].str.len()
    resp_words = sub_df['response'].str.split().str.len()
    print('Chars: mean={:.1f}, std={:.1f}, median={:.1f}, min={}, max={}'.format(
        resp_chars.mean(), resp_chars.std(), resp_chars.median(), resp_chars.min(), resp_chars.max()
    ))
    print('Words: mean={:.1f}, std={:.1f}, median={:.1f}, min={}, max={}'.format(
        resp_words.mean(), resp_words.std(), resp_words.median(), resp_words.min(), resp_words.max()
    ))

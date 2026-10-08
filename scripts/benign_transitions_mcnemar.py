import sys
import pandas as pd
import numpy as np
from scipy.stats import mcnemar

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Run previous labeling logic to get b dataframe
from compute_benign_full_metrics import b

# Group by unique_pair_id (100 pairs)
pairs = b.groupby('unique_pair_id')

transitions = []
for pid, grp in pairs:
    orig = grp[grp['attack_type'] == 'original']
    if orig.empty:
        continue
    orig_row = orig.iloc[0]
    orig_label = orig_row['human_label']
    
    for at in ['prefix', 'suffix', 'insertion']:
        v = grp[grp['attack_type'] == at]
        if not v.empty:
            v_row = v.iloc[0]
            v_label = v_row['human_label']
            transitions.append({
                'pair_id': pid,
                'category': orig_row['category'],
                'variant_type': at,
                'orig_label': orig_label,
                'variant_label': v_label,
                'is_same': (orig_label == v_label),
                'orig_over_refusal': (orig_label == 'OVER_REFUSAL'),
                'variant_over_refusal': (v_label == 'OVER_REFUSAL')
            })

tdf = pd.DataFrame(transitions)
print(f"Total evaluated paired transitions: {len(tdf)}")
print(f"Identical behavior across transitions: {tdf['is_same'].sum()} / {len(tdf)} ({tdf['is_same'].mean()*100:.2f}%)")

print("\n=== TRANSITION MATRIX: ORIGINAL vs VARIANT (All 300 transitions) ===")
print(pd.crosstab(tdf['orig_label'], tdf['variant_label'], margins=True))

print("\n=== MCNEMAR TESTS FOR OVER-REFUSAL (Binary: Over-Refusal vs Other) ===")
for at in ['prefix', 'suffix', 'insertion']:
    sub = tdf[tdf['variant_type'] == at]
    # Build 2x2 table:
    # Rows: Original (0: No, 1: Yes)
    # Cols: Variant (0: No, 1: Yes)
    ct = pd.crosstab(sub['orig_over_refusal'], sub['variant_over_refusal'], dropna=False)
    # Ensure full 2x2 shape
    for r in [False, True]:
        if r not in ct.index:
            ct.loc[r] = 0
    for c in [False, True]:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct.loc[[False, True], [False, True]]
    
    b_cell = ct.loc[False, True] # Orig No, Variant Yes
    c_cell = ct.loc[True, False] # Orig Yes, Variant No
    
    # McNemar test
    # If b + c < 25, use exact binomial test
    res = mcnemar(ct.values, exact=True)
    print(f"\n--- Original vs {at.capitalize()} ---")
    print("Contingency Table (Orig [rows] vs Variant [cols]):")
    print(ct)
    print(f"Discordant pairs: b (0->1) = {b_cell}, c (1->0) = {c_cell}")
    print(f"McNemar test statistic: {res.statistic}, p-value: {res.pvalue:.4f}")
    if res.pvalue < 0.05:
        print("Conclusion: Statistically significant difference (p < 0.05)")
    else:
        print("Conclusion: No statistically significant difference (p >= 0.05); null hypothesis retained.")

import sys
import pandas as pd
from scipy.stats import binomtest

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from compute_benign_full_metrics import b

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

print("\n=== TABLE 5: BEHAVIORAL TRANSITION MATRIX (Original -> Variant, N=300) ===")
tm = pd.crosstab(tdf['orig_label'], tdf['variant_label'], margins=True)
print(tm)

print("\nNormalized transition matrix (%):")
print((pd.crosstab(tdf['orig_label'], tdf['variant_label'], normalize='index') * 100).round(2))

print("\n=== MCNEMAR TESTS FOR OVER-REFUSAL (Binary: Over-Refusal vs Other) ===")
for at in ['prefix', 'suffix', 'insertion']:
    sub = tdf[tdf['variant_type'] == at]
    ct = pd.crosstab(sub['orig_over_refusal'], sub['variant_over_refusal'], dropna=False)
    # Ensure complete 2x2 with explicit boolean index
    ct = ct.reindex(index=[False, True], columns=[False, True], fill_value=0)
    
    b_cell = int(ct.loc[False, True]) # Orig No, Variant Yes
    c_cell = int(ct.loc[True, False]) # Orig Yes, Variant No
    
    # Exact binomial test on discordant pairs (b and c)
    n_discordant = b_cell + c_cell
    if n_discordant == 0:
        pval = 1.0
    else:
        res = binomtest(b_cell, n_discordant, 0.5, alternative='two-sided')
        pval = res.pvalue
        
    print(f"\n--- Original vs {at.capitalize()} ---")
    print(f"Contingency Table (Orig [rows] vs {at.capitalize()} [cols]):")
    print(ct)
    print(f"Discordant pairs: b (Orig 0 -> Var 1) = {b_cell}, c (Orig 1 -> Var 0) = {c_cell}, Total discordant = {n_discordant}")
    print(f"Exact McNemar / Binomial p-value: {pval:.4f}")
    if pval < 0.05:
        print("Decision: Reject H0 (Statistically significant shift)")
    else:
        print("Decision: Fail to reject H0 (No statistically significant difference)")

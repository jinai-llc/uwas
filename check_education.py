#!/usr/bin/env python3
"""Quick check of education column values in ADNI data."""
import pandas as pd
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'ADNI_merged_data.csv'
df = pd.read_csv(path, low_memory=False)

# Find education columns
edu_cols = [c for c in df.columns if 'educat' in c.lower() or 'education' in c.lower()]
print(f"Education columns found: {edu_cols}\n")

for col in edu_cols:
    vals = pd.to_numeric(df[col], errors='coerce')
    print(f"{'='*50}")
    print(f"Column: {col}")
    print(f"  dtype:    {df[col].dtype}")
    print(f"  non-null: {vals.notna().sum()} / {len(df)}")
    print(f"  min:      {vals.min()}")
    print(f"  max:      {vals.max()}")
    print(f"  mean:     {vals.mean():.2f}")
    print(f"  median:   {vals.median():.1f}")
    print(f"  std:      {vals.std():.2f}")
    print(f"\n  Value counts (top 20):")
    vc = vals.dropna().astype(int).value_counts().sort_index()
    for v, n in vc.head(20).items():
        print(f"    {v:4d} years: {n:5d} subjects")
    print()

# ── CN vs AD breakdown ──
dx_col = None
for c in ['PHC_Diagnosis', 'DX_LABEL', 'DX_bl', 'DX', 'AD_Label']:
    if c in df.columns:
        dx_col = c
        break

if dx_col and edu_cols:
    col = edu_cols[0]
    vals = pd.to_numeric(df[col], errors='coerce')
    dx = df[dx_col].astype(str).str.strip().str.upper()

    # Map to CN/AD
    cn_map = {'CN': True, 'NL': True, 'NORMAL': True, '0': True, '0.0': True}
    ad_map = {'AD': True, 'DEMENTIA': True, 'ALZHEIMER': True, '1': True, '1.0': True}
    cn_mask = dx.map(cn_map).fillna(False).astype(bool)
    ad_mask = dx.map(ad_map).fillna(False).astype(bool)

    print(f"{'='*50}")
    print(f"Education by Diagnosis ({col})")
    print(f"{'='*50}")
    for label, mask in [('CN', cn_mask), ('AD', ad_mask)]:
        v = vals[mask].dropna()
        if len(v) == 0:
            print(f"  {label}: no data")
            continue
        print(f"\n  {label} (n={len(v)}):")
        print(f"    mean:   {v.mean():.2f} ± {v.std():.2f}")
        print(f"    median: {v.median():.1f}")
        print(f"    range:  {v.min():.0f} – {v.max():.0f}")
        vc = v.astype(int).value_counts().sort_index()
        total = len(v)
        print(f"    distribution:")
        max_pct = max(n / total * 100 for n in vc.values)
        for yr, n in vc.items():
            pct = n / total * 100
            bar = '█' * int(pct / max_pct * 30)
            print(f"      {yr:3d} yrs: {n:4d} ({pct:5.1f}%) {bar}")

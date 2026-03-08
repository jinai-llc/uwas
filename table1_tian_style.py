#!/usr/bin/env python3
"""
Table 1 Generator — Tian et al. (Translational Psychiatry) Style
=================================================================
Generates a compact demographics table with:
  - CN vs AD only (no Overall column)
  - mean ± 95% CI for continuous, % for categorical
  - Full numeric p-values (0.0013, <0.0001)
  - Minimal footnotes
  - Three-rule table (top, header, bottom)

Usage:
  python table1_tian_style.py --input ADNI_merged_data.csv
  python table1_tian_style.py --input ADNI_merged_data.csv --phs PHS_scores.csv --apoe APOERES.csv
  python table1_tian_style.py --input ADNI_merged_data.csv --consents consents.csv -o results/
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')


# =============================================================================
# COLUMN DETECTION
# =============================================================================

def find_col(df, candidates, contains=None):
    """Find a column by exact match, then case-insensitive, then contains."""
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    if contains:
        for kw in contains:
            matches = [c for c in df.columns if kw.lower() in c.lower()]
            if matches:
                return matches[0]
    return None


def detect_columns(df):
    """Auto-detect ADNI column names."""
    mapping = {}

    # Diagnosis / target
    mapping['dx'] = find_col(df,
        ['PHC_Diagnosis', 'DX', 'DIAGNOSIS', 'diagnosis', 'AD_Label', 'DX_LABEL'],
        contains=['diagnosis', 'dx_bl'])

    # Demographics
    mapping['age'] = find_col(df,
        ['PHC_Age_Biomarker_clin1', 'PHC_Age_clin', 'PTAGE', 'AGE', 'Age'],
        contains=['age_biomarker', 'ptage', 'age_at'])
    mapping['sex'] = find_col(df,
        ['PHC_Sex_clin2', 'PHC_Sex_clin', 'PTGENDER', 'SEX', 'Sex'],
        contains=['sex_clin', 'ptgender'])
    mapping['education'] = find_col(df,
        ['PHC_Education_clin2', 'PHC_Education_clin', 'PTEDUCAT', 'Education'],
        contains=['education_clin', 'pteducat'])
    mapping['race'] = find_col(df,
        ['PHC_Race_clin2', 'PHC_Race_clin', 'PTRACCAT', 'Race'],
        contains=['race_clin', 'ptraccat'])
    mapping['ethnicity'] = find_col(df,
        ['PHC_Ethnicity_clin', 'PTETHCAT', 'Ethnicity'],
        contains=['ethnicity', 'ptethcat', 'hispanic'])

    # Vascular risk
    mapping['htn'] = find_col(df,
        ['PHC_Hypertension_clin', 'Hypertension', 'HTN'],
        contains=['hypertension', 'htn'])
    mapping['diabetes'] = find_col(df,
        ['PHC_Diabetes_clin', 'Diabetes', 'DM'],
        contains=['diabetes', '_dm_'])
    mapping['bmi'] = find_col(df,
        ['PHC_BMI_clin', 'BMI', 'bmi'],
        contains=['bmi', 'body_mass'])
    mapping['smoking'] = find_col(df,
        ['MH16SMOK_medi77', 'MH16SMOK', 'Smoking'],
        contains=['smok', 'mh16'])
    mapping['cvd'] = find_col(df,
        ['MH4CARD_medi77', 'MH4CARD', 'CardiovascularDisease'],
        contains=['mh4card', 'cardiovascular'])

    # Genetic
    mapping['apoe'] = find_col(df,
        ['APOE_Carrier', 'APOE_clin', 'APOE_gene', 'APOE4'],
        contains=['apoe_carrier', 'apoe4', 'apoe_e4'])
    mapping['apoe_alleles'] = find_col(df,
        ['APOE_Allele_Count', 'APOE_gene1', 'APOE_Count'],
        contains=['apoe_allele', 'apoe_count', 'apoe_gene'])
    mapping['phs'] = find_col(df,
        ['PHS', 'PHS_gene', 'PolygeneticHazardScore'],
        contains=['polygenic', 'phs'])

    # Cognitive
    mapping['mmse'] = find_col(df,
        ['MMSCORE_cogn1', 'MMSE', 'MMSCORE'],
        contains=['mmscore', 'mmse'])
    mapping['moca'] = find_col(df,
        ['MoCA_Total_cogn3', 'MoCA_Total', 'MOCA'],
        contains=['moca_total', 'moca'])
    mapping['cdrsb'] = find_col(df,
        ['CDRSB_cogn4', 'CDRSB', 'CDR_SB'],
        contains=['cdrsb', 'cdr_sb'])
    mapping['cdr_global'] = find_col(df,
        ['CDGLOBAL_cogn4', 'CDGLOBAL', 'CDR_Global'],
        contains=['cdglobal', 'cdr_global'])
    mapping['adas'] = find_col(df,
        ['ADAS13_cogn5', 'ADAS13', 'ADAS_Cog13'],
        contains=['adas13', 'adas_cog'])

    # Neuroimaging availability
    mapping['has_t1'] = None  # computed later
    mapping['has_dti'] = None  # computed later

    return mapping


# =============================================================================
# TARGET ENCODING
# =============================================================================

def encode_target(df, dx_col):
    """Encode CN=0, AD=1, drop MCI. Handles numeric (1/2/3, 1.0/2.0/3.0) and string labels."""
    raw = df[dx_col].copy()
    print(f"\n  Diagnosis column: {dx_col}")
    print(f"  Value counts:")
    for val, cnt in raw.value_counts(dropna=False).head(10).items():
        print(f"    '{val}' (type={type(val).__name__}): {cnt}")

    y = pd.Series(np.nan, index=df.index)

    # Try numeric first — convert everything to float for comparison
    raw_num = pd.to_numeric(raw, errors='coerce')
    n_numeric = raw_num.notna().sum()

    if n_numeric > 0.5 * len(raw):
        # Round to handle 1.0/2.0/3.0 float encoding
        raw_rounded = raw_num.dropna().round(0).astype(int)
        unique_ints = sorted(raw_rounded.unique())
        print(f"  Detected numeric encoding: {unique_ints}")

        raw_int = raw_num.round(0)

        if 1 in unique_ints and 3 in unique_ints:
            # Standard ADNI: 1=CN, 2=MCI, 3=AD
            y[raw_int == 1] = 0
            y[raw_int == 3] = 1
            print(f"  Mapping: 1→CN, 3→AD, 2→dropped")
        elif 0 in unique_ints and 1 in unique_ints and len(unique_ints) == 2:
            # Already binary: 0=CN, 1=AD
            y[raw_int == 0] = 0
            y[raw_int == 1] = 1
            print(f"  Mapping: 0→CN, 1→AD")
        elif 0 in unique_ints and 2 in unique_ints:
            # 0=CN, 1=MCI, 2=AD
            y[raw_int == 0] = 0
            y[raw_int == 2] = 1
            print(f"  Mapping: 0→CN, 2→AD, 1→dropped")
        else:
            print(f"  WARNING: Unrecognized numeric encoding: {unique_ints}")
    else:
        # String labels — comprehensive matching
        raw_str = raw.astype(str).str.strip()
        cn_found, ad_found, other = [], [], []
        for val in raw_str.unique():
            v = val.upper()
            if v in ['CN', 'NL', 'NORMAL', 'CONTROL', 'HC', 'CU',
                      'COGNITIVELY NORMAL', 'HEALTHY CONTROL', 'NO DEMENTIA',
                      'COGNITIVELY NORMAL (CN)', 'CN (COGNITIVELY NORMAL)']:
                y[raw_str == val] = 0
                cn_found.append(val)
            elif v in ['AD', 'DEMENTIA', 'ALZHEIMER', "ALZHEIMER'S", 'ALZHEIMERS',
                        "ALZHEIMER'S DISEASE", 'ALZHEIMERS DISEASE', 'ALZHEIMER DISEASE',
                        'AD/DEMENTIA', 'DEMENTIA/AD', 'AD (ALZHEIMER)', 'PROBABLE AD',
                        'PROBABLE ALZHEIMER']:
                y[raw_str == val] = 1
                ad_found.append(val)
            else:
                other.append(val)
        print(f"  CN labels matched: {cn_found}")
        print(f"  AD labels matched: {ad_found}")
        if other:
            print(f"  Unmatched (dropped): {other[:10]}")

    mask = y.notna()
    n_cn = int((y == 0).sum())
    n_ad = int((y == 1).sum())
    n_drop = int((~mask).sum())
    print(f"  → {n_cn} CN, {n_ad} AD, {n_drop} dropped")

    if n_cn == 0 and n_ad == 0:
        print(f"\n  *** FATAL: No CN or AD subjects found! ***")
        print(f"  Check the values printed above and add them to the label matching logic.")
        sys.exit(1)

    return y, mask


# =============================================================================
# LONGITUDINAL DEDUP
# =============================================================================

def dedup_longitudinal(df, y):
    """Latest visit for AD, baseline for CN."""
    rid_col = find_col(df, ['RID', 'PTID', 'Subject', 'rid'])
    viscode_col = find_col(df, ['VISCODE', 'VISCODE2', 'Visit', 'viscode'])

    if rid_col is None:
        print("  No RID column — skipping dedup")
        return df, y

    n_before = len(df)
    df = df.copy()
    df['_y'] = y if isinstance(y, np.ndarray) else y.values

    if viscode_col:
        # Sort: AD by latest visit, CN by earliest
        df['_visit_order'] = df[viscode_col].astype(str).str.extract(r'(\d+)').astype(float)
        df['_visit_order'] = df['_visit_order'].fillna(0)

        cn = df[df['_y'] == 0].sort_values('_visit_order', ascending=True).drop_duplicates(rid_col, keep='first')
        ad = df[df['_y'] == 1].sort_values('_visit_order', ascending=False).drop_duplicates(rid_col, keep='first')
        df = pd.concat([cn, ad], ignore_index=True)
        df.drop(columns=['_visit_order'], inplace=True, errors='ignore')
    else:
        df = df.drop_duplicates(rid_col, keep='last')

    y = df['_y'].values
    df.drop(columns=['_y'], inplace=True, errors='ignore')
    print(f"  Dedup: {n_before} → {len(df)} subjects")
    return df, y


# =============================================================================
# STATISTICS
# =============================================================================

def ci95(vals):
    """Mean ± 95% CI string."""
    vals = vals.dropna()
    if len(vals) < 2:
        return "—", np.nan, np.nan
    m = vals.mean()
    se = vals.std() / np.sqrt(len(vals))
    ci = 1.96 * se
    return f"{m:.1f} ± {ci:.2f}", m, ci


def pval_str(p):
    """Format p-value in Tian style: full numeric or <0.0001."""
    if p < 0.0001:
        return "<0.0001", True
    elif p < 0.05:
        return f"{p:.4f}", True
    else:
        return f"{p:.4f}", False


def pct(series, val=1):
    """Compute percentage of val in series."""
    s = pd.to_numeric(series, errors='coerce')
    valid = s.dropna()
    if len(valid) == 0:
        return "—", 0
    n_pos = (valid > 0).sum() if val == 1 else (valid == val).sum()
    pct_val = n_pos / len(valid) * 100
    return f"{pct_val:.1f}%", n_pos


def compare_continuous(cn_vals, ad_vals):
    """Wilcoxon rank-sum for continuous."""
    cn_clean = cn_vals.dropna()
    ad_clean = ad_vals.dropna()
    if len(cn_clean) < 5 or len(ad_clean) < 5:
        return "—", False
    _, p = stats.mannwhitneyu(cn_clean, ad_clean, alternative='two-sided')
    return pval_str(p)


def compare_categorical(cn_vals, ad_vals):
    """Chi-squared for categorical."""
    cn_clean = cn_vals.dropna()
    ad_clean = ad_vals.dropna()
    if len(cn_clean) < 5 or len(ad_clean) < 5:
        return "—", False

    combined = pd.concat([cn_clean.rename('val').to_frame().assign(grp=0),
                          ad_clean.rename('val').to_frame().assign(grp=1)])
    ct = pd.crosstab(combined['val'], combined['grp'])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return "—", False
    try:
        _, p, _, _ = stats.chi2_contingency(ct)
        return pval_str(p)
    except:
        return "—", False


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_table1(input_file, label_col='PHC_Diagnosis', phs_file=None,
               apoe_file=None, consents_file=None, output_dir='.'):
    """Generate Table 1 in Tian et al. style."""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  TABLE 1 GENERATOR — Tian et al. Style")
    print("=" * 60)

    # ── Load data ──
    print(f"\n  Loading: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    # ── Merge PHS if provided ──
    if phs_file and os.path.exists(phs_file):
        print(f"  Merging PHS from: {phs_file}")
        phs_df = pd.read_csv(phs_file)
        rid_col_phs = find_col(phs_df, ['RID', 'PTID', 'rid'])
        rid_col_main = find_col(df, ['RID', 'PTID', 'rid'])
        if rid_col_phs and rid_col_main:
            phs_col = find_col(phs_df, ['PHS', 'phs', 'PolygeneticHazardScore'],
                               contains=['phs', 'polygenic'])
            if phs_col:
                phs_merge = phs_df[[rid_col_phs, phs_col]].drop_duplicates(rid_col_phs)
                phs_merge.columns = [rid_col_main, 'PHS']
                df = df.merge(phs_merge, on=rid_col_main, how='left')
                print(f"    PHS merged: {df['PHS'].notna().sum()} subjects")

    # ── Merge APOE if provided ──
    if apoe_file and os.path.exists(apoe_file):
        print(f"  Merging APOE from: {apoe_file}")
        apoe_df = pd.read_csv(apoe_file)
        rid_col_apoe = find_col(apoe_df, ['RID', 'PTID', 'rid'])
        rid_col_main = find_col(df, ['RID', 'PTID', 'rid'])
        if rid_col_apoe and rid_col_main:
            # Look for APOE genotype columns
            gen1 = find_col(apoe_df, ['APGEN1', 'APOE1'], contains=['apgen1'])
            gen2 = find_col(apoe_df, ['APGEN2', 'APOE2'], contains=['apgen2'])
            if gen1 and gen2:
                apoe_df = apoe_df.drop_duplicates(rid_col_apoe)
                apoe_df['APOE_e4_count'] = ((apoe_df[gen1] == 4).astype(int) +
                                             (apoe_df[gen2] == 4).astype(int))
                apoe_df['APOE_Carrier'] = (apoe_df['APOE_e4_count'] > 0).astype(int)
                merge_cols = [rid_col_apoe, 'APOE_Carrier', 'APOE_e4_count']
                apoe_merge = apoe_df[merge_cols].copy()
                apoe_merge.columns = [rid_col_main, 'APOE_Carrier', 'APOE_e4_count']
                # Only fill if not already present
                if 'APOE_Carrier' not in df.columns:
                    df = df.merge(apoe_merge, on=rid_col_main, how='left')
                else:
                    df = df.merge(apoe_merge, on=rid_col_main, how='left', suffixes=('', '_new'))
                    mask = df['APOE_Carrier'].isna()
                    if 'APOE_Carrier_new' in df.columns:
                        df.loc[mask, 'APOE_Carrier'] = df.loc[mask, 'APOE_Carrier_new']
                        df.drop(columns=['APOE_Carrier_new', 'APOE_e4_count_new'],
                                inplace=True, errors='ignore')
                print(f"    APOE merged: {df['APOE_Carrier'].notna().sum()} subjects")

    # ── Detect columns ──
    cols = detect_columns(df)
    print(f"\n  Column detection:")
    for k, v in cols.items():
        if v:
            print(f"    {k:15s} → {v}")

    # ── Encode target ──
    # First try: if AD_Label already exists (from prior pipeline), use it directly
    if 'AD_Label' in df.columns:
        print(f"\n  Found pre-computed AD_Label column — using directly")
        y_pre = pd.to_numeric(df['AD_Label'], errors='coerce')
        mask = y_pre.isin([0, 1])
        y = y_pre.copy()
        df = df[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True).astype(int).values
        print(f"  CN={int((y==0).sum())}, AD={int((y==1).sum())}")
    elif cols['dx'] is not None:
        y, mask = encode_target(df, cols['dx'])
        df = df[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True).astype(int).values
    else:
        print("ERROR: No diagnosis column found!")
        print(f"  Tried: PHC_Diagnosis, DX, DIAGNOSIS, AD_Label")
        print(f"  Available columns: {[c for c in df.columns if 'diag' in c.lower() or 'dx' in c.lower() or 'label' in c.lower()]}")
        sys.exit(1)

    # ── Dedup ──
    df, y = dedup_longitudinal(df, y)

    cn = df[y == 0]
    ad = df[y == 1]
    n_cn, n_ad = len(cn), len(ad)
    print(f"\n  Final cohort: CN={n_cn}, AD={n_ad}, Total={n_cn + n_ad}")

    # ── Detect neuroimaging availability ──
    mri_cols = [c for c in df.columns if '_mri_' in c.lower() or 'freesurfer' in c.lower()
                or c.startswith('ST') or 'hippocampus' in c.lower()]
    dti_cols = [c for c in df.columns if 'dti' in c.lower() or '_freewater' in c.lower()
                or '_fw_' in c.lower() or '_md_' in c.lower() or '_ad_' in c.lower()
                or '_rd_' in c.lower() or '_fa_' in c.lower()]

    # ── Compute statistics ──
    print("\n  Computing statistics...\n")
    rows = []

    def add_continuous(label, col_key, superscript='a'):
        col = cols.get(col_key)
        if col and col in df.columns:
            cn_ci, _, _ = ci95(cn[col].apply(pd.to_numeric, errors='coerce'))
            ad_ci, _, _ = ci95(ad[col].apply(pd.to_numeric, errors='coerce'))
            p_str, sig = compare_continuous(
                cn[col].apply(pd.to_numeric, errors='coerce'),
                ad[col].apply(pd.to_numeric, errors='coerce'))
            n_avail = pd.to_numeric(df[col], errors='coerce').notna().sum()
            rows.append(('', f'{label}<sup>{superscript}</sup>', cn_ci, ad_ci, p_str, sig, n_avail))
        else:
            rows.append(('', f'{label}', '—', '—', '—', False, 0))

    def add_binary(label, col_key, cat_label=None, superscript=None):
        col = cols.get(col_key)
        if col and col in df.columns:
            cn_pct, _ = pct(cn[col])
            ad_pct, _ = pct(ad[col])
            p_str, sig = compare_categorical(
                pd.to_numeric(cn[col], errors='coerce').dropna().apply(lambda x: 1 if x > 0 else 0),
                pd.to_numeric(ad[col], errors='coerce').dropna().apply(lambda x: 1 if x > 0 else 0))
            n_avail = pd.to_numeric(df[col], errors='coerce').notna().sum()
            sup = f'<sup>{superscript}</sup>' if superscript else ''
            label_display = f'{label}{sup}'
            if cat_label:
                rows.append((cat_label, label_display, cn_pct, ad_pct, p_str, sig, n_avail))
            else:
                rows.append(('', label_display, cn_pct, ad_pct, p_str, sig, n_avail))
        else:
            rows.append(('', label, '—', '—', '—', False, 0))

    # ── Demographics ──
    add_continuous('Age', 'age')
    add_continuous('Education years', 'education')

    # Education ≤ 12 years (high school or less)
    edu_col = cols.get('education')
    if edu_col and edu_col in df.columns:
        cn_edu = pd.to_numeric(cn[edu_col], errors='coerce').dropna()
        ad_edu = pd.to_numeric(ad[edu_col], errors='coerce').dropna()
        cn_le12 = (cn_edu <= 12).sum() / len(cn_edu) * 100 if len(cn_edu) > 0 else 0
        ad_le12 = (ad_edu <= 12).sum() / len(ad_edu) * 100 if len(ad_edu) > 0 else 0
        # Fisher's exact test on ≤12 vs >12
        cn_bin = (cn_edu <= 12).astype(int)
        ad_bin = (ad_edu <= 12).astype(int)
        p_str, sig = compare_categorical(cn_bin, ad_bin)
        rows.append(('', '\u2003\u2264 High school (\u226412 yrs)', f'{cn_le12:.1f}%', f'{ad_le12:.1f}%', p_str, sig, len(cn_edu) + len(ad_edu)))
        # Education ≥ 16 years (undergraduate or higher)
        cn_ge16 = (cn_edu > 16).sum() / len(cn_edu) * 100 if len(cn_edu) > 0 else 0
        ad_ge16 = (ad_edu > 16).sum() / len(ad_edu) * 100 if len(ad_edu) > 0 else 0
        cn_bin16 = (cn_edu > 16).astype(int)
        ad_bin16 = (ad_edu > 16).astype(int)
        p_str16, sig16 = compare_categorical(cn_bin16, ad_bin16)
        rows.append(('', '\u2003> Undergraduate (>16 yrs)', f'{cn_ge16:.1f}%', f'{ad_ge16:.1f}%', p_str16, sig16, len(cn_edu) + len(ad_edu)))

    # Sex
    if cols['sex'] and cols['sex'] in df.columns:
        sex_col = cols['sex']
        cn_sex = pd.to_numeric(cn[sex_col], errors='coerce').dropna()
        ad_sex = pd.to_numeric(ad[sex_col], errors='coerce').dropna()
        # Female: depends on encoding (1=M/2=F or 0=M/1=F)
        unique_vals = sorted(pd.to_numeric(df[sex_col], errors='coerce').dropna().unique())
        if len(unique_vals) == 2 and max(unique_vals) == 2:
            # 1=M, 2=F encoding
            cn_fpct = (cn_sex == 2).sum() / len(cn_sex) * 100
            ad_fpct = (ad_sex == 2).sum() / len(ad_sex) * 100
        else:
            # 0=M, 1=F encoding
            cn_fpct = (cn_sex == 1).sum() / len(cn_sex) * 100
            ad_fpct = (ad_sex == 1).sum() / len(ad_sex) * 100
        p_str, sig = compare_categorical(cn_sex, ad_sex)
        rows.append(('', 'Sex % female', f'{cn_fpct:.2f}%', f'{ad_fpct:.2f}%', p_str, sig, len(cn_sex) + len(ad_sex)))

    # APOE
    if 'APOE_Carrier' in df.columns:
        apoe_col = 'APOE_Carrier'
    elif cols['apoe'] and cols['apoe'] in df.columns:
        apoe_col = cols['apoe']
    else:
        apoe_col = None

    if apoe_col:
        cn_apoe = pd.to_numeric(cn[apoe_col], errors='coerce').dropna()
        ad_apoe = pd.to_numeric(ad[apoe_col], errors='coerce').dropna()
        cn_carrier = (cn_apoe > 0).sum() / len(cn_apoe) * 100 if len(cn_apoe) > 0 else 0
        ad_carrier = (ad_apoe > 0).sum() / len(ad_apoe) * 100 if len(ad_apoe) > 0 else 0
        p_str, sig = compare_categorical(
            cn_apoe.apply(lambda x: 1 if x > 0 else 0),
            ad_apoe.apply(lambda x: 1 if x > 0 else 0))
        n_avail = cn_apoe.notna().sum() + ad_apoe.notna().sum()
        rows.append(('', f'<em>APOE</em> ε4 % carrier<sup>b</sup>',
                     f'{cn_carrier:.2f}%', f'{ad_carrier:.2f}%', p_str, sig, n_avail))

    # PHS
    add_continuous('Polygenic hazard score', 'phs', superscript='a,c')

    # BMI
    add_continuous('BMI, kg/m²', 'bmi')

    # ── Vascular risk / Race ──
    # Race breakdown
    if cols['race'] and cols['race'] in df.columns:
        race_col = cols['race']
        cn_race = cn[race_col].dropna().astype(str)
        ad_race = ad[race_col].dropna().astype(str)
        all_race = df[race_col].dropna().astype(str)

        # Get unique categories
        race_cats = all_race.value_counts().index.tolist()

        # Map numeric to labels if needed
        race_map = {
            '1': 'American Indian/Alaska Native',
            '2': 'Asian',
            '3': 'Native Hawaiian/Pacific Islander',
            '4': 'Black or African American',
            '5': 'White',
            '6': 'More than one race',
            '7': 'Unknown',
        }

        # Check if values are numeric
        if all(r.replace('.0', '').isdigit() for r in race_cats if r not in ['nan', '']):
            race_cats_mapped = [(r, race_map.get(r.replace('.0', ''), f'Other ({r})')) for r in race_cats]
        else:
            race_cats_mapped = [(r, r) for r in race_cats]

        # Chi-squared for overall race
        p_str, sig = compare_categorical(cn_race, ad_race)

        first = True
        for val, label in race_cats_mapped:
            if label in ['Unknown', 'nan', '']:
                continue
            cn_rpct = (cn_race == val).sum() / len(cn_race) * 100 if len(cn_race) > 0 else 0
            ad_rpct = (ad_race == val).sum() / len(ad_race) * 100 if len(ad_race) > 0 else 0
            cat = 'Race' if first else ''
            rows.append((cat, label, f'{cn_rpct:.2f}%', f'{ad_rpct:.2f}%',
                         p_str if first else '', sig if first else False, 0))
            first = False

    # Vascular risk
    add_binary('Hypertension', 'htn', cat_label='Vascular risk')
    add_binary('Diabetes mellitus', 'diabetes')
    add_binary('Smoking history', 'smoking')
    add_binary('Cardiovascular disease', 'cvd', superscript='d')

    # Neuroimaging
    if mri_cols:
        cn_has_mri = cn[mri_cols].notna().any(axis=1).sum()
        ad_has_mri = ad[mri_cols].notna().any(axis=1).sum()
        cn_mri_pct = cn_has_mri / n_cn * 100
        ad_mri_pct = ad_has_mri / n_ad * 100
        rows.append(('Neuroimaging', 'T1-weighted MRI',
                     f'{cn_mri_pct:.1f}%', f'{ad_mri_pct:.1f}%', '', False, 0))
    if dti_cols:
        cn_has_dti = cn[dti_cols].notna().any(axis=1).sum()
        ad_has_dti = ad[dti_cols].notna().any(axis=1).sum()
        cn_dti_pct = cn_has_dti / n_cn * 100
        ad_dti_pct = ad_has_dti / n_ad * 100
        rows.append(('', 'DTI scan',
                     f'{cn_dti_pct:.1f}%', f'{ad_dti_pct:.1f}%', '', False, 0))

    # ── Print summary ──
    print(f"  {'Feature':40s} {'CN':>15s} {'AD':>15s} {'p':>12s}")
    print("  " + "-" * 85)
    for cat, label, cn_v, ad_v, p_v, sig, n_a in rows:
        # Strip HTML for console
        clean = label.replace('<sup>', '').replace('</sup>', '').replace('<em>', '').replace('</em>', '')
        prefix = f"[{cat}] " if cat else "  "
        print(f"  {prefix}{clean:38s} {cn_v:>15s} {ad_v:>15s} {p_v:>12s} {'***' if sig else ''}")

    # ── Compute footnote availability ──
    avail_notes = []
    if apoe_col:
        n_apoe = pd.to_numeric(df[apoe_col], errors='coerce').notna().sum()
        avail_notes.append(f'APOE genotyping available for <em>n</em> = {n_apoe} ({n_apoe/(n_cn+n_ad)*100:.1f}%)')
    if cols['phs'] and cols['phs'] in df.columns:
        n_phs = pd.to_numeric(df[cols['phs']], errors='coerce').notna().sum()
        avail_notes.append(f'PHS available for <em>n</em> = {n_phs} ({n_phs/(n_cn+n_ad)*100:.1f}%)')
    if cols['cvd'] and cols['cvd'] in df.columns:
        n_cvd = pd.to_numeric(df[cols['cvd']], errors='coerce').notna().sum()
        avail_notes.append(f'Cardiovascular disease available for <em>n</em> = {n_cvd} ({n_cvd/(n_cn+n_ad)*100:.1f}%)')

    # ── Generate HTML ──
    table_rows = []
    for i, (cat, label, cn_v, ad_v, p_v, sig, _) in enumerate(rows):
        p_class = 'ps' if sig else 'pv'
        # Handle rowspan for race p-value
        p_cell = f'<td class="{p_class}">{p_v}</td>' if p_v != '' else ''
        # Count how many race rows follow for rowspan
        if cat == 'Race':
            # Count subsequent rows with empty cat until next non-empty cat
            n_race = 1
            for j in range(i + 1, len(rows)):
                if rows[j][0] == '' and rows[j][4] == '':
                    n_race += 1
                else:
                    break
            p_cell = f'<td class="{p_class}" rowspan="{n_race}">{p_v}</td>'

        cat_cell = f'<td>{cat}</td>' if cat or True else '<td></td>'

        is_last = (i == len(rows) - 1)
        row_class = ' class="last"' if is_last else ''
        table_rows.append(f'<tr{row_class}><td>{cat}</td><td>{label}</td><td>{cn_v}</td><td>{ad_v}</td>{p_cell}</tr>')

    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Table 1</title>
<style>
@import url("https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Source+Sans+3:wght@300;400;500;600;700&display=swap");
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Source Sans 3','Helvetica Neue',sans-serif;background:#fff;color:#1a1a2e;
     padding:36px 40px;display:flex;justify-content:center;}}
.w{{width:620px;}}
.ti{{font-family:'Source Serif 4',Georgia,serif;font-size:12.5px;font-weight:700;margin-bottom:14px;line-height:1.4;}}
table{{width:100%;border-collapse:collapse;font-size:10.5px;line-height:1.5;}}
thead tr:first-child th{{border-top:2px solid #000;padding-top:7px;}}
thead tr:last-child th{{border-bottom:1px solid #000;padding-bottom:6px;}}
tbody tr:last-child td{{border-bottom:2px solid #000;padding-bottom:6px;}}
th{{font-weight:700;text-align:center;padding:4px 10px;font-size:10px;vertical-align:bottom;}}
th:first-child,th:nth-child(2){{text-align:left;}}
td{{padding:3px 10px;text-align:center;font-size:10.5px;}}
td:first-child{{text-align:left;width:80px;font-weight:600;color:#555;font-size:10px;}}
td:nth-child(2){{text-align:left;}}
.pv{{font-size:10px;color:#333;font-style:italic;}}
.ps{{font-size:10px;font-weight:600;color:#000;font-style:italic;}}
.fn{{margin-top:10px;font-size:9px;color:#555;line-height:1.6;}}
.fn p{{margin-bottom:1px;}}
</style></head><body>
<div class="w">
<div class="ti">Table 1 Demographics of the ADNI cohort.</div>
<table><thead><tr>
<th></th>
<th></th>
<th>CN (<em>n</em> = {n_cn})</th>
<th>AD (<em>n</em> = {n_ad})</th>
<th><em>p</em> value</th>
</tr></thead><tbody>
{"".join(table_rows)}
</tbody></table>
<div class="fn">
<p><sup>a</sup>Data represented by mean ± 95% CI or percentage. Two-tailed Wilcoxon rank-sum test for continuous variables, χ² and Fisher's exact probability test for categorical variables.</p>
<p><sup>b</sup>{avail_notes[0] if len(avail_notes) > 0 else ""}. <sup>c</sup>{avail_notes[1] if len(avail_notes) > 1 else ""}. <sup>d</sup>{avail_notes[2] if len(avail_notes) > 2 else ""}.</p>
</div>
</div></body></html>'''

    output_path = os.path.join(output_dir, 'table1_tian_style.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  Table 1 saved to: {output_path}")

    # ── Also save CSV for reference ──
    csv_rows = []
    for cat, label, cn_v, ad_v, p_v, sig, n_a in rows:
        clean = label.replace('<sup>', '').replace('</sup>', '').replace('<em>', '').replace('</em>', '')
        csv_rows.append({
            'Category': cat,
            'Feature': clean,
            'CN': cn_v,
            'AD': ad_v,
            'p_value': p_v,
            'significant': sig,
            'n_available': n_a
        })
    csv_path = os.path.join(output_dir, 'table1_stats.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  Stats CSV saved to: {csv_path}")
    print(f"\n{'='*60}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Table 1 Generator — Tian et al. Style',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python table1_tian_style.py --input ADNI_merged_data.csv
  python table1_tian_style.py --input ADNI_merged_data.csv --phs PHS_scores.csv --apoe APOERES.csv
  python table1_tian_style.py -i ADNI_merged_data.csv --phs PHS.csv --apoe APOERES.csv -o results/
""")

    parser.add_argument('--input', '-i', required=True, help='Path to merged ADNI CSV')
    parser.add_argument('--label', '-l', default='PHC_Diagnosis', help='Target column (default: PHC_Diagnosis)')
    parser.add_argument('--phs', default=None, help='Path to PHS CSV (optional)')
    parser.add_argument('--apoe', default=None, help='Path to APOE genotype CSV (optional)')
    parser.add_argument('--consents', '-c', default=None, help='Path to consents CSV (optional, not used currently)')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory')

    args = parser.parse_args()

    run_table1(
        input_file=args.input,
        label_col=args.label,
        phs_file=args.phs,
        apoe_file=args.apoe,
        consents_file=args.consents,
        output_dir=args.output_dir,
    )

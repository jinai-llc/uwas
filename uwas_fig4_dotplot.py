#!/usr/bin/env python3
"""
Figure 4: Minimal Discriminative Feature Set via Backward Elimination
=====================================================================
Pools clinical (excl. HTN/DM) + MRI features, performs backward elimination
per HTN x DM group, identifies minimal feature sets, and shows feature-feature
association heatmaps.

Usage:
  python uwas_fig4.py -i ADNI_merged_data.csv -c consents.csv --apoe APOERES.csv -o .
"""
import argparse, os, re, sys, warnings, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# FEATURE UTILITIES
# ═══════════════════════════════════════════════════════════════

LEAKAGE_PATTERNS = [
    'mmse', 'moca', 'cdr', 'adas', 'faq', 'npi', 'gds',
    'ravlt', 'digitscor', 'trailb', 'traila', 'boston',
    'catflu', 'ecog', 'dx', 'diagnosis', 'label',
    'mem_clin', 'exf_clin', 'lan_clin', 'vsp_clin',
]
COGNITIVE_PATTERNS = ['mem_clin', 'exf_clin', 'lan_clin', 'vsp_clin', 'adni_mem', 'adni_ef', 'composite']


def has_suffix(col, suffix):
    cl = col.lower()
    sl = suffix.lower()
    if cl.endswith(sl): return True
    if re.search(re.escape(sl) + r'_?\d+$', cl): return True
    return False

def is_leakage(col):
    cl = col.lower()
    return any(p in cl for p in LEAKAGE_PATTERNS)

def is_cognitive(col):
    cl = col.lower()
    return any(p in cl for p in COGNITIVE_PATTERNS)


def classify_features(df, target_col='AD_Label'):
    """Classify columns into clinical and MRI pools. Return col_info with HTN/DM."""
    clinical, mri = [], []
    col_info = {}
    for col in df.columns:
        cl = col.lower()
        if col == target_col or is_leakage(col) or is_cognitive(col):
            continue
        if any(p in cl for p in ['hypertension', '_htn']) and has_suffix(col, '_clin'):
            col_info['htn'] = col; continue
        if any(p in cl for p in ['diabetes', '_dm_']) and has_suffix(col, '_clin'):
            col_info['dm'] = col; continue
        if has_suffix(col, '_clin') or has_suffix(col, '_medi'):
            clinical.append(col)
        elif has_suffix(col, '_gene'):
            clinical.append(col)
        elif has_suffix(col, '_mri') or has_suffix(col, '_othe'):
            if any(p in cl for p in ['_pet', '_csf', '_tau_', 'amyloid', 'suvr']):
                continue
            mri.append(col)
    return clinical, mri, col_info


def annotate_feature(col):
    """Human-readable name for a feature — comprehensive annotation."""
    nl = col.lower()
    base = re.sub(r'(_clin\d*|_medi\d*|_gene\d*|_mri_?\d+|_othe\d*)$', '',
                  nl, flags=re.IGNORECASE)
    base = re.sub(r'^phc_', '', base)

    # ── Known clinical features ──
    known_clinical = {
        'age_cognition': 'Age', 'age_cardiovascularrisk': 'Age',
        'age_biomarker': 'Age',
        'sex': 'Sex (M to F)', 'education': 'Education (Low to High)', 'bmi': 'BMI',
        'hypertension': 'Hypertension', 'diabetes': 'Diabetes',
        'heart': 'Heart disease', 'stroke': 'Stroke',
        'smoker': 'Smoking', 'smoking': 'Smoking',
        'sbp': 'Systolic BP', 'bp_med': 'BP medication',
        'ascvd_10y': 'ASCVD 10-yr risk',
        'phs': 'PHS', 'genotype': 'APOE Carrier', 'apoe_e4_carrier': 'APOE Carrier',
        'tomm40': 'TOMM40',
        'mhpsych': 'Hx: Psychiatric', 'mh2neurl': 'Hx: Neurological',
        'mh3head': 'Hx: Head injury', 'mh4card': 'Hx: Cardiovascular',
        'mh5resp': 'Hx: Respiratory', 'mh6hepat': 'Hx: Hepatic',
        'mh7derm': 'Hx: Dermatologic', 'mh8muscl': 'Hx: Musculoskeletal',
        'mh9endo': 'Hx: Endocrine', 'mh10gast': 'Hx: Gastrointestinal',
        'mh11hema': 'Hx: Hematologic', 'mh12rena': 'Hx: Renal',
        'mh14alch': 'Hx: Alcohol abuse', 'mh16smok': 'Hx: Smoking',
        'mh17mali': 'Hx: Malignancy', 'mh18surg': 'Hx: Surgical',
        'ihsurg': 'Hx: Surgical', 'ihchron': 'Hx: Chronic illness',
        'ihsever': 'Illness severity',
    }
    for pat, label in known_clinical.items():
        if pat in base:
            return label

    # ── MRI: FreeSurfer UCSF (ST##XX) codes ──
    _UCSF_REGIONS = {
        'st9': 'L Banks STS',       'st80': 'R Banks STS',
        'st10': 'L Caud Ant Cing',  'st81': 'R Caud Ant Cing',
        'st11': 'L Hippocampus',    'st88': 'R Hippocampus',
        'st12': 'L Caud Mid Front', 'st82': 'R Caud Mid Front',
        'st13': 'L Cuneus',         'st84': 'R Cuneus',
        'st14': 'L Inf Lat Vent',   'st91': 'R Inf Lat Vent',
        'st22': 'L Insula',         'st86': 'R Insula',
        'st23': 'L Lingual',        'st87': 'R Lingual',
        'st24': 'L Entorhinal',     'st83': 'R Entorhinal',
        'st25': 'L Rostral Mid Fr', 'st94': 'R Rostral Mid Fr',
        'st26': 'L Fusiform',       'st85': 'R Fusiform',
        'st29': 'L Inf Temporal',   'st90': 'R Inf Temporal',
        'st30': 'L Isthmus Cing',   'st89': 'R Isthmus Cing',
        'st31': 'L Lat Occ',        'st92': 'R Lat Occ',
        'st32': 'L Lat Orbitofr',   'st93': 'R Lat Orbitofr',
        'st33': 'L Precentral',     'st120': 'R Precentral',
        'st34': 'L Med Orbitofr',   'st95': 'R Med Orbitofr',
        'st35': 'L Mid Temporal',   'st96': 'R Mid Temporal',
        'st36': 'L Parahipp',       'st97': 'R Parahipp',
        'st37': 'L Paracentral',    'st98': 'R Paracentral',
        'st38': 'L Pars Operc',     'st99': 'R Pars Operc',
        'st39': 'L Pars Triang',    'st100': 'R Pars Triang',
        'st40': 'L Pericalcarine',  'st101': 'R Pericalcarine',
        'st41': 'L Postcentral',    'st102': 'R Postcentral',
        'st42': 'L Precuneus',      'st103': 'R Precuneus',
        'st43': 'L Rost Ant Cing',  'st104': 'R Rost Ant Cing',
        'st44': 'L Sup Frontal',    'st105': 'R Sup Frontal',
        'st45': 'L Sup Parietal',   'st106': 'R Sup Parietal',
        'st46': 'L Sup Temporal',   'st107': 'R Sup Temporal',
        'st47': 'L Supramarginal',  'st108': 'R Supramarginal',
        'st48': 'L Frontal Pole',   'st118': 'R Frontal Pole',
        'st49': 'L Temporal Pole',  'st119': 'R Temporal Pole',
        'st50': 'L Thalamus',       'st109': 'R Thalamus',
        'st52': 'L Caudate',        'st110': 'R Caudate',
        'st53': 'L Putamen',        'st111': 'R Putamen',
        'st54': 'L Pallidum',       'st112': 'R Pallidum',
        'st55': 'L Amygdala',       'st113': 'R Amygdala',
        'st56': 'L Accumbens',      'st114': 'R Accumbens',
        'st58': 'L Cerebral WM',    'st115': 'R Cerebral WM',
        'st59': 'L Cerebral Ctx',   'st116': 'R Cerebral Ctx',
        'st60': 'L Lat Vent',       'st117': 'R Lat Vent',
        'st62': 'ICV',              'st70': 'Brainstem',
        'st71': 'L Ventral DC',     'st72': 'R Ventral DC',
        # Hippocampal subfields
        'st121': 'L Hipp CA1',      'st135': 'R Hipp CA1',
        'st122': 'L Hipp CA2/3',    'st136': 'R Hipp CA2/3',
        'st123': 'L Hipp CA4/DG',   'st137': 'R Hipp CA4/DG',
        'st124': 'L Subiculum',     'st138': 'R Subiculum',
        'st125': 'L Presubiculum',  'st139': 'R Presubiculum',
        'st126': 'L Parasubiculum', 'st140': 'R Parasubiculum',
        'st127': 'L Fimbria',       'st141': 'R Fimbria',
        'st128': 'L Mol Layer HP',  'st142': 'R Mol Layer HP',
        'st129': 'L GC-ML-DG',     'st143': 'R GC-ML-DG',
        'st130': 'L HATA',          'st144': 'R HATA',
        'st131': 'L HP Tail',       'st145': 'R HP Tail',
        'st132': 'L HP Body',       'st146': 'R HP Body',
        'st133': 'L HP Head',       'st147': 'R HP Head',
    }
    _UCSF_METRICS = {'sv': 'Vol', 'cv': 'Ctx Vol', 'sa': 'Area', 'ta': 'Thick',
                      'ts': 'Thick SD', 'hs': 'HS Vol'}
    m_ucsf = re.match(r'st(\d+)([a-z]{2})', base)
    if m_ucsf:
        code = 'st{}'.format(m_ucsf.group(1))
        metric = _UCSF_METRICS.get(m_ucsf.group(2), m_ucsf.group(2).upper())
        region = _UCSF_REGIONS.get(code, 'Reg{}'.format(m_ucsf.group(1)))
        return '{} {}'.format(region, metric)

    # ── MRI: T1-segmentation DTI features ──
    _T1SEG_ABBREV = {
        'hippocampus': 'Hippocampus', 'itg': 'Inf Temporal G.',
        'fug': 'Fusiform G.', 'fusiform_gyrus': 'Fusiform G.',
        'phg': 'Parahippocampal G.', 'parahippocampal_gyrus': 'Parahippocampal G.',
        'amygdala': 'Amygdala', 'entorhinal': 'Entorhinal',
        'stg': 'Sup Temporal G.', 'mtg': 'Mid Temporal G.',
        'precuneus': 'Precuneus', 'cingulate': 'Cingulate',
        'insula': 'Insula', 'thalamus': 'Thalamus',
        'caudate': 'Caudate', 'putamen': 'Putamen', 'pallidum': 'Pallidum',
    }
    _DTI_METRICS = {'md': 'MD', 'ad': 'AD', 'rd': 'RD', 'fa': 'FA', 'freewater': 'FW'}
    _DTI_STATS = {'mean': 'mean', 'median': 'med', 'std': 'SD'}
    mri_base = base  # already stripped of phc_

    # t1seg_left/right_REGION_METRIC_STAT
    m_t1 = re.match(r't1seg_(left|right)_(.+?)_(md|ad|rd|fa|freewater)_(mean|median|std)$', mri_base)
    if m_t1:
        side = 'L' if m_t1.group(1) == 'left' else 'R'
        region_raw = m_t1.group(2)
        region = region_raw
        for abbr, label in _T1SEG_ABBREV.items():
            if abbr in region_raw:
                region = label; break
        else:
            region = region_raw.replace('_', ' ').title()
        metric = _DTI_METRICS.get(m_t1.group(3), m_t1.group(3).upper())
        stat = _DTI_STATS.get(m_t1.group(4), m_t1.group(4))
        return '{} {} {} ({})'.format(side, region, metric, stat)

    # t1seg_left/right_REGION_volume/thick/area
    m_t1b = re.match(r't1seg_(left|right)_(.+?)_(volume|thick|area)$', mri_base)
    if m_t1b:
        side = 'L' if m_t1b.group(1) == 'left' else 'R'
        region_raw = m_t1b.group(2)
        region = region_raw
        for abbr, label in _T1SEG_ABBREV.items():
            if abbr in region_raw:
                region = label; break
        else:
            region = region_raw.replace('_', ' ').title()
        return '{} {} {}'.format(side, region, m_t1b.group(3).title())

    # Bilateral t1seg (no left/right prefix)
    m_bi = re.match(r't1seg_(.+?)_(md|ad|rd|fa|freewater|volume)_(mean|median|std)$', mri_base)
    if m_bi:
        region_raw = m_bi.group(1)
        region = region_raw
        for abbr, label in _T1SEG_ABBREV.items():
            if abbr in region_raw:
                region = label; break
        else:
            region = region_raw.replace('_', ' ').title()
        metric = _DTI_METRICS.get(m_bi.group(2), m_bi.group(2).title())
        stat = _DTI_STATS.get(m_bi.group(3), m_bi.group(3))
        return '{} {} ({})'.format(region, metric, stat)

    # ── MRI: JHU atlas DTI features ──
    m_jhu = re.match(r'jhu_(.+?)_(fwcorrected_)?(md|ad|rd|fa|freewater)_(mean|median|std)$', mri_base)
    if m_jhu:
        region_raw = m_jhu.group(1)
        fwc = 'FWc ' if m_jhu.group(2) else ''
        metric = _DTI_METRICS.get(m_jhu.group(3), m_jhu.group(3).upper())
        stat = _DTI_STATS.get(m_jhu.group(4), m_jhu.group(4))
        side = ''
        region = region_raw
        if region.endswith('_right'):
            side = 'R '; region = region[:-6]
        elif region.endswith('_left'):
            side = 'L '; region = region[:-5]
        elif region.startswith('right_'):
            side = 'R '; region = region[6:]
        elif region.startswith('left_'):
            side = 'L '; region = region[5:]
        region = region.replace('_', ' ').title()
        return '{}{} {}{} ({})'.format(side, region, fwc, metric, stat)

    # ── MRI: Known special features ──
    _KNOWN_MRI = {
        'hci_2014': 'Hippocampal Vol Index (HCI)',
        'hci': 'Hippocampal Vol Index (HCI)',
        'right_sub_vol': 'R Subcortical Vol',
        'left_sub_vol': 'L Subcortical Vol',
        'tau_metaroi': 'Tau Meta-ROI',
        'ab_metaroi': 'Amyloid Meta-ROI',
        'sroi': 'Summary ROI',
        'right_ca1_vol': 'R Hipp CA1 Vol',
        'left_ca1_vol': 'L Hipp CA1 Vol',
        'right_subiculum_vol': 'R Subiculum Vol',
        'left_subiculum_vol': 'L Subiculum Vol',
        'right_hippocampus_vol': 'R Hippocampus Vol',
        'left_hippocampus_vol': 'L Hippocampus Vol',
        'right_hippocampus': 'R Hippocampus Vol',
        'left_hippocampus': 'L Hippocampus Vol',
        'right_hippo': 'R Hippocampus Vol',
        'left_hippo': 'L Hippocampus Vol',
        'hippocampus_right': 'R Hippocampus Vol',
        'hippocampus_left': 'L Hippocampus Vol',
        'hippo_right': 'R Hippocampus Vol',
        'hippo_left': 'L Hippocampus Vol',
        'right_entorhinal': 'R Entorhinal Vol',
        'left_entorhinal': 'L Entorhinal Vol',
        'right_amygdala': 'R Amygdala Vol',
        'left_amygdala': 'L Amygdala Vol',
        'right_fusiform': 'R Fusiform Vol',
        'left_fusiform': 'L Fusiform Vol',
        'right_parahippocampal': 'R Parahippocampal Vol',
        'left_parahippocampal': 'L Parahippocampal Vol',
        'right_inferiortemporal': 'R Inf Temporal Vol',
        'left_inferiortemporal': 'L Inf Temporal Vol',
        'right_precuneus': 'R Precuneus Vol',
        'left_precuneus': 'L Precuneus Vol',
        'right_lateralventricle': 'R Lat Vent Vol',
        'left_lateralventricle': 'L Lat Vent Vol',
        'right_ventricle': 'R Lat Vent Vol',
        'left_ventricle': 'L Lat Vent Vol',
        'icv': 'Intracranial Vol (ICV)',
        'wholebrain': 'Whole Brain Vol',
        'whole_brain': 'Whole Brain Vol',
        'brainstem': 'Brainstem Vol',
        'cerebral_white_matter': 'Cerebral WM Vol',
        'cortical_vol': 'Total Cortical Vol',
        'md_cgh': 'Cingulum (Hipp) MD',
        'fa_cgh': 'Cingulum (Hipp) FA',
        'rd_cgh': 'Cingulum (Hipp) RD',
        'ad_cgh': 'Cingulum (Hipp) AD',
        'md_cgc': 'Cingulum (Cing) MD',
        'fa_cgc': 'Cingulum (Cing) FA',
        'md_fx': 'Fornix MD',
        'fa_fx': 'Fornix FA',
        'rd_fx': 'Fornix RD',
        'md_ifo': 'Inf Fronto-Occ Fasciculus MD',
        'fa_ifo': 'Inf Fronto-Occ Fasciculus FA',
        'md_unc': 'Uncinate MD',
        'fa_unc': 'Uncinate FA',
        'md_slf': 'Sup Long Fasciculus MD',
        'fa_slf': 'Sup Long Fasciculus FA',
        'md_ilf': 'Inf Long Fasciculus MD',
        'fa_ilf': 'Inf Long Fasciculus FA',
        'aci': 'Anterior Commissure Index',
        'cci': 'Corpus Callosum Index',
        'measure1': 'Cortical Summary Measure',
        'measure_1': 'Cortical Summary Measure',
        'spare_ad': 'SPARE-AD Score',
        'wmh': 'White Matter Hyperintensities',
        'wmh_vol': 'WMH Volume',
    }
    for pat, label in _KNOWN_MRI.items():
        if pat in base:
            return label

    # ── ADNI _othe columns: try to clean up ──
    m_othe = re.match(r'(.+?)_othe\d*$', nl)
    if m_othe:
        raw = m_othe.group(1)
        raw = re.sub(r'^phc_', '', raw, flags=re.IGNORECASE)
        # Check known patterns in the raw name
        for pat, label in _KNOWN_MRI.items():
            if pat in raw.lower():
                return label
        # Try known clinical patterns too
        for pat, label in known_clinical.items():
            if pat in raw.lower():
                return label
        # Handle LEFT_/RIGHT_ prefixed anatomy
        rl = raw.lower()
        side = ''
        inner = rl
        if rl.startswith('right_') or rl.startswith('r_'):
            side = 'R '
            inner = re.sub(r'^(right|r)_', '', rl)
        elif rl.startswith('left_') or rl.startswith('l_'):
            side = 'L '
            inner = re.sub(r'^(left|l)_', '', rl)
        # Check known MRI with side stripped
        for pat, label in _KNOWN_MRI.items():
            if pat in inner:
                return side + label if side else label
        clean = inner.replace('_', ' ').title()
        if side:
            clean = side + clean
        # Append Vol if looks like a brain region
        region_hints = ['hippocampus','hippo','amygdala','entorhinal','fusiform',
                        'temporal','parietal','frontal','cingulate','precuneus',
                        'ventricle','thalamus','caudate','putamen','pallidum',
                        'insula','cerebellum','brainstem','accumbens','cortex',
                        'subiculum','ca1','ca2','ca3','ca4','fimbria','hata']
        if any(h in inner for h in region_hints) and 'vol' not in inner:
            clean += ' Vol'
        if len(clean) > 40:
            clean = clean[:37] + '...'
        return clean

    # ── _mri columns not caught above: clean up ──
    m_mri = re.match(r'(.+?)_mri_?\d*$', nl)
    if m_mri:
        raw = m_mri.group(1)
        raw = re.sub(r'^phc_', '', raw, flags=re.IGNORECASE)
        for pat, label in _KNOWN_MRI.items():
            if pat in raw.lower():
                return label
        # Handle LEFT_/RIGHT_ prefixed anatomy
        rl = raw.lower()
        side = ''
        inner = rl
        if rl.startswith('right_') or rl.startswith('r_'):
            side = 'R '
            inner = re.sub(r'^(right|r)_', '', rl)
        elif rl.startswith('left_') or rl.startswith('l_'):
            side = 'L '
            inner = re.sub(r'^(left|l)_', '', rl)
        for pat, label in _KNOWN_MRI.items():
            if pat in inner:
                return side + label if side else label
        clean = inner.replace('_', ' ').title()
        if side:
            clean = side + clean
        region_hints = ['hippocampus','hippo','amygdala','entorhinal','fusiform',
                        'temporal','parietal','frontal','cingulate','precuneus',
                        'ventricle','thalamus','caudate','putamen','pallidum',
                        'insula','cerebellum','brainstem','accumbens','cortex',
                        'subiculum','ca1','ca2','ca3','ca4','fimbria','hata']
        if any(h in inner for h in region_hints) and 'vol' not in inner:
            clean += ' Vol'
        if len(clean) > 40:
            clean = clean[:37] + '...'
        return clean

    # ── Fallback: clean up ──
    clean = base.replace('_', ' ').title()
    if len(clean) > 45:
        clean = clean[:42] + '...'
    return clean


# ═══════════════════════════════════════════════════════════════
# DATA LOADING — copied from uwas_strata_multi.py (Figure 3)
# ═══════════════════════════════════════════════════════════════

def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower: return cols_lower[c.lower()]
    return None

def find_target_column(df, label_col='PHC_Diagnosis'):
    if label_col in df.columns:
        return label_col
    for alt in ['PHC_Diagnosis_clin', 'PHC_Diagnosis_clin1', 'PHC_Diagnosis',
                'DX_LABEL', 'DX_bl', 'DX', 'DIAGNOSIS', 'AD_Label']:
        if alt in df.columns:
            return alt
    raise ValueError("Target column '{}' not found".format(label_col))

def encode_target(df, target_col):
    """Encode AD_Label: 1=AD/Dementia, 0=CN. Drop MCI and unknown."""
    labels = pd.Series(np.nan, index=df.index)

    # Handle numeric codes: 1=CN, 2=MCI(drop), 3=AD
    num = pd.to_numeric(df[target_col], errors='coerce')
    if num.notna().any():
        labels[num == 1] = 0
        labels[num == 3] = 1
        # num==2 stays NaN → dropped

    # Handle string labels
    raw = df[target_col].astype(str).str.strip().str.upper()
    ad_pats = ['AD', 'DEMENTIA']
    cn_pats = ['CN', 'NL', 'NORMAL', 'SMC']
    # MCI patterns → stay NaN → dropped
    for i, v in raw.items():
        if pd.notna(labels[i]):
            continue
        if any(p in v for p in ad_pats):
            labels[i] = 1
        elif any(p in v for p in cn_pats):
            labels[i] = 0

    n_before = len(df)
    df = df.copy()
    df['AD_Label'] = labels
    df = df.dropna(subset=['AD_Label']).reset_index(drop=True)
    df['AD_Label'] = df['AD_Label'].astype(int)

    n_ad = int((df['AD_Label'] == 1).sum())
    n_cn = int((df['AD_Label'] == 0).sum())
    n_drop = n_before - len(df)
    print("  Encoded: CN={}, AD={} (dropped {} MCI/unknown)".format(n_cn, n_ad, n_drop))
    return df

def dedup_longitudinal(df):
    """One row per subject: AD → latest visit, CN → baseline. Matches Figure 3."""
    ptid_col = next((c for c in df.columns if c.upper() in ['PTID', 'RID', 'SUBJECT_ID']), None)
    if not ptid_col or not df[ptid_col].duplicated().any():
        return df

    viscode_col = next((c for c in df.columns if c.upper() in ['VISCODE', 'VISCODE2']), None)
    if viscode_col:
        viscode_order = {'sc': -1, 'bl': 0, 'scmri': 0}
        for m in [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 72, 84, 96, 108, 120]:
            viscode_order['m{:02d}'.format(m)] = m
            viscode_order['m{}'.format(m)] = m
        df = df.copy()
        df['_visit_order'] = df[viscode_col].astype(str).str.lower().map(viscode_order).fillna(999)
    else:
        df = df.copy()
        df['_visit_order'] = range(len(df))

    keep_idx = []
    for pid, grp in df.groupby(ptid_col):
        if len(grp) == 1:
            keep_idx.append(grp.index[0])
        else:
            if grp['AD_Label'].iloc[0] == 1:
                keep_idx.append(grp['_visit_order'].idxmax())
            else:
                keep_idx.append(grp['_visit_order'].idxmin())

    df = df.loc[keep_idx].drop(columns=['_visit_order']).reset_index(drop=True)
    y = df['AD_Label'].values
    print("  Longitudinal dedup: {} subjects (CN={}, AD={})".format(
        len(df), int((y==0).sum()), int((y==1).sum())))
    return df

def add_site_info(df, consents_path=None):
    """Add SITEID — from existing col, consents file, or PTID extraction. Matches Figure 3."""
    # Check existing columns
    for col in df.columns:
        cl = col.lower()
        if cl in ('siteid', 'site_id', 'site', '_siteid_'):
            print("  Site column found: {} ({} non-null)".format(col, df[col].notna().sum()))
            return df, col
    # Check siteid* variants
    for col in df.columns:
        if 'siteid' in col.lower():
            print("  Site column found: {} ({} non-null)".format(col, df[col].notna().sum()))
            return df, col

    # From consents file
    if consents_path and os.path.exists(consents_path):
        print("  Loading site info from: {}".format(consents_path))
        try:
            consent = pd.read_csv(consents_path, low_memory=False)
            if 'RID' in consent.columns and 'SITEID' in consent.columns:
                site_map = consent[['RID', 'SITEID']].drop_duplicates(subset='RID')
                rid_col = next((c for c in df.columns if c.lower() == 'rid'), None)
                if not rid_col:
                    ptid_col = next((c for c in df.columns if c.lower().startswith('ptid')), None)
                    if ptid_col:
                        df['_RID_TEMP_'] = df[ptid_col].astype(str).str.extract(r'_S_(\d+)')[0].astype(float)
                        rid_col = '_RID_TEMP_'
                if rid_col:
                    df = df.merge(site_map, left_on=rid_col, right_on='RID', how='left')
                    df.rename(columns={'SITEID': '_SITEID_'}, inplace=True)
                    if '_RID_TEMP_' in df.columns:
                        df.drop(columns=['_RID_TEMP_'], inplace=True)
                    print("  Merged SITEID: {}/{} matched ({} sites)".format(
                        df['_SITEID_'].notna().sum(), len(df), df['_SITEID_'].nunique()))
                    return df, '_SITEID_'
        except Exception as e:
            print("  WARNING: Consents load failed: {}".format(e))

    # From PTID pattern
    ptid_col = next((c for c in df.columns if c.lower().startswith('ptid')), None)
    if ptid_col:
        df['_SITEID_'] = df[ptid_col].astype(str).str.extract(r'^(\d+)_S_')[0].astype(float)
        n = df['_SITEID_'].notna().sum()
        if n > 0:
            print("  Extracted SITEID from PTID: {}/{} ({} sites)".format(
                n, len(df), int(df['_SITEID_'].nunique())))
            return df, '_SITEID_'

    print("  WARNING: No site info — will fall back to 5-fold CV")
    return df, None

def load_data(args):
    """Full data loading pipeline — matches Figure 2/3."""
    print("[1/6] Loading data from {}...".format(args.input))
    df = pd.read_csv(args.input, low_memory=False)
    print("  Loaded: {:,} rows x {:,} columns".format(len(df), len(df.columns)))

    # ─── Encode target ───
    target_col = find_target_column(df)
    dx_vals = df[target_col].value_counts()
    print("  Diagnosis distribution ({}):".format(target_col))
    for v, n in dx_vals.items():
        print("    {}: {}".format(v, n))

    df = encode_target(df, target_col)

    # ─── Longitudinal dedup ───
    df = dedup_longitudinal(df)
    y = df['AD_Label'].values

    # ─── Merge APOE from external file ───
    if args.apoe and os.path.exists(args.apoe):
        print("  Loading APOE from {}...".format(args.apoe))
        adf = pd.read_csv(args.apoe, low_memory=False)
        if 'GENOTYPE' in adf.columns:
            adf['APOE_e4_carrier'] = adf['GENOTYPE'].apply(lambda g: 1 if '4' in str(g) else 0)
            adf = adf[['RID', 'APOE_e4_carrier']].drop_duplicates(subset='RID')
            if 'RID' in df.columns:
                df = df.merge(adf, on='RID', how='left', suffixes=('', '_ext'))
                # Use _gene suffix so classify_features picks it up as clinical
                if 'APOE_e4_carrier' in df.columns:
                    df.rename(columns={'APOE_e4_carrier': 'APOE_e4_carrier_gene'}, inplace=True)
                df['APOE_e4_carrier_gene'] = df['APOE_e4_carrier_gene'].fillna(0).astype(int)
                print("  APOE e4 carrier merged: {}/{} carriers".format(
                    int(df['APOE_e4_carrier_gene'].sum()), len(df)))
            y = df['AD_Label'].values

    # ─── Merge PHS from external file ───
    if args.phs:
        if os.path.exists(args.phs):
            pdf = pd.read_csv(args.phs, low_memory=False)
            rc = find_col(pdf, ['RID','PTID','Subject'])
            dr = find_col(df, ['RID','PTID','Subject'])
            phs_col = None
            for c in pdf.columns:
                if c.lower() in ('ptid','rid','subject','phase','viscode'): continue
                if 'phs' in c.lower():
                    phs_col = c; break
            if phs_col is None:
                for c in pdf.columns:
                    if c.lower() in ('ptid','rid','subject','phase','viscode'): continue
                    if pd.api.types.is_numeric_dtype(pdf[c]):
                        phs_col = c; break
            if rc and dr and phs_col:
                print("  PHS file: key={}, score={}, {} rows".format(rc, phs_col, len(pdf)))
                pdf[rc] = pdf[rc].astype(str).str.strip()
                df[dr] = df[dr].astype(str).str.strip()
                pm = pdf.drop_duplicates(subset=[rc], keep='last').set_index(rc)[phs_col]
                df['PHS_external_gene'] = df[dr].map(pm).astype(float)
                n_mapped = df['PHS_external_gene'].notna().sum()
                print("  PHS: mapped {} / {} from {}".format(n_mapped, len(df), args.phs))
                if n_mapped == 0:
                    print("  WARNING: 0 PHS mapped! df keys: {}, PHS keys: {}".format(
                        list(df[dr].head(3)), list(pm.index[:3])))
            else:
                print("  WARNING: PHS file missing columns (key={}, score={})".format(rc, phs_col))
        else:
            print("  WARNING: PHS file not found: {}".format(args.phs))
    else:
        print("  NOTE: --phs not provided, PHS excluded from clinical pool")

    # ─── Site info for LOSO ───
    df, site_col = add_site_info(df, consents_path=args.consents)

    # ─── Median imputation (keeps all subjects) ───
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    pre_impute_coverage = df[numeric_cols].notna().mean()
    n_missing = int(df[numeric_cols].isna().sum().sum())
    if n_missing > 0:
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)
        print("  Imputation: filled {:,} missing values with column medians".format(n_missing))

    print("  Final: {} (CN={}, AD={})".format(
        len(df), int((df['AD_Label']==0).sum()), int((df['AD_Label']==1).sum())))
    return df, site_col, pre_impute_coverage


# ═══════════════════════════════════════════════════════════════
# BACKWARD ELIMINATION
# ═══════════════════════════════════════════════════════════════

def evaluate_features(X, y, features, site_series=None, min_site_n=10, max_sites=10):
    """Evaluate feature set via LOSO (Leave-One-Site-Out) AUC.
    Falls back to 5-fold stratified CV if site info unavailable."""
    Xf = X[features].copy()
    Xf = Xf.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    med = Xf.median()
    Xf = Xf.fillna(med).fillna(0)

    # Jitter zero-variance columns
    rng = np.random.RandomState(42)
    for col in Xf.columns:
        if Xf[col].std() < 1e-10:
            Xf[col] = Xf[col] + rng.normal(0, 1e-6, size=len(Xf))

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos < 5 or n_neg < 5:
        return 0.5

    # ── Try LOSO ──
    if site_series is not None:
        sites_in_sub = site_series.reindex(X.index).dropna()
        if len(sites_in_sub) > 0:
            try:
                sites_in_sub = sites_in_sub.astype(int)
            except (ValueError, TypeError):
                pass
            site_counts = sites_in_sub.value_counts()
            eligible = site_counts[site_counts >= min_site_n]

            if len(eligible) >= 3:
                loso_sites = eligible.head(max_sites)
                site_aucs = []
                for site_id in loso_sites.index:
                    test_mask = (sites_in_sub == site_id)
                    train_mask = (~test_mask) & sites_in_sub.notna()
                    test_idx = test_mask[test_mask].index.intersection(Xf.index)
                    train_idx = train_mask[train_mask].index.intersection(Xf.index)
                    if len(test_idx) < 3 or len(train_idx) < 10:
                        continue
                    y_tr, y_te = y[train_idx], y[test_idx]
                    if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
                        continue
                    scaler = StandardScaler()
                    Xtr = scaler.fit_transform(Xf.loc[train_idx])
                    Xte = scaler.transform(Xf.loc[test_idx])
                    try:
                        clf = LogisticRegression(C=1.0, penalty='l2', max_iter=2000, random_state=42)
                        clf.fit(Xtr, y_tr)
                        prob = clf.predict_proba(Xte)[:, 1]
                        site_aucs.append(roc_auc_score(y_te, prob))
                    except:
                        continue
                if len(site_aucs) >= 3:
                    return np.mean(site_aucs)

    # ── Fallback: 5-fold stratified CV ──
    scaler = StandardScaler()
    Xsc = scaler.fit_transform(Xf)
    n_s = min(5, n_pos, n_neg)
    if n_s < 2: return 0.5
    skf = StratifiedKFold(n_splits=n_s, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in skf.split(Xsc, y):
        clf = LogisticRegression(C=1.0, penalty='l2', max_iter=2000, random_state=42)
        clf.fit(Xsc[train_idx], y.values[train_idx])
        if len(np.unique(y.values[test_idx])) < 2: continue
        prob = clf.predict_proba(Xsc[test_idx])[:, 1]
        aucs.append(roc_auc_score(y.values[test_idx], prob))
    return np.mean(aucs) if aucs else 0.5


def backward_elimination(df_group, y_group, feature_pool, clinical_set=None,
                         site_series=None,
                         min_features=6, min_clinical=3, min_mri=3, verbose=True):
    """
    Backward elimination with minimum clinical/MRI protection.
    Protects at least min_clinical clinical and min_mri MRI features.
    """
    current = list(feature_pool)
    if clinical_set is None:
        clinical_set = set()
    history = []

    auc = evaluate_features(df_group, y_group, current, site_series=site_series)
    history.append({
        'n_features': len(current),
        'auc': auc,
        'removed': None,
        'features': list(current),
    })
    if verbose:
        n_clin = sum(1 for f in current if f in clinical_set)
        n_mri = len(current) - n_clin
        print("    Start: {} features ({} clin + {} MRI), AUC={:.3f}".format(
            len(current), n_clin, n_mri, auc))

    while len(current) > min_features:
        best_auc_after_removal = -1
        worst_feature = None

        n_clinical_now = sum(1 for f in current if f in clinical_set)
        n_mri_now = len(current) - n_clinical_now

        for feat in current:
            if feat in clinical_set and n_clinical_now <= min_clinical:
                continue
            if feat not in clinical_set and n_mri_now <= min_mri:
                continue
            remaining = [f for f in current if f != feat]
            auc_without = evaluate_features(df_group, y_group, remaining, site_series=site_series)
            if auc_without > best_auc_after_removal:
                best_auc_after_removal = auc_without
                worst_feature = feat

        if worst_feature is None:
            break

        current.remove(worst_feature)
        history.append({
            'n_features': len(current),
            'auc': best_auc_after_removal,
            'removed': worst_feature,
            'features': list(current),
        })

        if verbose and len(current) % 5 == 0:
            nc = sum(1 for f in current if f in clinical_set)
            nm = len(current) - nc
            print("    {} features ({} clin + {} MRI): AUC={:.3f} (dropped {})".format(
                len(current), nc, nm, best_auc_after_removal,
                annotate_feature(worst_feature)))

    if verbose:
        nc = sum(1 for f in current if f in clinical_set)
        nm = len(current) - nc
        print("    Final: {} features ({} clin + {} MRI), AUC={:.3f}".format(
            len(current), nc, nm, history[-1]['auc']))

    return history


def find_elbow(history, threshold=0.02):
    """Find elbow: smallest feature set within `threshold` AUC of the peak."""
    peak_auc = max(h['auc'] for h in history)
    cutoff = peak_auc - threshold

    # Walk from smallest to largest, find first that exceeds cutoff
    for h in sorted(history, key=lambda x: x['n_features']):
        if h['auc'] >= cutoff:
            return h
    return history[0]


# ═══════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════

GROUP_COLORS = {
    'HTN- DM-': '#1b9e77',   # teal-green
    'HTN+ DM-': '#2166ac',   # deep blue
    'HTN- DM+': '#e66101',   # burnt orange
    'HTN+ DM+': '#7b2d8e',   # purple
}
GROUP_ORDER = ['HTN- DM-', 'HTN+ DM-', 'HTN- DM+', 'HTN+ DM+']
GROUP_DM_PLUS = {'HTN- DM+', 'HTN+ DM+'}


def generate_figure(elbow_features, elbow_info, elimination_histories,
                    group_ns, clinical_set, output_dir):
    """
    2x2 layout, 15pt Helvetica:
      Top-left  (A): Elimination curves
      Top-right (B): Unified weight heatmap
      Bottom    (C): Single radar with all 4 groups overlaid, spanning full width
    """
    from matplotlib.lines import Line2D

    FS = 12
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'TeX Gyre Heros', 'DejaVu Sans'],
        'font.size': FS,
        'axes.labelsize': FS,
        'axes.titlesize': FS,
        'xtick.labelsize': FS,
        'ytick.labelsize': FS,
        'legend.fontsize': FS - 1,
        'axes.linewidth': 0.6,
    })

    CLIN_COLOR = '#1a5276'
    MRI_COLOR = '#7b2d8e'

    # ── Build union of all elbow features ──
    all_feat_map = {}
    for gname in GROUP_ORDER:
        if gname not in elbow_info:
            continue
        for f in elbow_info[gname]['features']:
            if f not in all_feat_map:
                all_feat_map[f] = 'clin' if f in clinical_set else 'mri'
    all_feats = sorted(all_feat_map.keys(),
                       key=lambda f: (0 if all_feat_map[f] == 'clin' else 1,
                                      annotate_feature(f)))
    all_labels = [annotate_feature(f) for f in all_feats]
    all_types = [all_feat_map[f] for f in all_feats]
    n_feats = len(all_feats)

    # ── Collect weights and betas ──
    weight_matrix = {}
    beta_matrix = {}
    max_weight = 0.01
    for gname in GROUP_ORDER:
        if gname not in elbow_info:
            continue
        w = elbow_info[gname].get('weights', {})
        b = elbow_info[gname].get('betas', {})
        weight_matrix[gname] = w
        beta_matrix[gname] = b
        if w:
            max_weight = max(max_weight, max(w.values()))
    max_weight = max(max_weight, 0.2)

    active_groups = [g for g in GROUP_ORDER if g in elbow_info]
    n_groups = len(active_groups)

    # ══════════════════════════════════════════════
    # Figure: A/B stacked left, C square right
    # ══════════════════════════════════════════════
    fig = plt.figure(figsize=(22, 12), facecolor='white')
    gs_outer = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2],
                                 width_ratios=[0.8, 1, 1],
                                 hspace=0.35, wspace=0.20,
                                 top=0.95, bottom=0.08,
                                 left=0.10, right=0.96)

    # ══════════════════════════════════════
    # Panel A — Elimination Curves (top-left)
    # ══════════════════════════════════════
    ax_a = fig.add_subplot(gs_outer[0, 0])

    for gname in GROUP_ORDER:
        if gname not in elimination_histories:
            continue
        hist = elimination_histories[gname]
        ns = [h['n_features'] for h in hist]
        aucs = [h['auc'] for h in hist]
        is_dm = gname in GROUP_DM_PLUS
        ax_a.plot(ns, aucs,
                  marker='D' if is_dm else 'o',
                  linestyle='--' if is_dm else '-',
                  color=GROUP_COLORS[gname],
                  markersize=5, linewidth=1.8,
                  markeredgecolor='white', markeredgewidth=0.5,
                  alpha=0.85, zorder=2)
        if gname in elbow_info:
            ei = elbow_info[gname]
            ax_a.plot(ei['n_features'], ei['auc'], '*',
                      color='#e6a817', markersize=20, zorder=5,
                      markeredgecolor='#b8860b', markeredgewidth=0.8)

    ax_a.set_xlabel('Number of Features', fontsize=FS)
    ax_a.set_ylabel('AUC (LOSO CV)', fontsize=FS)
    # Panel labels using fig coordinates for alignment
    fig.text(0.02, 0.97, 'A', fontsize=20, fontweight='bold', va='top', ha='left')
    ax_a.set_xlim(left=0)
    ax_a.set_ylim(0.45, 1.02)
    ax_a.grid(True, alpha=0.12, linewidth=0.4, color='#ccc')
    ax_a.invert_xaxis()
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.spines['left'].set_linewidth(0.5)
    ax_a.spines['bottom'].set_linewidth(0.5)
    ax_a.spines['left'].set_color('#888')
    ax_a.spines['bottom'].set_color('#888')
    ax_a.tick_params(axis='both', colors='#555', width=0.5)

    handles = []
    for gname in GROUP_ORDER:
        is_dm = gname in GROUP_DM_PLUS
        h = Line2D([0], [0], color=GROUP_COLORS[gname],
                    linestyle='--' if is_dm else '-',
                    marker='D' if is_dm else 'o',
                    markersize=5, linewidth=1.8, label=gname,
                    markeredgecolor='white', markeredgewidth=0.5)
        handles.append(h)
    ax_a.legend(handles=handles, loc='lower left', frameon=False,
                fontsize=FS - 1)

    # ══════════════════════════════════════
    # Panel B — Dot Plot (top-right)
    # ══════════════════════════════════════
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    ax_b = fig.add_subplot(gs_outer[1, 0])
    fig.text(0.02, 0.52, 'B', fontsize=20, fontweight='bold', va='top', ha='left')

    # Diverging colormap: vivid blue → white → vivid red
    cmap_div = LinearSegmentedColormap.from_list('beta_dir', [
        (0.00, '#08306b'),   # very deep blue
        (0.10, '#2171b5'),   # strong blue
        (0.25, '#4292c6'),   # medium blue
        (0.40, '#9ecae1'),   # light blue
        (0.50, '#f7f7f7'),   # white
        (0.60, '#fcbba1'),   # light red
        (0.75, '#ef3b2c'),   # medium red
        (0.90, '#cb181d'),   # strong red
        (1.00, '#67000d'),   # very deep red
    ])
    # Find max |beta| for color normalization
    max_abs_beta = 0.01
    for gname in active_groups:
        b = beta_matrix.get(gname, {})
        if b:
            max_abs_beta = max(max_abs_beta, max(abs(v) for v in b.values()))
    norm_beta = Normalize(vmin=-max_abs_beta, vmax=max_abs_beta)

    max_dot_s = 800
    min_dot_s = 25

    def w2s_b(w):
        if w is None or np.isnan(w) or w < 1e-4:
            return min_dot_s
        return min_dot_s + (max_dot_s - min_dot_s) * min(w / max_weight, 1.0)

    # White cell backgrounds with light grid
    for i in range(n_feats):
        for j in range(n_groups):
            ax_b.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94,
                           facecolor='white', edgecolor='#e0e0e0',
                           linewidth=0.5, zorder=1))

    # Column separator lines
    for j in range(1, n_groups):
        ax_b.axvline(x=j - 0.5, color='#bbb', linewidth=0.8, zorder=2)

    # Draw dots
    for i in range(n_feats):
        f = all_feats[i]
        for j in range(n_groups):
            gname = active_groups[j]
            w = weight_matrix.get(gname, {}).get(f, None)
            beta = beta_matrix.get(gname, {}).get(f, None)

            if w is None or (isinstance(w, float) and np.isnan(w)):
                # Feature not in this group — small grey dot
                ax_b.scatter(j, i, s=min_dot_s * 0.5, c='#e5e7eb',
                             edgecolors='#ccc', linewidths=0.3, zorder=3)
            else:
                s = w2s_b(w)
                color_val = beta if beta is not None else 0
                c = cmap_div(norm_beta(color_val))
                ax_b.scatter(j, i, s=s, c=[c], edgecolors='black',
                             linewidths=0.5, zorder=5)

    # Section divider between clinical and MRI features
    for i in range(n_feats):
        if i > 0 and all_types[i] != all_types[i - 1]:
            ax_b.axhline(y=i - 0.5, color='#999', linewidth=1.0, zorder=4)

    ax_b.set_xlim(-0.5, n_groups - 0.5)
    ax_b.set_ylim(n_feats - 0.5, -0.5)
    ax_b.set_xticks(range(n_groups))
    group_labels = []
    for g in active_groups:
        ni = group_ns.get(g, {})
        group_labels.append('{}\nn={}'.format(g, ni.get('total', '?')))
    ax_b.set_xticklabels(group_labels, fontsize=FS)
    for j, g in enumerate(active_groups):
        ax_b.get_xticklabels()[j].set_color(GROUP_COLORS[g])
        ax_b.get_xticklabels()[j].set_fontweight('bold')

    ax_b.set_yticks(range(n_feats))
    ax_b.set_yticklabels(all_labels, fontsize=FS)
    for i, t in enumerate(all_types):
        color = CLIN_COLOR if t == 'clin' else MRI_COLOR
        ax_b.get_yticklabels()[i].set_color(color)
        ax_b.get_yticklabels()[i].set_fontweight('bold' if t == 'clin' else 'normal')

    ax_b.tick_params(length=0)
    for spine in ax_b.spines.values():
        spine.set_visible(False)

    # Colorbar for β direction (below Panel B)
    cbar_ax = fig.add_axes([0.03, 0.025, 0.14, 0.008])
    sm = plt.cm.ScalarMappable(cmap=cmap_div, norm=norm_beta)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([-max_abs_beta, 0, max_abs_beta])
    cbar.set_ticklabels(['Protective', '0', 'Risk'])
    cbar.ax.tick_params(labelsize=FS - 1, length=2, colors='black')
    cbar.set_label('\u03b2 direction', fontsize=FS - 1, color='black')
    cbar.outline.set_linewidth(0.4)

    # Size legend (next to colorbar) — use proper scatter legend
    size_handles = []
    for lbl, wval in [('0.1', 0.1 * max_weight), ('0.5', 0.5 * max_weight), ('max', max_weight)]:
        s = w2s_b(wval)
        size_handles.append(ax_b.scatter([], [], s=s, c='#d4d4d4', edgecolors='black',
                                          linewidths=0.5, label=lbl))
    size_leg = ax_b.legend(handles=size_handles, title='Weight', title_fontsize=FS - 1,
                           fontsize=FS - 1, loc='upper right',
                           bbox_to_anchor=(1.0, -0.08),
                           frameon=True, fancybox=True, framealpha=0.95,
                           edgecolor='#ddd', handletextpad=0.5, borderpad=0.8,
                           labelspacing=1.5, ncol=3)

    # ══════════════════════════════════════
    # Panel C — Radar with signed β (right column, full height)
    # ══════════════════════════════════════
    ax_c = fig.add_subplot(gs_outer[:, 1:], projection='polar')
    # Use fig.text for C label — polar transAxes doesn't align with cartesian panels
    # Place at same y=0.97 as A, x at start of column 1
    fig.text(0.35, 0.97, 'C', fontsize=20, fontweight='bold', va='top', ha='left')

    angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    # Use signed β scale: find max |β| across all groups/features
    max_abs = 0.01
    for gname in active_groups:
        b = beta_matrix.get(gname, {})
        for f in all_feats:
            if f in b:
                max_abs = max(max_abs, abs(b[f]))
    max_abs = np.ceil(max_abs * 10) / 10  # round up to nearest 0.1

    # Offset to make 0 the center: shift all values by max_abs
    # Radar plots from 0 outward, so map [-max_abs, +max_abs] → [0, 2*max_abs]
    radar_max = 2 * max_abs

    # Concentric rings at -max_abs, -0.5*max_abs, 0, +0.5*max_abs, +max_abs
    ring_positions = [0, 0.5 * max_abs, max_abs, 1.5 * max_abs, radar_max]
    ring_labels = ['{:+.1f}'.format(-max_abs), '{:+.1f}'.format(-0.5 * max_abs),
                   '0', '{:+.1f}'.format(0.5 * max_abs), '{:+.1f}'.format(max_abs)]

    ax_c.set_ylim(0, radar_max * 1.08)
    ax_c.set_yticks(ring_positions)
    ax_c.set_yticklabels(ring_labels, fontsize=FS - 1, color='#555')

    # Feature labels
    ax_c.set_xticks(angles)
    ax_c.set_xticklabels(all_labels, fontsize=FS)
    for i, label in enumerate(ax_c.get_xticklabels()):
        c = CLIN_COLOR if all_types[i] == 'clin' else MRI_COLOR
        label.set_color(c)
        label.set_fontweight('bold' if all_types[i] == 'clin' else 'normal')

    ax_c.grid(True, alpha=0.6, linewidth=1.0, color='black')
    ax_c.spines['polar'].set_visible(True)
    ax_c.spines['polar'].set_linewidth(1.0)
    ax_c.spines['polar'].set_color('black')

    # Plot each group — thicker lines, more distinct
    for gname in GROUP_ORDER:
        if gname not in elbow_info:
            continue
        b = beta_matrix.get(gname, {})
        vals = [b.get(f, 0) + max_abs for f in all_feats]
        vals_closed = vals + [vals[0]]
        is_dm = gname in GROUP_DM_PLUS

        ax_c.fill(angles_closed, vals_closed,
                  color=GROUP_COLORS[gname], alpha=0.08)
        ax_c.plot(angles_closed, vals_closed,
                  color=GROUP_COLORS[gname], linewidth=2.5,
                  linestyle='--' if is_dm else '-',
                  marker='D' if is_dm else 'o',
                  markersize=8, markerfacecolor=GROUP_COLORS[gname],
                  markeredgecolor='white', markeredgewidth=1.2, zorder=3)

    # Zero ring highlight (β=0)
    ax_c.plot(np.linspace(0, 2 * np.pi, 100), [max_abs] * 100,
              color='black', linewidth=1.5, linestyle='-', alpha=0.7, zorder=1)

    # Legend
    legend_handles = []
    for gname in GROUP_ORDER:
        if gname not in elbow_info:
            continue
        is_dm = gname in GROUP_DM_PLUS
        ni = group_ns.get(gname, {})
        ei = elbow_info[gname]
        lbl = '{} (n={}, AUC={:.2f})'.format(gname, ni.get('total', '?'), ei['auc'])
        h = Line2D([0], [0], color=GROUP_COLORS[gname],
                    linestyle='--' if is_dm else '-',
                    marker='D' if is_dm else 'o',
                    markersize=6, linewidth=1.8, label=lbl,
                    markeredgecolor='white', markeredgewidth=0.5)
        legend_handles.append(h)
    legend_handles.append(Line2D([0], [0], marker='s', color='w',
                                  markerfacecolor=CLIN_COLOR, markersize=10,
                                  label='Clinical', linestyle='None'))
    legend_handles.append(Line2D([0], [0], marker='s', color='w',
                                  markerfacecolor=MRI_COLOR, markersize=10,
                                  label='MRI', linestyle='None'))
    ax_c.legend(handles=legend_handles, loc='upper right',
                bbox_to_anchor=(1.25, 0.95), frameon=False, fontsize=FS - 1)

    # ── Save ──
    fig.savefig(os.path.join(output_dir, 'figure4.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'figure4.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure4.pdf/png")


def main():
    pa = argparse.ArgumentParser(description='Figure 4: Minimal feature sets by HTN x DM')
    pa.add_argument('--input', '-i', required=True)
    pa.add_argument('--consents', '-c', default=None)
    pa.add_argument('--apoe', default=None)
    pa.add_argument('--phs', default=None, help='PHS scores CSV (needs RID/PTID + PHS column)')
    pa.add_argument('--output', '-o', default='.')
    pa.add_argument('--max-pool', type=int, default=30, help='Max features to start elimination from')
    pa.add_argument('--min-features', type=int, default=6, help='Minimum features to keep total')
    pa.add_argument('--min-clinical', type=int, default=2, help='Minimum clinical features to keep (2-4)')
    pa.add_argument('--min-mri', type=int, default=2, help='Minimum MRI features to keep (2-4)')
    pa.add_argument('--auc-threshold', type=float, default=0.02, help='AUC drop threshold for elbow')
    args = pa.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df, site_col, pre_impute_coverage = load_data(args)
    site_series = df[site_col] if site_col else None

    # [2] Classify features
    print("\n[2/6] Classifying features...")
    clinical_pool, mri_pool, col_info = classify_features(df)
    print("  Clinical: {}  MRI: {}".format(len(clinical_pool), len(mri_pool)))

    if 'htn' not in col_info or 'dm' not in col_info:
        print("  ERROR: No HTN/DM columns!"); sys.exit(1)
    print("  HTN col: {}  DM col: {}".format(col_info['htn'], col_info['dm']))

    # Combined pool: Fig 2/3 clinical features + all MRI features
    # ── Define clinical features explicitly from Figure 2/3 ──
    FIG23_CLINICAL_PATTERNS = [
        # Demographics & risk factors
        ('age_cardiovascularrisk', 'Age'),
        ('age_cognition', 'Age'),
        ('sex', 'Sex'),
        ('education', 'Education'),
        ('bmi', 'BMI'),
        # Genetic
        ('phs_external', 'PHS'),       # from --phs file only
        ('apoe_e4_carrier', 'APOE Carrier'),
    ]

    # Match patterns against actual columns (case-insensitive, first match per label wins)
    fig23_clinical = []
    fig23_clinical_names = {}
    used_labels = set()
    for col in df.columns:
        cl = col.lower()
        # Skip strata columns and leakage
        if col == col_info.get('htn') or col == col_info.get('dm'):
            continue
        if is_leakage(col) or is_cognitive(col):
            continue
        for pat, label in FIG23_CLINICAL_PATTERNS:
            if label in used_labels:
                continue
            if pat in cl and pd.api.types.is_numeric_dtype(df[col]):
                fig23_clinical.append(col)
                fig23_clinical_names[col] = label
                used_labels.add(label)
                break

    clinical_pool_final = fig23_clinical
    clinical_set = set(clinical_pool_final)
    print("  Figure 2/3 clinical features matched: {}".format(len(clinical_pool_final)))
    for c in clinical_pool_final:
        print("    {} -> {}".format(c, fig23_clinical_names.get(c, annotate_feature(c))))

    # All MRI features (numeric, >30% pre-imputation coverage)
    valid_mri = []
    n_othe_excluded = 0
    for c in mri_pool:
        if has_suffix(c, '_othe'):
            n_othe_excluded += 1
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cov = pre_impute_coverage.get(c, 0)
            if cov > 0.3:
                valid_mri.append(c)
    n_before_dedup = len(valid_mri)

    # Deduplicate: if two columns produce the same annotation, keep the one
    # with higher pre-imputation coverage (avoids duplicate labels in figures)
    ann_best = {}  # annotation -> (col, coverage)
    for c in valid_mri:
        ann = annotate_feature(c)
        cov = pre_impute_coverage.get(c, 0)
        if ann not in ann_best or cov > ann_best[ann][1]:
            ann_best[ann] = (c, cov)
    valid_mri = [col for col, _ in ann_best.values()]
    n_dedup = n_before_dedup - len(valid_mri)

    print("  MRI features: {} (>30% coverage, excluded {} _othe, deduplicated {})".format(
        len(valid_mri), n_othe_excluded, n_dedup))

    full_pool = clinical_pool_final + valid_mri
    print("  Total pool: {} ({} clinical + {} MRI)".format(
        len(full_pool), len(clinical_pool_final), len(valid_mri)))

    # [3] Define groups
    print("\n[3/6] Defining HTN x DM subgroups...")
    htn = df[col_info['htn']].fillna(0).astype(int)
    dm = df[col_info['dm']].fillna(0).astype(int)
    groups = {
        'HTN- DM-': (htn == 0) & (dm == 0),
        'HTN+ DM-': (htn == 1) & (dm == 0),
        'HTN- DM+': (htn == 0) & (dm == 1),
        'HTN+ DM+': (htn == 1) & (dm == 1),
    }

    group_ns = {}
    for gname, gmask in groups.items():
        n = int(gmask.sum())
        ncn = int((df.loc[gmask, 'AD_Label'] == 0).sum())
        nad = int((df.loc[gmask, 'AD_Label'] == 1).sum())
        group_ns[gname] = {'total': n, 'cn': ncn, 'ad': nad}
        print("  {}: n={} (CN={}, AD={})".format(gname, n, ncn, nad))

    # [4] Pre-select top features per group (to make elimination tractable)
    # All clinical features from Fig 2/3 are always included; fill rest with top MRI
    print("\n[4/6] Pre-selecting top features per group...")
    print("  Pool breakdown: {} clinical (Fig 2/3), {} MRI".format(
        len(clinical_pool_final), len(valid_mri)))

    group_top_features = {}
    for gname, gmask in groups.items():
        df_g = df.loc[gmask].copy()
        y_g = df_g['AD_Label']
        if y_g.sum() < 10 or (len(y_g) - y_g.sum()) < 10:
            print("  {} - too few samples, skipping".format(gname))
            continue

        # Rank features by individual AUC
        def rank_feats(pool):
            ranked = []
            for col in pool:
                try:
                    x = df_g[col].values.astype(float)
                except (ValueError, TypeError):
                    continue
                yv = y_g.values.astype(float)
                valid = np.isfinite(x) & np.isfinite(yv)
                if valid.sum() < 20: continue
                try:
                    auc = roc_auc_score(yv[valid], x[valid])
                    auc = max(auc, 1 - auc)
                    ranked.append((col, auc))
                except:
                    continue
            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked

        clin_ranked = rank_feats(clinical_pool_final)
        mri_ranked = rank_feats(valid_mri)

        # Always include ALL matched Fig 2/3 clinical features
        top_clin = [c for c, _ in clin_ranked]
        # Fill remaining slots with top MRI
        n_mri_slots = max(0, args.max_pool - len(top_clin))
        top_mri = [c for c, _ in mri_ranked[:n_mri_slots]]

        combined = top_clin + top_mri
        group_top_features[gname] = combined
        print("  {}: {} features ({} clinical + {} MRI, best clin AUC={:.3f}, best MRI AUC={:.3f})".format(
            gname, len(combined), len(top_clin), len(top_mri),
            clin_ranked[0][1] if clin_ranked else 0,
            mri_ranked[0][1] if mri_ranked else 0))

    # [5] Backward elimination per group (protect at least 1 clinical feature)
    print("\n[5/6] Backward elimination per group...")
    elimination_histories = {}
    elbow_features = {}
    elbow_info = {}

    for gname in GROUP_ORDER:
        if gname not in group_top_features: continue
        print("\n  --- {} ---".format(gname))
        gmask = groups[gname]
        df_g = df.loc[gmask].copy()
        y_g = df_g['AD_Label']
        feat_pool = group_top_features[gname]

        history = backward_elimination(
            df_g, y_g, feat_pool,
            clinical_set=clinical_set,
            site_series=site_series,
            min_features=max(args.min_features, args.min_clinical + args.min_mri),
            min_clinical=args.min_clinical, min_mri=args.min_mri, verbose=True)
        elimination_histories[gname] = history

        elbow = find_elbow(history, threshold=args.auc_threshold)
        elbow_features[gname] = elbow['features']

        # Train final model on full group to get feature weights
        feats = elbow['features']
        Xf = df_g[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        Xf = Xf.fillna(Xf.median()).fillna(0)
        for c in Xf.columns:
            if Xf[c].std() < 1e-10:
                Xf[c] = Xf[c] + np.random.RandomState(42).normal(0, 1e-6, size=len(Xf))
        scaler = StandardScaler()
        Xsc = scaler.fit_transform(Xf)
        clf = LogisticRegression(C=1.0, penalty='l2', max_iter=2000, random_state=42)
        clf.fit(Xsc, y_g)
        weights = dict(zip(feats, np.abs(clf.coef_[0])))
        betas = dict(zip(feats, clf.coef_[0]))

        # Diagnostic: print direction of key clinical features
        for f in feats:
            ann = annotate_feature(f)
            if 'Sex' in ann or 'Education' in ann or 'Age' in ann or 'PHS' in ann:
                b = betas[f]
                w = weights[f]
                # Check raw correlation with AD label
                xraw = df_g[f].apply(pd.to_numeric, errors='coerce')
                corr = xraw.corr(y_g)
                print("    {} [{}]: beta={:+.3f}, |w|={:.3f}, corr(raw,AD)={:+.3f}".format(
                    ann, f, b, w, corr if not np.isnan(corr) else 0))

        elbow_info[gname] = {
            'n_features': elbow['n_features'],
            'auc': elbow['auc'],
            'features': elbow['features'],
            'df_group': df_g,
            'weights': weights,
            'betas': betas,
        }

        print("  Elbow: {} features, AUC={:.3f}".format(
            elbow['n_features'], elbow['auc']))
        for f in elbow['features']:
            ftype = 'CLIN' if f in clinical_set else 'MRI'
            ann = annotate_feature(f)
            print("    [{}] {} <- {}".format(ftype, ann, f))

    # [6] Generate figures
    print("\n[6/6] Generating figures...")
    generate_figure(elbow_features, elbow_info, elimination_histories,
                    group_ns, clinical_set, args.output)

    # Save report
    report = {
        'groups': {},
    }
    for gname in GROUP_ORDER:
        if gname not in elbow_info: continue
        ei = elbow_info[gname]
        report['groups'][gname] = {
            'n_subjects': group_ns[gname],
            'n_minimal_features': ei['n_features'],
            'auc': round(ei['auc'], 4),
            'features': [(f, annotate_feature(f)) for f in ei['features']],
        }

    with open(os.path.join(args.output, 'figure4_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print("  Saved: figure4_report.json")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()

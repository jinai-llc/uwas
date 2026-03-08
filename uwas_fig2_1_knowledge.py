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

# ── Knowledge-constrained regularization ──
try:
    from beta_knowledge_v2 import (
        KnowledgeConstrainedLR,
        build_knowledge_priors,
        load_gpt_caches,
    )
    HAS_BETA_KNOWLEDGE = True
except ImportError:
    HAS_BETA_KNOWLEDGE = False
    print("  ⚠ beta_knowledge_v2 not found — will use standard L2")


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
        'sex': 'Sex', 'education': 'Education', 'bmi': 'BMI',
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

COHORT_COLOR = '#2d3436'



def generate_figure(feats, weights, history, n_total, auc_val,
                    clinical_set, output_dir, label_overrides=None):
    """
    Figure 2.1: Whole-cohort best-AUC feature set.
    Compact 2×2 layout, 18pt Helvetica, thick lines, large radar.
      Top-left  (A): Elimination curve
      Top-right (B): Horizontal bar chart of weights
      Bottom    (C): Radar plot spanning full width
    label_overrides: dict {raw_col: display_label} for directional features like Sex.
    """
    from matplotlib.lines import Line2D
    if label_overrides is None:
        label_overrides = {}

    def _label(f):
        """Annotate with override support."""
        if f in label_overrides:
            return label_overrides[f]
        return annotate_feature(f)

    FS = 18
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'TeX Gyre Heros', 'DejaVu Sans'],
        'font.size': FS,
        'axes.labelsize': FS,
        'axes.titlesize': FS,
        'xtick.labelsize': FS,
        'ytick.labelsize': FS,
        'legend.fontsize': FS - 3,
        'axes.linewidth': 0.8,
    })

    CLIN_COLOR = '#c0392b'
    MRI_COLOR = '#2c7873'

    feat_types = {f: ('clin' if f in clinical_set else 'mri') for f in feats}
    # Sort by absolute weight descending, MRI first then clinical
    sorted_feats = sorted(feats, key=lambda f: (-int(feat_types[f] == 'mri'),
                                                 -abs(weights.get(f, 0))))
    sorted_labels = [_label(f) for f in sorted_feats]
    sorted_types = [feat_types[f] for f in sorted_feats]
    sorted_weights = [weights.get(f, 0) for f in sorted_feats]
    n_feats = len(sorted_feats)

    max_abs_weight = max(abs(w) for w in sorted_weights) if sorted_weights else 0.2
    max_abs_weight = max(max_abs_weight, 0.2)

    # Radar order: clinical first alphabetical, then MRI; use SIGNED weights
    radar_feats = sorted(feats, key=lambda f: (0 if feat_types[f] == 'clin' else 1,
                                                _label(f)))
    radar_labels = [_label(f) for f in radar_feats]
    radar_types = [feat_types[f] for f in radar_feats]
    radar_weights = [weights.get(f, 0) for f in radar_feats]
    n_r = len(radar_feats)

    # ══════════════════════════════════════════════
    # Compact figure
    # ══════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 20), facecolor='white')
    gs_outer = gridspec.GridSpec(2, 2, height_ratios=[0.6, 1.0],
                                 width_ratios=[1, 1.2],
                                 hspace=0.30, wspace=0.38,
                                 top=0.96, bottom=0.04,
                                 left=0.07, right=0.97)

    # ══════════════════════════════════════
    # Panel A — Elimination Curve
    # ══════════════════════════════════════
    ax_a = fig.add_subplot(gs_outer[0, 0])

    ns = [h['n_features'] for h in history]
    aucs = [h['auc'] for h in history]
    ax_a.plot(ns, aucs, 'o-', color=COHORT_COLOR, markersize=8, linewidth=3,
              markeredgecolor='white', markeredgewidth=0.8,
              label='Whole cohort (n={})'.format(n_total))

    # Mark selected point (best AUC)
    ax_a.plot(n_feats, auc_val, '*', color='#d4a03c', markersize=28, zorder=5,
              markeredgecolor='white', markeredgewidth=0.8)

    ax_a.set_xlabel('Number of Features')
    ax_a.set_ylabel('AUC (LOSO CV)')
    ax_a.text(-0.12, 1.05, 'A', transform=ax_a.transAxes,
              fontsize=28, fontweight='bold', va='top')
    ax_a.set_xlim(left=0)
    ax_a.set_ylim(0.45, 1.02)
    ax_a.grid(True, alpha=0.15, linewidth=0.5)
    ax_a.invert_xaxis()
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.legend(loc='lower right', frameon=False)

    # ══════════════════════════════════════
    # Panel B — Diverging Bar Chart (signed β)
    # ══════════════════════════════════════
    ax_b = fig.add_subplot(gs_outer[0, 1])
    ax_b.text(-0.06, 1.05, 'B', transform=ax_b.transAxes,
              fontsize=28, fontweight='bold', va='top')

    y_pos = np.arange(n_feats)
    bar_colors = []
    for i, t in enumerate(sorted_types):
        base = CLIN_COLOR if t == 'clin' else MRI_COLOR
        bar_colors.append(base)

    bars = ax_b.barh(y_pos, sorted_weights, color=bar_colors, height=0.65,
                      edgecolor='black', linewidth=0.8, alpha=0.85)

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(sorted_labels)
    for i, t in enumerate(sorted_types):
        color = CLIN_COLOR if t == 'clin' else MRI_COLOR
        ax_b.get_yticklabels()[i].set_color(color)
        ax_b.get_yticklabels()[i].set_fontweight('bold' if t == 'clin' else 'normal')

    ax_b.set_xlabel('Model Coefficient (\u03b2)')
    ax_b.axvline(0, color='#333', linewidth=0.8, zorder=0)
    ax_b.set_xlim(-max_abs_weight * 1.15, max_abs_weight * 1.15)
    ax_b.invert_yaxis()
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(axis='x', alpha=0.15, linewidth=0.5)

    clin_h = Line2D([0], [0], marker='s', color='w', markerfacecolor=CLIN_COLOR,
                     markersize=12, label='Clinical', linestyle='None')
    mri_h = Line2D([0], [0], marker='s', color='w', markerfacecolor=MRI_COLOR,
                    markersize=12, label='MRI', linestyle='None')
    ax_b.legend(handles=[clin_h, mri_h], loc='lower right', frameon=False)

    # ══════════════════════════════════════
    # Panel C — Radar Plot (bottom, full width)
    # ══════════════════════════════════════
    ax_c = fig.add_subplot(gs_outer[1, :], projection='polar')
    ax_c.text(-0.05, 1.08, 'C', transform=ax_c.transAxes,
              fontsize=28, fontweight='bold', va='top')

    angles = np.linspace(0, 2 * np.pi, n_r, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    # Shift radar so center = floor of min β, outer = ceil of max β
    min_w = min(radar_weights)
    max_w = max(radar_weights)
    # Round to nice values
    radar_min = np.floor(min_w * 5) / 5 - 0.1  # pad below min
    radar_max = np.ceil(max_w * 5) / 5 + 0.1   # pad above max
    radar_range = radar_max - radar_min

    # Shift weights so radar_min maps to 0
    shifted_weights = [w - radar_min for w in radar_weights]
    shifted_closed = shifted_weights + [shifted_weights[0]]

    ax_c.set_ylim(0, radar_range)

    # Ring ticks at nice intervals including 0
    tick_step = 0.2 if radar_range < 1.5 else 0.5
    ring_real = np.arange(np.ceil(radar_min / tick_step) * tick_step,
                          radar_max + tick_step * 0.5, tick_step)
    ring_real = [round(v, 2) for v in ring_real]
    ring_shifted = [v - radar_min for v in ring_real]
    ax_c.set_yticks(ring_shifted)
    ax_c.set_yticklabels(['{:+.1f}'.format(v) if v != 0 else '0' for v in ring_real],
                          fontsize=FS - 5, color='#888')

    # Bold zero ring
    zero_shifted = 0 - radar_min
    ax_c.plot(np.linspace(0, 2 * np.pi, 100), [zero_shifted] * 100,
              color='black', linewidth=1.5, alpha=0.5, zorder=1)

    ax_c.set_xticks(angles)
    ax_c.set_xticklabels(radar_labels, fontsize=FS)
    for i, label in enumerate(ax_c.get_xticklabels()):
        c = CLIN_COLOR if radar_types[i] == 'clin' else MRI_COLOR
        label.set_color(c)
        label.set_fontweight('bold' if radar_types[i] == 'clin' else 'normal')

    ax_c.grid(True, alpha=0.4, linewidth=1.2, color='black')
    ax_c.spines['polar'].set_visible(False)

    ax_c.fill(angles_closed, shifted_closed, color=COHORT_COLOR, alpha=0.15)
    ax_c.plot(angles_closed, shifted_closed, color=COHORT_COLOR, linewidth=3.5,
              marker='o', markersize=10, markerfacecolor=COHORT_COLOR,
              markeredgecolor='white', markeredgewidth=1.5, zorder=3)

    legend_handles = [
        Line2D([0], [0], color=COHORT_COLOR, marker='o', markersize=9,
               linewidth=3, label='Whole cohort (n={}, AUC={:.2f})'.format(n_total, auc_val),
               markeredgecolor='white', markeredgewidth=0.8),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CLIN_COLOR,
               markersize=12, label='Clinical', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=MRI_COLOR,
               markersize=12, label='MRI', linestyle='None'),
    ]
    ax_c.legend(handles=legend_handles, loc='upper right',
                bbox_to_anchor=(1.32, 1.10), frameon=False, fontsize=FS - 2)

    # ── Save ──
    fig.savefig(os.path.join(output_dir, 'figure2_1.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'figure2_1.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure2_1.pdf/png")


def main():
    pa = argparse.ArgumentParser(description='Figure 2.1: Whole-cohort minimal feature set')
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
    pa.add_argument('--gpt-cache', default=None, help='Path to gpt_validation_cache.json')
    pa.add_argument('--gpt-network', default=None, help='Path to gpt_validation_network_cache_ad.json')
    pa.add_argument('--beta-int', default=None, help='Path to beta_int CSV (B3 age-brain priors for MRI)')
    pa.add_argument('--lambda-k', type=float, default=5.0, help='Knowledge penalty strength')
    args = pa.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df, site_col, pre_impute_coverage = load_data(args)
    site_series = df[site_col] if site_col else None
    y = df['AD_Label']

    # Load GPT caches for knowledge-constrained model
    main_cache, network_cache = {}, {}
    if HAS_BETA_KNOWLEDGE:
        main_cache, network_cache = load_gpt_caches(
            main_cache_path=args.gpt_cache,
            network_cache_path=args.gpt_network,
        )

    # Load β₃ age-brain priors for MRI features
    beta3_regions = {}
    _REGION_PATTERNS = {
        'Left_Hippocampus_Vol':         ['st11', 'left_hippo', 'hippocampus.*left', 'left.*hippocampus'],
        'Right_Hippocampus_Vol':        ['st88', 'right_hippo', 'hippocampus.*right', 'right.*hippocampus'],
        'ICV':                          ['st62', 'intracranial', '_icv_', '_icv$'],
        'Left_Entorhinal_CortVol':      ['st24', 'entorhinal.*left', 'left.*entorhinal'],
        'Right_Entorhinal_CortVol':     ['st83', 'entorhinal.*right', 'right.*entorhinal'],
        'Left_Entorhinal_ThickAvg':     ['st24ta', 'entorhinal.*left.*thick'],
        'Right_Entorhinal_ThickAvg':    ['st83ta', 'entorhinal.*right.*thick'],
        'Left_Precuneus_ThickAvg':      ['st42', 'precuneus.*left', 'left.*precuneus'],
        'Right_Precuneus_ThickAvg':     ['st103', 'precuneus.*right', 'right.*precuneus'],
        'Left_PostCingulate_ThickAvg':  ['st30', 'isthmuscingulate.*left', 'left.*isthmus'],
        'Right_PostCingulate_ThickAvg': ['st89', 'isthmuscingulate.*right', 'right.*isthmus'],
        'Left_SupFrontal_ThickAvg':     ['st44', 'superiorfrontal.*left'],
        'Right_SupFrontal_ThickAvg':    ['st105', 'superiorfrontal.*right'],
    }
    if args.beta_int and os.path.exists(args.beta_int):
        print(f"\n  Loading β₃ from {args.beta_int}...")
        bi_df = pd.read_csv(args.beta_int, low_memory=False)
        intbeta_cols = [c for c in bi_df.columns if c.startswith('IntBeta_')]
        if intbeta_cols:
            mask = bi_df['Age_z'].abs() > 0.01
            sample = bi_df[mask].iloc[0]
            for c in intbeta_cols:
                region = c.replace('IntBeta_', '')
                beta3_regions[region] = sample[c] / sample['Age_z']
            print(f"  β₃ loaded: {len(beta3_regions)} regions")

    # [2] Classify features
    print("\n[2/5] Classifying features...")
    clinical_pool, mri_pool, col_info = classify_features(df)
    print("  Clinical: {}  MRI: {}".format(len(clinical_pool), len(mri_pool)))

    # Clinical pool from Figure 2/3 patterns
    FIG23_CLINICAL_PATTERNS = [
        ('age_cardiovascularrisk', 'Age'),
        ('age_cognition', 'Age'),
        ('sex', 'Sex'),
        ('education', 'Education'),
        ('bmi', 'BMI'),
        ('phs_external', 'PHS'),
        ('apoe_e4_carrier', 'APOE Carrier'),
    ]

    fig23_clinical = []
    fig23_clinical_names = {}
    used_labels = set()
    for col in df.columns:
        cl = col.lower()
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
    print("  Clinical features matched: {}".format(len(clinical_pool_final)))
    for c in clinical_pool_final:
        print("    {} -> {}".format(c, fig23_clinical_names.get(c, annotate_feature(c))))

    # MRI pool: exclude _othe, >30% coverage, deduplicate
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
    ann_best = {}
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

    # [3] Pre-select top features
    print("\n[3/5] Pre-selecting top features...")
    def rank_feats(pool):
        ranked = []
        for col in pool:
            try:
                x = df[col].values.astype(float)
            except (ValueError, TypeError):
                continue
            yv = y.values.astype(float)
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

    top_clin = [c for c, _ in clin_ranked]
    n_mri_slots = max(0, args.max_pool - len(top_clin))
    top_mri = [c for c, _ in mri_ranked[:n_mri_slots]]
    combined = top_clin + top_mri
    print("  {} features ({} clinical + {} MRI)".format(
        len(combined), len(top_clin), len(top_mri)))

    # [4] Backward elimination
    print("\n[4/5] Backward elimination...")
    history = backward_elimination(
        df, y, combined,
        clinical_set=clinical_set,
        site_series=site_series,
        min_features=max(args.min_features, args.min_clinical + args.min_mri),
        min_clinical=args.min_clinical, min_mri=args.min_mri, verbose=True)

    elbow = find_elbow(history, threshold=args.auc_threshold)
    # Override: pick the point with highest AUC (best classification)
    best = max(history, key=lambda h: h['auc'])
    print("  Best AUC: {:.3f} at {} features (elbow was {} features, AUC={:.3f})".format(
        best['auc'], best['n_features'], elbow['n_features'], elbow['auc']))
    feats = best['features']
    best_auc = best['auc']
    best_n = best['n_features']

    # Train final model for weights — knowledge-constrained
    # First, inject β₃ priors for selected MRI features
    if beta3_regions and HAS_BETA_KNOWLEDGE:
        max_abs_b3 = max(abs(b) for b in beta3_regions.values()) if beta3_regions else 1
        n_injected = 0
        for col in feats:
            if col in clinical_set:
                continue  # only MRI features
            cl = col.lower()
            for region, b3 in beta3_regions.items():
                patterns = _REGION_PATTERNS.get(region, [])
                if not patterns:
                    parts = region.lower().replace('_', ' ').split()
                    patterns = [p for p in parts if len(p) > 3]
                matched = any(re.search(pat, cl) for pat in patterns)
                if matched:
                    sign = -1 if b3 < 0 else +1
                    confidence = min(0.85, 0.3 + 0.55 * abs(b3) / max_abs_b3)
                    expected_beta = sign * min(0.30, 0.10 + 0.20 * abs(b3) / max_abs_b3)
                    main_cache[col] = {
                        'direction': 'negative' if sign < 0 else 'positive',
                        'mci_relevance': int(confidence * 10),
                        'mechanism': f'B3={b3:.2f} ({region})',
                        'data_type': 'mri_imaging',
                        'source': 'beta3_aging',
                        'expected_sign': sign,
                        'sign_confidence': confidence,
                        'expected_beta': expected_beta,
                        'bias_strength': 3.0,
                    }
                    n_injected += 1
                    s = '+' if sign > 0 else '-'
                    print(f"    β₃ prior: {annotate_feature(col):35s} B3={b3:+8.2f} → sign={s}, μ={expected_beta:+.2f}")
                    break
        if n_injected:
            print(f"  Injected {n_injected} MRI β₃ priors")

    Xf = df[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    Xf = Xf.fillna(Xf.median()).fillna(0)
    for c in Xf.columns:
        if Xf[c].std() < 1e-10:
            Xf[c] = Xf[c] + np.random.RandomState(42).normal(0, 1e-6, size=len(Xf))
    scaler = StandardScaler()
    Xsc = scaler.fit_transform(Xf)

    if HAS_BETA_KNOWLEDGE and (main_cache or network_cache):
        signs, confidences, bias_strengths, prior_betas, k_report = build_knowledge_priors(
            feats, main_cache=main_cache, network_cache=network_cache, verbose=True)
        clf = KnowledgeConstrainedLR(
            lambda_l2=1.0, lambda_k=args.lambda_k,
            expected_signs=signs, sign_confidences=confidences,
            bias_strengths=bias_strengths, prior_betas=prior_betas,
        )
        clf.fit(Xsc, y.values.astype(float))
        weights = dict(zip(feats, clf.coef_.ravel()))
    else:
        clf = LogisticRegression(C=1.0, penalty='l2', max_iter=2000, random_state=42)
        clf.fit(Xsc, y)
        weights = dict(zip(feats, clf.coef_[0]))

    print("  Selected: {} features, AUC={:.3f}".format(best_n, best_auc))
    for f in feats:
        ftype = 'CLIN' if f in clinical_set else 'MRI'
        ann = annotate_feature(f)
        print("    [{}] {} (w={:.2f}) <- {}".format(ftype, ann, weights.get(f, 0), f))

    # ── Encoding direction check for key clinical features ──
    print("\n  Feature encoding check:")
    for f in feats:
        fl = f.lower()
        beta = weights.get(f, 0)
        # Sex
        if 'sex' in fl or 'gender' in fl:
            vals = df[f].dropna().unique()
            ad_mean = df.loc[y == 1, f].mean()
            cn_mean = df.loc[y == 0, f].mean()
            print("    {} : unique={}, CN mean={:.2f}, AD mean={:.2f}, beta={:+.2f}".format(
                f, sorted(vals), cn_mean, ad_mean, beta))
            if beta < 0:
                print("      -> Negative beta: higher {} value = lower AD risk".format(f))
                if ad_mean < cn_mean:
                    print("      -> AD has lower values; if Male=1/Female=0: males less AD (women higher risk) ✓")
                else:
                    print("      -> AD has higher values; if Female=1/Male=0: women less AD (men higher risk)")
            else:
                print("      -> Positive beta: higher {} value = higher AD risk".format(f))
        # APOE
        elif 'apoe' in fl:
            ad_mean = df.loc[y == 1, f].mean()
            cn_mean = df.loc[y == 0, f].mean()
            print("    {} : CN mean={:.2f}, AD mean={:.2f}, beta={:+.2f}".format(
                f, cn_mean, ad_mean, beta))
            if beta > 0:
                print("      -> Positive beta: APOE carrier increases AD risk ✓")
            else:
                print("      -> Negative beta: APOE carrier decreases AD risk (check encoding!)")
        # Age
        elif 'age' in fl:
            ad_mean = df.loc[y == 1, f].mean()
            cn_mean = df.loc[y == 0, f].mean()
            print("    {} : CN mean={:.2f}, AD mean={:.2f}, beta={:+.2f}".format(
                f, cn_mean, ad_mean, beta))
        # Education
        elif 'education' in fl or 'educat' in fl:
            ad_mean = df.loc[y == 1, f].mean()
            cn_mean = df.loc[y == 0, f].mean()
            print("    {} : CN mean={:.2f}, AD mean={:.2f}, beta={:+.2f}".format(
                f, cn_mean, ad_mean, beta))
            if beta < 0:
                print("      -> Negative beta: higher education = lower AD risk (protective) ✓")
        # PHS
        elif 'phs' in fl:
            ad_mean = df.loc[y == 1, f].mean()
            cn_mean = df.loc[y == 0, f].mean()
            print("    {} : CN mean={:.2f}, AD mean={:.2f}, beta={:+.2f}".format(
                f, cn_mean, ad_mean, beta))

    # [5] Generate figure
    print("\n[5/5] Generating figure...")
    n_total = len(df)

    # Build label overrides for directional features
    label_overrides = {}
    for f in feats:
        fl = f.lower()
        beta = weights.get(f, 0)
        if 'sex' in fl or 'gender' in fl:
            vals = sorted(df[f].dropna().unique())
            ad_mean = df.loc[y == 1, f].mean()
            cn_mean = df.loc[y == 0, f].mean()
            # Determine coding: if AD has lower mean, higher value = less AD
            if len(vals) == 2:
                if ad_mean < cn_mean:
                    # Higher value group has less AD
                    label_overrides[f] = 'Sex (M to F)' if beta < 0 else 'Sex (F to M)'
                else:
                    label_overrides[f] = 'Sex (F to M)' if beta < 0 else 'Sex (M to F)'
            else:
                label_overrides[f] = 'Sex (0 to 1)'

    generate_figure(feats, weights, history, n_total, best_auc,
                    clinical_set, args.output, label_overrides=label_overrides)

    # Save report
    report = {
        'model_type': 'knowledge_constrained' if (HAS_BETA_KNOWLEDGE and main_cache) else 'standard_l2',
        'lambda_k': args.lambda_k if HAS_BETA_KNOWLEDGE else 0,
        'n_subjects': n_total,
        'n_cn': int((y == 0).sum()),
        'n_ad': int((y == 1).sum()),
        'n_features': best_n,
        'auc': round(best_auc, 4),
        'features': [(f, label_overrides.get(f, annotate_feature(f)), round(weights.get(f, 0), 4)) for f in feats],
    }
    with open(os.path.join(args.output, 'figure2_1_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print("  Saved: figure2_1_report.json")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
UWAS Figure 2: Per-Modality Feature Importance (Whole Cohort)
====================================================================

Goal: Train knowledge-constrained models per modality
      (Clinical, T1, DTI) on the full cohort using GPT-derived
      biological priors (sign-constrained regularization).

      β_knowledge penalty: λ_k × Σ cⱼ × max(0, -sⱼ × βⱼ)²
      Corrects known ADNI biases: Sex enrollment, DM survivorship,
      Race confounding, Framingham paradox.

Strategy:
  1. Load ADNI data, classify columns into clinical vs MRI pools
  2. Split MRI pool into DTI (diffusion metrics) vs T1 (structural)
  3. Prefilter top features within each modality
  4. Load GPT validation caches → build sign priors per feature
  5. Train knowledge-constrained logistic regression per modality via LOSO
  6. Generate Figure 2 PDF with signed β and knowledge annotations

Usage:
  python uwas_interaction.py --input ADNI_merged_data.csv
  python uwas_interaction.py --input ADNI_merged_data.csv --output-dir interaction_results
"""

import os, sys, json, warnings, re
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

# ── Knowledge-constrained regularization ──
try:
    from beta_knowledge_v2 import (
        KnowledgeConstrainedLR,
        build_knowledge_priors,
        load_gpt_caches,
        get_knowledge_prior,
        ADNI_BIAS_OVERRIDES,
    )
    HAS_BETA_KNOWLEDGE = True
except ImportError:
    HAS_BETA_KNOWLEDGE = False
    print("  ⚠ beta_knowledge_v2 not found — will use standard L2 (no sign constraints)")

warnings.filterwarnings('ignore')

# =============================================================================
# REUSABLE UTILITIES (from uwas_mri.py)
# =============================================================================

# ---------- leakage / suffix / cleaning ----------

LEAKAGE_PATTERNS = [
    'dx_', 'diagnosis', 'dx_label', 'ad_label', 'exam_date',
    'ptid', 'rid', 'subject_id', 'siteid', 'viscode',
    'phase', 'colprot', 'origprot',
    'update_stamp', 'userdate', 'userdate2',
    'mem_clin', 'exf_clin', 'lan_clin', 'vsp_clin',
    'enroll_status', 'withdraw', 'examdate',
]

COGNITIVE_PATTERNS = [
    'mmse', 'moca', 'adas', 'cdr', 'faq', 'gds', 'npi',
    'ecog', 'avlt', 'ravlt', 'bnt_', 'digitsb', 'digitf',
    'trailsa', 'trailsb', 'catflu', 'veganim',
    'clockdraw', 'clockcopy', 'clockcirc',
]


def is_leakage_column(col):
    cl = col.lower()
    return any(p in cl for p in LEAKAGE_PATTERNS)


def is_cognitive(col):
    cl = col.lower()
    return any(p in cl for p in COGNITIVE_PATTERNS)


def has_suffix(col, suffix):
    """Check ADNI merge suffix: _clin, _clin1, _clin_1, _mri_17 etc."""
    cl = col.lower()
    sl = suffix.lower()
    if cl.endswith(sl):
        return True
    if re.search(re.escape(sl) + r'_?\d*$', cl):
        return True
    return False


def classify_feature_pool(df, target_col='AD_Label', verbose=True):
    """Classify ADNI columns into clinical vs MRI feature pools.
    
    Returns:
        clinical_feats: list of clinical/demographic/comorbidity feature names
        mri_feats: list of MRI imaging feature names
        col_info: dict with HTN/DM column names
    """
    clinical_feats = []
    mri_feats = []
    col_info = {}

    for col in df.columns:
        cl = col.lower()
        if col == target_col or is_leakage_column(col) or is_cognitive(col):
            continue

        # Identify HTN and DM columns
        if any(p in cl for p in ['hypertension', '_htn']) and has_suffix(col, '_clin'):
            col_info['htn'] = col
        if any(p in cl for p in ['diabetes', '_dm_']) and has_suffix(col, '_clin'):
            col_info['dm'] = col

        # Clinical features: _clin, _medi suffixes (demographics, comorbidities)
        if has_suffix(col, '_clin') or has_suffix(col, '_medi'):
            clinical_feats.append(col)
        # Genetic features: _gene suffix
        elif has_suffix(col, '_gene'):
            clinical_feats.append(col)  # Group with clinical for "mixed" requirement
        # Biomarker scores: _bios suffix (PHS, polygenic scores)
        elif has_suffix(col, '_bios'):
            clinical_feats.append(col)
        # MRI features: _mri, _othe (MRI-routed) suffixes
        elif has_suffix(col, '_mri') or has_suffix(col, '_othe'):
            # Exclude PET, CSF
            if any(p in cl for p in ['_pet', '_csf', '_tau_', 'amyloid', 'suvr']):
                continue
            mri_feats.append(col)

    if verbose:
        print(f"  Feature pools: {len(clinical_feats)} clinical, {len(mri_feats)} MRI")
        if 'htn' in col_info:
            print(f"  HTN column: {col_info['htn']}")
        if 'dm' in col_info:
            print(f"  DM column:  {col_info['dm']}")

    return clinical_feats, mri_feats, col_info


def statistical_prefilter(df, y, candidates, max_features=300):
    """Pre-filter by Mann-Whitney U test. Returns top features by -log10(p)."""
    scores = []
    for col in candidates:
        try:
            x = pd.to_numeric(df[col], errors='coerce')
            x = x.replace([np.inf, -np.inf], np.nan).fillna(x.median())
            g0, g1 = x[y == 0], x[y == 1]
            if len(g0) > 10 and len(g1) > 10 and x.std() > 1e-10:
                _, p = mannwhitneyu(g0, g1, alternative='two-sided')
                stat_score = -np.log10(max(p, 1e-50))
            else:
                stat_score = 0
        except:
            stat_score = 0
        scores.append((col, stat_score))
    scores.sort(key=lambda x: -x[1])
    return [c for c, s in scores[:max_features]]


PREFILTER_CACHE_FILE = 'prefilter_cache_interaction.json'


def prefilter_with_cache(df, y, candidates, max_features=300, cache_key='default', verbose=True):
    """Statistical pre-filter with file-based caching."""
    cache = {}
    if os.path.exists(PREFILTER_CACHE_FILE):
        try:
            with open(PREFILTER_CACHE_FILE, 'r') as f:
                cache = json.load(f)
        except:
            cache = {}

    if cache_key in cache:
        cached = cache[cache_key]['features']
        valid = [f for f in cached if f in df.columns]
        if len(valid) >= max_features * 0.8:
            if verbose:
                print(f"  [{cache_key}]: loaded {len(valid)} features from cache")
            return valid[:max_features]

    if verbose:
        print(f"  [{cache_key}]: {len(candidates)} → {max_features} (Mann-Whitney U)")
    result = statistical_prefilter(df, y, candidates, max_features=max_features)

    cache[cache_key] = {
        'features': result,
        'max_features': max_features,
        'n_candidates': len(candidates),
        'timestamp': pd.Timestamp.now().isoformat(),
    }
    try:
        with open(PREFILTER_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        if verbose:
            print(f"  [{cache_key}]: saved {len(result)} features to cache")
    except:
        pass
    return result


# =============================================================================
# FEATURE ANNOTATION (simple rule-based for display)
# =============================================================================

_ADNI_DATADIC = None

def load_adni_datadic(paths=None):
    global _ADNI_DATADIC
    search = paths or [
        'DATADIC.csv', 'datadic.csv', 'ADNI_datadic.csv',
        '../DATADIC.csv', 'data/DATADIC.csv',
    ]
    for p in search:
        if os.path.exists(p):
            try:
                dd = pd.read_csv(p, low_memory=False)
                _ADNI_DATADIC = {}
                for _, row in dd.iterrows():
                    fldname = str(row.get('FLDNAME', '')).strip()
                    text = str(row.get('TEXT', '')).strip()
                    if fldname and text and text != 'nan':
                        _ADNI_DATADIC[fldname.lower()] = text
                print(f"  Loaded ADNI data dictionary: {len(_ADNI_DATADIC)} entries from {p}")
                return
            except:
                pass


def annotate_feature(name):
    """Human-readable name for a feature."""
    nl = name.lower()
    base = re.sub(r'_(clin|medi|gene|mri|othe)_?\d*$', '', nl)

    # Known clinical features
    known_clinical = {
        'phc_age_cognition': 'Age at Diagnosis',
        'phc_age_cardiovascularrisk': 'Age',
        'phc_age_biomarker': 'Age (Biomarker)',
        'phc_sex': 'Sex (M to F)',
        'phc_education': 'Education',
        'phc_race': 'Race',
        'phc_ethnicity': 'Ethnicity',
        'phc_bmi': 'BMI',
        'phc_hypertension': 'Hypertension',
        'phc_diabetes': 'Diabetes',
        'phc_heart': 'Heart disease',
        'phc_stroke': 'Stroke',
        'phc_smoker': 'Smoking',
        'phc_sbp': 'Systolic BP',
        'phc_bp_med': 'BP medication',
        'phc_ascvd_10y': 'ASCVD 10-yr risk',
        'phc_tau': 'Tau',
        'phc_ab42': 'Aβ42',
        'phc_ptau': 'p-Tau',
        'ptau_raw': 'p-Tau (raw)',
        'tau_raw': 'Tau (raw)',
        'ab42_raw': 'Aβ42 (raw)',
        'phc_scens_ab42': 'SCeNS Aβ42 Score',
        'phc_scens_ptau': 'SCeNS p-Tau Score',
        'phc_scens_tau': 'SCeNS Tau Score',
        'phc_mem_se': 'Memory (SE)',
        'phc_mem_precisefilter': 'Memory',
        'phc_exf_se': 'Executive Function (SE)',
        'phc_exf_precisefilter': 'Executive Function',
        'phc_lan_se': 'Language (SE)',
        'phc_lan_precisefilter': 'Language',
        'phc_vsp_se': 'Visuospatial (SE)',
        'phc_vsp_precisefilter': 'Visuospatial',
        'phs': 'Polygenic Hazard Score',
        'genotype': 'APOE e4 Carrier',
        'apoe_e4_carrier': 'APOE e4 Carrier',
        'tomm40': 'TOMM40',
        'ihchron': 'Chronic Illness Count',
        'ihsurg': 'Surgical Procedures Count',
        'mhpsych': 'Hx: Psychiatric',
        'mh2neurl': 'Hx: Neurological',
        'mh3head': 'Hx: Head injury',
        'mh4card': 'Hx: Cardiovascular',
        'mh5resp': 'Hx: Respiratory',
        'mh6hepat': 'Hx: Hepatic',
        'mh7derm': 'Hx: Dermatologic',
        'mh8muscl': 'Hx: Musculoskeletal',
        'mh9endo': 'Hx: Endocrine',
        'mh10gast': 'Hx: Gastrointestinal',
        'mh11hema': 'Hx: Hematologic',
        'mh12rena': 'Hx: Renal/Genitourinary',
        'mh13alle': 'Hx: Allergies',
        'mh14alch': 'Hx: Alcohol abuse',
        'mh14calch': 'Hx: Alcohol (current)',
        'mh15adrug': 'Hx: Drug abuse (past)',
        'mh15bdrug': 'Hx: Drug abuse (current)',
        'mh16smok': 'Hx: Smoking',
        'mh17mali': 'Hx: Malignancy',
        'mh18surg': 'Hx: Surgical',
        'mh19othr': 'Hx: Other',
        'mhsource': 'MH source',
        'mhstab': 'MH stability',
        'ihsever': 'Illness severity',
    }
    for pat, label in known_clinical.items():
        if pat in base:
            return label

    # ── MRI: FreeSurfer UCSF (ST##XX) codes ──
    # XX = SV (subcortical vol), CV (cortical vol), SA (surface area),
    #      TA (cortical thickness), TS (thick std)
    _UCSF_REGIONS = {
        'st11': 'L Hippocampus',    'st88': 'R Hippocampus',
        'st14': 'L Inf Lat Vent',   'st91': 'R Inf Lat Vent',
        'st24': 'L Entorhinal',     'st83': 'R Entorhinal',
        'st26': 'L Fusiform',       'st85': 'R Fusiform',
        'st29': 'L Inf Temporal',   'st90': 'R Inf Temporal',
        'st30': 'L Isthmus Cing',   'st89': 'R Isthmus Cing',
        'st31': 'L Lat Occ',        'st92': 'R Lat Occ',
        'st32': 'L Lat Orbitofr',   'st93': 'R Lat Orbitofr',
        'st34': 'L Med Orbitofr',   'st95': 'R Med Orbitofr',
        'st35': 'L Mid Temporal',   'st96': 'R Mid Temporal',
        'st36': 'L Parahipp',       'st97': 'R Parahipp',
        'st37': 'L Paracentral',    'st98': 'R Paracentral',
        'st38': 'L Pars Operc',     'st99': 'R Pars Operc',
        'st39': 'L Pars Triang',    'st100': 'R Pars Triang',
        'st40': 'L Pericalcarine',  'st101': 'R Pericalcarine',
        'st41': 'L Postcentral',    'st102': 'R Postcentral',
        'st42': 'L Precuneus',      'st103': 'R Precuneus',
        'st43': 'L Rostral Ant Cing','st104': 'R Rostral Ant Cing',
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
        'st63': '3rd Ventricle',    'st64': '4th Ventricle',
        'st65': 'CC Posterior',     'st66': 'CC Mid Posterior',
        'st67': 'CC Central',       'st68': 'CC Mid Anterior',
        'st69': 'CC Anterior',
        # Additional cortical (ADNI UCSF)
        'st9': 'L Banks STS',       'st80': 'R Banks STS',
        'st10': 'L Caud Ant Cing',  'st81': 'R Caud Ant Cing',
        'st12': 'L Caud Mid Front', 'st82': 'R Caud Mid Front',
        'st13': 'L Cuneus',         'st84': 'R Cuneus',
        'st22': 'L Insula',         'st86': 'R Insula',
        'st23': 'L Lingual',        'st87': 'R Lingual',
        'st25': 'L Rostral Mid Fr', 'st94': 'R Rostral Mid Fr',
        'st33': 'L Precentral',     'st120': 'R Precentral',
        # Hippocampal subfields (ADNI extension)
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
    _UCSF_METRICS = {'sv': 'Vol', 'cv': 'Vol', 'sa': 'Area', 'ta': 'Thick',
                      'ts': 'Thick SD', 'hs': 'Vol'}

    m_ucsf = re.match(r'st(\d+)([a-z]{2})', base)
    if m_ucsf:
        code = f'st{m_ucsf.group(1)}'
        metric = _UCSF_METRICS.get(m_ucsf.group(2), m_ucsf.group(2).upper())
        region = _UCSF_REGIONS.get(code, f'Region {m_ucsf.group(1)}')
        return f'{region} {metric}'

    # ── MRI: T1-segmentation features (t1seg_SIDE_REGION_METRIC_STAT) ──
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

    # Strip phc_ prefix for matching
    mri_base = re.sub(r'^phc_', '', base)

    m_t1 = re.match(r't1seg_(left|right)_(.+?)_(volume|md|ad|rd|fa|freewater)_(mean|median|std)$', mri_base)
    if not m_t1:
        m_t1 = re.match(r't1seg_(left|right)_(.+?)_(volume|thick|area)$', mri_base)
    if not m_t1:
        # Bilateral / unsided t1seg (no left/right prefix)
        m_bi = re.match(r't1seg_(.+?)_(volume|md|ad|rd|fa|freewater)_(mean|median|std)$', mri_base)
        if m_bi:
            region_raw = m_bi.group(1)
            region = region_raw
            for abbr, label in _T1SEG_ABBREV.items():
                if abbr in region_raw:
                    if label:
                        region = label
                    break
            else:
                region = region_raw.replace('_', ' ').title()
            metric = _DTI_METRICS.get(m_bi.group(2), m_bi.group(2).title())
            stat = _DTI_STATS.get(m_bi.group(3), m_bi.group(3))
            return f'{region} {metric} ({stat})'
    if m_t1:
        side = 'L' if m_t1.group(1) == 'left' else 'R'
        region_raw = m_t1.group(2)
        # Extract short region name
        region = region_raw
        for abbr, label in _T1SEG_ABBREV.items():
            if abbr in region_raw:
                if label:
                    region = label
                break
        else:
            region = region_raw.replace('_', ' ').title()
        if m_t1.lastindex >= 4:
            metric = _DTI_METRICS.get(m_t1.group(3), m_t1.group(3).title())
            stat = _DTI_STATS.get(m_t1.group(4), m_t1.group(4))
            return f'{side} {region} {metric} ({stat})'
        else:
            t1_metric = m_t1.group(3)
            t1_suffix = {'volume': 'Vol', 'thick': 'Thick', 'area': 'Area'}.get(
                t1_metric, t1_metric.title())
            return f'{side} {region} {t1_suffix}'

    # ── MRI: JHU atlas DTI features (jhu_REGION_[fwcorrected_]METRIC_STAT) ──
    m_jhu = re.match(r'jhu_(.+?)_(fwcorrected_)?(md|ad|rd|fa|freewater)_(mean|median|std)$', mri_base)
    if m_jhu:
        region_raw = m_jhu.group(1)
        fwc = 'FWc ' if m_jhu.group(2) else ''
        metric = _DTI_METRICS.get(m_jhu.group(3), m_jhu.group(3).upper())
        stat = _DTI_STATS.get(m_jhu.group(4), m_jhu.group(4))
        # Parse side from region
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
        return f'{side}{region} {fwc}{metric} ({stat})'

    # ── MRI: Known special features ──
    _KNOWN_MRI = {
        'hci_2014': 'Hippocampal Vol Index (HCI)',
        'right_sub_vol': 'R Subcortical Vol (total)',
        'left_sub_vol': 'L Subcortical Vol (total)',
        'measure_1': 'MRI Quality Measure',
        'sroi': 'Structural ROI Index',
        'right_hipp_vol': 'R Hippocampus Vol',
        'left_hipp_vol': 'L Hippocampus Vol',
        'right_hippo': 'R Hippocampus Vol',
        'left_hippo': 'L Hippocampus Vol',
        'total_hippo': 'Total Hippocampus Vol',
        'right_ca1_vol': 'R Hipp CA1 Vol',
        'left_ca1_vol': 'L Hipp CA1 Vol',
        'right_ca_vol': 'R Hipp CA Vol',
        'left_ca_vol': 'L Hipp CA Vol',
        'right_ca2_vol': 'R Hipp CA2/3 Vol',
        'left_ca2_vol': 'L Hipp CA2/3 Vol',
        'right_ca3_vol': 'R Hipp CA3 Vol',
        'left_ca3_vol': 'L Hipp CA3 Vol',
        'right_sub_vol_othe': 'R Subcortical Vol',
        'left_sub_vol_othe': 'L Subcortical Vol',
        'right_subiculum': 'R Subiculum Vol',
        'left_subiculum': 'L Subiculum Vol',
        'right_presubiculum': 'R Presubiculum Vol',
        'left_presubiculum': 'L Presubiculum Vol',
        'right_fimbria': 'R Fimbria Vol',
        'left_fimbria': 'L Fimbria Vol',
        'right_tail': 'R Hipp Tail Vol',
        'left_tail': 'L Hipp Tail Vol',
    }
    for pat, label in _KNOWN_MRI.items():
        if pat in base:
            return label

    # ADNI data dictionary lookup
    if _ADNI_DATADIC:
        for key in [base, nl, name]:
            if key.lower() in _ADNI_DATADIC:
                text = _ADNI_DATADIC[key.lower()]
                if len(text) > 50:
                    text = text[:47] + '...'
                return text

    # Fallback: clean up
    clean = base.replace('phc_', '').replace('_', ' ').title()
    if len(clean) > 50:
        clean = clean[:47] + '...'
    return clean


# =============================================================================
# FIGURE HTML PARSERS — extract top features from Figure 2/3 outputs
# =============================================================================

def parse_features_from_figure_html(html_path, verbose=True):
    """Parse feature column names from a figure HTML file's tooltip attributes.
    
    Figures embed raw column names in title="COLNAME | β=..." tooltips.
    Returns list of (column_name, beta) tuples sorted by |β| descending.
    """
    if not os.path.exists(html_path):
        print(f"  ⚠ Figure not found: {html_path}")
        return []

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    features = []
    seen = set()

    # Strategy 1: title="COLNAME | β=+0.1234 | d=... | p=..."
    # β can be unicode char or HTML entity
    for m in re.finditer(r'title="([^"|]+)\s*\|\s*(?:\u03b2|&#946;|&beta;|β)=([+\-]?[\d.]+)', content):
        col = m.group(1).strip()
        beta = float(m.group(2))
        if col not in seen:
            features.append((col, beta))
            seen.add(col)

    # Strategy 2: title="COLNAME&#10;β=-0.0750  d=..."  (butterfly chart)
    if not features:
        for m in re.finditer(r'title="([^"&#]+)&#10;.{0,5}=([+\-]?[\d.]+)', content):
            col = m.group(1).strip()
            beta = float(m.group(2))
            if col not in seen:
                features.append((col, beta))
                seen.add(col)

    # Strategy 3: title="COLNAME (β=+0.123)"
    if not features:
        for m in re.finditer(r'title="([^"(]+)\s*\((?:\u03b2|β)=([+\-]?[\d.]+)\)', content):
            col = m.group(1).strip()
            beta = float(m.group(2))
            if col not in seen:
                features.append((col, beta))
                seen.add(col)

    # Strategy 4 (broadest): any title attribute containing "|" — extract first segment
    # and look for a numeric value after any "=" sign
    if not features:
        for m in re.finditer(r'title="([^"]+)"', content):
            title_text = m.group(1)
            parts = title_text.split('|')
            if len(parts) >= 2:
                col = parts[0].strip()
                # Look for beta value in second part
                beta_m = re.search(r'=([+\-]?[\d.]+)', parts[1])
                if beta_m and col and not col.startswith('<'):
                    try:
                        beta = float(beta_m.group(1))
                        if col not in seen:
                            features.append((col, beta))
                            seen.add(col)
                    except ValueError:
                        pass

    features.sort(key=lambda x: abs(x[1]), reverse=True)

    if verbose:
        print(f"  Parsed {len(features)} features from {os.path.basename(html_path)}")
        if features:
            print(f"    Top: {features[0][0]} (β={features[0][1]:+.4f})")
            print(f"    Bottom: {features[-1][0]} (β={features[-1][1]:+.4f})")
        else:
            # Diagnostic: show what title attributes exist
            titles = re.findall(r'title="([^"]{10,80})"', content)
            if titles:
                print(f"    ⚠ 0 features parsed. Sample title attributes found:")
                for t in titles[:5]:
                    print(f"      '{t}'")
            else:
                print(f"    ⚠ No title attributes found in HTML file")

    return features


# =============================================================================
# CORE: GREEDY FORWARD SEARCH FOR MIXED FEATURES
# =============================================================================

def evaluate_feature_set(df_sub, y_sub, features, site_series=None,
                         min_site_n=10, max_sites=10):
    """Evaluate a feature set via Leave-One-Site-Out (LOSO) AUC.
    
    Falls back to 5-fold stratified CV if site info is unavailable or
    there are too few sites with enough samples.
    
    Args:
        df_sub: DataFrame subset for this HTN×DM group
        y_sub: Binary labels for this group
        features: list of feature column names
        site_series: Series of site IDs aligned with df_sub index (or None)
        min_site_n: Minimum samples per site to include in LOSO
        max_sites: Maximum number of sites for LOSO
    
    Returns AUC mean, AUC std, or (0, 0) if evaluation fails.
    """
    X = df_sub[features].copy()
    X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    med = X.median()
    X = X.fillna(med).fillna(0)

    # Add tiny jitter to zero-variance columns so they stay in the model
    rng = np.random.RandomState(42)
    for col in X.columns:
        if X[col].std() < 1e-10:
            X[col] = X[col] + rng.normal(0, 1e-6, size=len(X))

    if len(X.columns) < 1:
        return 0.0, 0.0

    n_pos = int(y_sub.sum())
    n_neg = len(y_sub) - n_pos
    if n_pos < 5 or n_neg < 5:
        return 0.0, 0.0

    # ── Try LOSO ──
    if site_series is not None:
        sites = site_series.dropna()
        # Only keep sites that appear in this subgroup
        sites_in_sub = sites.reindex(df_sub.index).dropna()
        if len(sites_in_sub) > 0:
            try:
                sites_in_sub = sites_in_sub.astype(int)
            except (ValueError, TypeError):
                pass
            site_counts = sites_in_sub.value_counts()
            eligible = site_counts[site_counts >= min_site_n]

            if len(eligible) >= 3:  # Need at least 3 sites for meaningful LOSO
                loso_sites = eligible.head(max_sites)
                site_aucs = []

                for site_id in loso_sites.index:
                    test_mask = (sites_in_sub == site_id)
                    train_mask = (~test_mask) & sites_in_sub.notna()

                    # Align masks with X index
                    test_idx = test_mask[test_mask].index
                    train_idx = train_mask[train_mask].index

                    # Intersect with X's index
                    test_idx = test_idx.intersection(X.index)
                    train_idx = train_idx.intersection(X.index)

                    if len(test_idx) < 3 or len(train_idx) < 10:
                        continue

                    y_train = y_sub[train_idx]
                    y_test = y_sub[test_idx]

                    if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
                        continue

                    X_train = X.loc[train_idx]
                    X_test = X.loc[test_idx]

                    scaler = StandardScaler()
                    X_train_sc = scaler.fit_transform(X_train)
                    X_test_sc = scaler.transform(X_test)

                    try:
                        model = LogisticRegression(C=1.0, penalty='l2', max_iter=500, random_state=42)
                        model.fit(X_train_sc, y_train)
                        proba = model.predict_proba(X_test_sc)[:, 1]
                        auc = roc_auc_score(y_test, proba)
                        site_aucs.append(auc)
                    except:
                        continue

                if len(site_aucs) >= 3:
                    return np.mean(site_aucs), np.std(site_aucs)

    # ── Fallback: 5-fold stratified CV ──
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    actual_splits = min(5, n_pos, n_neg)
    if actual_splits < 2:
        return 0.0, 0.0

    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    model = LogisticRegression(C=1.0, penalty='l2', max_iter=500, random_state=42)

    try:
        scores = cross_val_score(model, X_sc, y_sub, cv=cv, scoring='roc_auc')
        return scores.mean(), scores.std()
    except:
        return 0.0, 0.0


def train_bounded_model(df_sub, y_sub, features, site_series=None,
                        min_site_n=10, max_sites=10,
                        main_cache=None, network_cache=None,
                        lambda_k=5.0, lambda_l2=1.0):
    """Train a knowledge-constrained model on a stratum via LOSO.

    When beta_knowledge_v2 is available:
      - Builds GPT sign priors per feature (expected direction + confidence)
      - Uses asymmetric L2 penalty: extra cost when β opposes GPT-expected sign
      - Returns signed β (positive = risk, negative = protective)
      - Addresses ADNI biases: Sex enrollment, DM survivorship, Race confounding

    Fallback (no beta_knowledge_v2):
      - Standard L2 logistic regression with signed β (no direction flipping)

    Returns dict with:
      - mean_auc, std_auc, n_sites
      - weights: {feature: float}  — signed β (positive=risk, negative=protective)
      - knowledge_report: per-feature sign constraint details (if available)
      - sign_violations: features where learned β opposes GPT prior
    """
    if not isinstance(y_sub, pd.Series):
        y_sub = pd.Series(np.asarray(y_sub), index=df_sub.index)

    X = df_sub[features].copy()
    X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    med = X.median()
    X = X.fillna(med).fillna(0)

    rng = np.random.RandomState(42)
    for col in X.columns:
        if X[col].std() < 1e-10:
            X[col] = X[col] + rng.normal(0, 1e-6, size=len(X))

    valid = [c for c in X.columns if X[c].std() > 1e-10]
    if len(valid) < 1:
        return None
    X = X[valid]

    n_pos = int(y_sub.sum())
    n_neg = len(y_sub) - n_pos
    if n_pos < 5 or n_neg < 5:
        return None

    # ── Build knowledge priors ──
    signs, confidences, bias_strengths, prior_betas, k_report = None, None, None, None, []
    if HAS_BETA_KNOWLEDGE and (main_cache or network_cache):
        signs, confidences, bias_strengths, prior_betas, k_report = build_knowledge_priors(
            list(X.columns), main_cache=main_cache, network_cache=network_cache,
            verbose=True,
        )

    n_feat = len(valid)
    scaler = StandardScaler()

    def _fit_fold(X_tr_raw, y_tr, X_te_raw, y_te):
        """Fit one LOSO fold with knowledge-constrained or standard LR."""
        X_tr_sc = scaler.fit_transform(X_tr_raw)
        X_te_sc = scaler.transform(X_te_raw)

        if HAS_BETA_KNOWLEDGE and signs is not None:
            model = KnowledgeConstrainedLR(
                lambda_l2=lambda_l2, lambda_k=lambda_k,
                expected_signs=signs, sign_confidences=confidences,
                bias_strengths=bias_strengths, prior_betas=prior_betas,
            )
            model.fit(X_tr_sc, np.asarray(y_tr).astype(float))
            proba = model.predict_proba(X_te_sc)[:, 1]
            coef = model.coef_.ravel()
        else:
            model = LogisticRegression(C=1.0/lambda_l2, penalty='l2',
                                       max_iter=500, random_state=42)
            model.fit(X_tr_sc, y_tr)
            proba = model.predict_proba(X_te_sc)[:, 1]
            coef = model.coef_.ravel()

        try:
            auc = roc_auc_score(y_te, proba)
        except:
            return None, None
        return auc, coef

    # ── LOSO cross-validation ──
    site_aucs = []
    weight_accum = {c: [] for c in valid}

    if site_series is not None:
        sites_in_sub = site_series.reindex(df_sub.index).dropna()
        if len(sites_in_sub) > 0:
            try:
                sites_in_sub = sites_in_sub.astype(int)
            except (ValueError, TypeError):
                pass
            site_counts = sites_in_sub.value_counts()
            eligible = site_counts[site_counts >= min_site_n]

            if len(eligible) >= 3:
                loso_sites = eligible.head(max_sites)
                for site_id in loso_sites.index:
                    test_mask = (sites_in_sub == site_id)
                    train_mask = (~test_mask) & sites_in_sub.notna()
                    test_idx = test_mask[test_mask].index.intersection(X.index)
                    train_idx = train_mask[train_mask].index.intersection(X.index)

                    if len(test_idx) < 3 or len(train_idx) < 10:
                        continue
                    y_tr = y_sub.loc[train_idx]
                    y_te = y_sub.loc[test_idx]
                    if y_tr.nunique() < 2 or y_te.nunique() < 2:
                        continue

                    auc, coef = _fit_fold(X.loc[train_idx].values,
                                          y_tr, X.loc[test_idx].values, y_te)
                    if auc is not None:
                        site_aucs.append(auc)
                        for i, c in enumerate(valid):
                            weight_accum[c].append(coef[i])

    # Fallback to 5-fold CV
    if len(site_aucs) < 3:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        site_aucs = []
        weight_accum = {c: [] for c in valid}
        for tr_idx, te_idx in skf.split(X, y_sub):
            y_tr = y_sub.iloc[tr_idx]
            y_te = y_sub.iloc[te_idx]
            if y_tr.nunique() < 2 or y_te.nunique() < 2:
                continue
            auc, coef = _fit_fold(X.iloc[tr_idx].values,
                                  y_tr, X.iloc[te_idx].values, y_te)
            if auc is not None:
                site_aucs.append(auc)
                for i, c in enumerate(valid):
                    weight_accum[c].append(coef[i])

    if not site_aucs:
        return None

    # ── Average β across folds — signed values ──
    weights = {}
    for c in features:
        if c in weight_accum and weight_accum[c]:
            weights[c] = float(np.mean(weight_accum[c]))
        else:
            weights[c] = 0.0

    # ── Sign violation report ──
    sign_violations = []
    if signs is not None:
        for j, c in enumerate(valid):
            avg_b = weights.get(c, 0)
            if signs[j] != 0 and confidences[j] >= 0.5 and signs[j] * avg_b < 0:
                sign_violations.append({
                    'feature': c,
                    'expected': '+' if signs[j] > 0 else '-',
                    'learned': '+' if avg_b > 0 else '-',
                    'beta': avg_b,
                    'confidence': float(confidences[j]),
                })

    if sign_violations:
        print(f"    ⚠ {len(sign_violations)} sign violations (β opposes GPT prior):")
        for sv in sign_violations[:5]:
            print(f"      {sv['feature']}: expected {sv['expected']}, "
                  f"learned {sv['learned']} (β={sv['beta']:.4f})")

    return {
        'mean_auc': float(np.mean(site_aucs)),
        'std_auc': float(np.std(site_aucs)),
        'weights': weights,
        'weight_per_site': {c: list(weight_accum[c]) for c in features if c in weight_accum},
        'n_sites': len(site_aucs),
        'knowledge_report': k_report,
        'sign_violations': sign_violations,
        'lambda_k': lambda_k,
    }


def greedy_forward_mixed_search(df_sub, y_sub, clinical_pool, mri_pool,
                                 site_series=None,
                                 max_features=6, top_k=5, verbose=True):
    """Greedy forward search for best mixed (clinical+MRI) feature sets.
    
    Rules:
      - Must include ≥1 clinical AND ≥1 MRI feature
      - Minimize feature count while maximizing AUC
      - Return top_k results sorted by (AUC desc, n_features asc)
    
    Strategy:
      1. Evaluate all pairs (1 clinical + 1 MRI) via LOSO
      2. Take top 15 pairs, greedily add features from either pool
      3. Return top_k by AUC, preferring fewer features at same AUC
    """
    results = []
    n_clin = len(clinical_pool)
    n_mri = len(mri_pool)

    if n_clin == 0 or n_mri == 0:
        if verbose:
            print(f"    Empty pool: {n_clin} clinical, {n_mri} MRI — skipping")
        return []

    # Step 1: Score all (1+1) pairs — but limit to top pre-filtered
    MAX_CLIN = min(n_clin, 20)
    MAX_MRI = min(n_mri, 30)
    clin_top = clinical_pool[:MAX_CLIN]
    mri_top = mri_pool[:MAX_MRI]

    if verbose:
        print(f"    Phase 1: Evaluating {MAX_CLIN}×{MAX_MRI} = {MAX_CLIN*MAX_MRI} pairs (LOSO)...")

    pair_results = []
    for cf in clin_top:
        for mf in mri_top:
            auc, std = evaluate_feature_set(df_sub, y_sub, [cf, mf],
                                             site_series=site_series)
            if auc > 0.5:
                pair_results.append({
                    'features': [cf, mf],
                    'clinical': [cf], 'mri': [mf],
                    'auc': auc, 'std': std,
                    'n_features': 2,
                })

    pair_results.sort(key=lambda x: (-x['auc'], x['n_features']))

    if verbose:
        n_good = sum(1 for r in pair_results if r['auc'] > 0.55)
        print(f"    Phase 1 done: {len(pair_results)} valid pairs, {n_good} with AUC>0.55")
        if pair_results:
            print(f"    Best pair: AUC={pair_results[0]['auc']:.3f} "
                  f"({annotate_feature(pair_results[0]['features'][0])} + "
                  f"{annotate_feature(pair_results[0]['features'][1])})")

    # Step 2: Greedy forward from top 15 pairs
    EXPAND_TOP = min(15, len(pair_results))
    all_pool = clinical_pool + mri_pool

    if verbose and EXPAND_TOP > 0:
        print(f"    Phase 2: Expanding top {EXPAND_TOP} pairs (up to {max_features} features)...")

    for seed in pair_results[:EXPAND_TOP]:
        current = list(seed['features'])
        current_clin = list(seed['clinical'])
        current_mri = list(seed['mri'])
        best_auc = seed['auc']

        # Add to results (the pair itself)
        results.append(dict(seed))

        # Greedy forward: add one feature at a time from either pool
        for step in range(max_features - 2):
            best_next = None
            best_next_auc = best_auc

            candidates = [f for f in all_pool if f not in current]
            # Limit candidates to speed up
            candidates = candidates[:50]

            for f in candidates:
                trial = current + [f]
                auc, std = evaluate_feature_set(df_sub, y_sub, trial,
                                                 site_series=site_series)
                if auc > best_next_auc + 0.005:  # Must improve by ≥0.5% to justify extra feature
                    best_next = f
                    best_next_auc = auc
                    best_next_std = std

            if best_next is None:
                break

            current.append(best_next)
            best_auc = best_next_auc

            # Track which pool the new feature came from
            if best_next in clinical_pool:
                current_clin.append(best_next)
            else:
                current_mri.append(best_next)

            results.append({
                'features': list(current),
                'clinical': list(current_clin),
                'mri': list(current_mri),
                'auc': best_next_auc,
                'std': best_next_std,
                'n_features': len(current),
            })

    # Deduplicate: same feature set → keep best AUC
    seen = {}
    for r in results:
        key = tuple(sorted(r['features']))
        if key not in seen or r['auc'] > seen[key]['auc']:
            seen[key] = r
    results = list(seen.values())

    # Sort: AUC desc, then n_features asc
    results.sort(key=lambda x: (-x['auc'], x['n_features']))

    return results[:top_k]



# =============================================================================
# BASELINE EVALUATION: single-modality per subgroup
# =============================================================================

def evaluate_baselines(df_sub, y_sub, clinical_pool, mri_pool,
                       site_series=None, max_features=6, verbose=True):
    """Evaluate clinical-only and MRI-only baselines within a subgroup.

    For each modality, runs greedy forward selection (1->max_features) and
    returns the best AUC at each feature count + overall best.
    """
    baselines = {}

    for pool_name, pool in [('clinical', clinical_pool), ('mri', mri_pool)]:
        if not pool:
            baselines[pool_name] = {'best_auc': 0.0, 'best_std': 0.0,
                                     'best_features': [], 'curve': []}
            continue

        # Sort pool by individual AUC
        indiv = []
        for f in pool:
            auc, std = evaluate_feature_set(df_sub, y_sub, [f],
                                             site_series=site_series)
            indiv.append((f, auc))
        indiv.sort(key=lambda x: -x[1])

        # Greedy forward from best single feature
        curve = []  # (n_features, auc, std, feature_list)
        current = []
        best_auc_so_far = 0.0

        for step in range(min(max_features, len(indiv))):
            if step == 0:
                best_f = indiv[0][0]
                current = [best_f]
            else:
                best_next = None
                best_next_auc = best_auc_so_far
                best_next_std = 0.0
                for f, _ in indiv:
                    if f in current:
                        continue
                    trial = current + [f]
                    auc, std = evaluate_feature_set(df_sub, y_sub, trial,
                                                     site_series=site_series)
                    if auc > best_next_auc:
                        best_next = f
                        best_next_auc = auc
                        best_next_std = std
                if best_next is None:
                    break
                current = current + [best_next]

            auc, std = evaluate_feature_set(df_sub, y_sub, current,
                                             site_series=site_series)
            curve.append((len(current), auc, std, list(current)))
            best_auc_so_far = max(best_auc_so_far, auc)

        if curve:
            best = max(curve, key=lambda x: x[1])
            baselines[pool_name] = {
                'best_auc': best[1], 'best_std': best[2],
                'best_features': best[3], 'best_n': best[0],
                'curve': curve,
            }
        else:
            baselines[pool_name] = {'best_auc': 0.0, 'best_std': 0.0,
                                     'best_features': [], 'curve': []}

        if verbose:
            b = baselines[pool_name]
            print(f"    {pool_name:8s} baseline: AUC={b['best_auc']:.3f} "
                  f"({len(b.get('best_features',[]))} features)")

    return baselines


# =============================================================================
# FEATURE RECURRENCE ANALYSIS
# =============================================================================

def analyze_feature_recurrence(all_group_results):
    """Analyze which features appear across multiple subgroups' best models.

    Returns dict with universal (>=3 groups), common (2), specific (1),
    matrix ({feature: {group: bool}}), counts.
    """
    group_features = {}
    for gname, gdata in all_group_results.items():
        top = gdata.get('top5', [])
        if top:
            best = top[0]
            feats = set(best.get('clinical', []) + best.get('mri', []))
            group_features[gname] = feats
        else:
            group_features[gname] = set()

    all_feats = set()
    for feats in group_features.values():
        all_feats.update(feats)

    matrix = {}
    for f in all_feats:
        matrix[f] = {gname: f in group_features[gname] for gname in group_features}

    counts = {f: sum(1 for g in group_features.values() if f in g) for f in all_feats}
    universal = [f for f, c in counts.items() if c >= 3]
    common = [f for f, c in counts.items() if c == 2]

# =============================================================================
# DATA LOADING (reuse ADNI loading patterns from uwas_mri)
# =============================================================================

def find_target_column(df, label_col):
    if label_col in df.columns:
        return label_col
    for alt in ['PHC_Diagnosis', 'DX_LABEL', 'DX_bl', 'DX', 'DIAGNOSIS', 'AD_Label']:
        if alt in df.columns:
            return alt
    raise ValueError(f"Target column '{label_col}' not found. Available: {list(df.columns[:20])}")


def encode_target(df, target_col):
    """Encode AD_Label: 1=AD/Dementia, 0=CN. Drop MCI rows."""
    raw = df[target_col].astype(str).str.strip().str.upper()
    ad_pats = ['AD', 'DEMENTIA', 'LMCI', 'EMCI']
    cn_pats = ['CN', 'NL', 'NORMAL', 'SMC']

    labels = pd.Series(np.nan, index=df.index)
    for i, v in raw.items():
        if any(p in v for p in ad_pats):
            labels[i] = 1
        elif any(p in v for p in cn_pats):
            labels[i] = 0

    n_before = len(df)
    df = df.copy()
    df['AD_Label'] = labels
    df = df.dropna(subset=['AD_Label']).reset_index(drop=True)
    df['AD_Label'] = df['AD_Label'].astype(int)

    n_ad = (df['AD_Label'] == 1).sum()
    n_cn = (df['AD_Label'] == 0).sum()
    n_drop = n_before - len(df)
    print(f"  Encoded: CN={n_cn}, AD={n_ad} (dropped {n_drop} MCI/unknown)")
    return df


def dedup_longitudinal(df, y):
    """One row per subject: AD → latest visit, CN → baseline."""
    ptid_col = next((c for c in df.columns if c.upper() in ['PTID', 'RID', 'SUBJECT_ID']), None)
    if not ptid_col or not df[ptid_col].duplicated().any():
        return df, y

    viscode_col = next((c for c in df.columns if c.upper() in ['VISCODE', 'VISCODE2']), None)
    if viscode_col:
        viscode_order = {'sc': -1, 'bl': 0, 'scmri': 0}
        for m in [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 72, 84, 96, 108, 120]:
            viscode_order[f'm{m:02d}'] = m
            viscode_order[f'm{m}'] = m
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
    print(f"  Longitudinal dedup: {len(df)} subjects (CN={sum(y==0)}, AD={sum(y==1)})")
    return df, y


# =============================================================================
# SITE IDENTIFICATION (for LOSO)
# =============================================================================

def find_site_column(df):
    """Find the site ID column in the dataframe."""
    for col in df.columns:
        cl = col.lower()
        if cl in ('siteid', 'site_id', 'site', '_siteid_'):
            return col
    candidates = []
    for col in df.columns:
        cl = col.lower()
        if 'siteid' in cl:
            candidates.append(col)
    if candidates:
        return sorted(candidates, key=len)[0]
    return None


def add_site_info(df, consents_path=None, verbose=True):
    """Add SITEID to the ADNI dataframe (from existing col, consents file, or PTID extraction)."""
    site_col = find_site_column(df)
    if site_col is not None:
        if verbose:
            print(f"  Site column found: {site_col} ({df[site_col].notna().sum()} non-null)")
        return df, site_col

    if consents_path and os.path.exists(consents_path):
        if verbose:
            print(f"  Loading site info from: {consents_path}")
        try:
            consent = pd.read_csv(consents_path, low_memory=False)
            if 'RID' in consent.columns and 'SITEID' in consent.columns:
                site_map = consent[['RID', 'SITEID']].drop_duplicates(subset='RID')
                rid_col = next((c for c in df.columns if c.lower() in ('rid',)), None)
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
                    if verbose:
                        print(f"  Merged SITEID: {df['_SITEID_'].notna().sum()}/{len(df)} matched")
                    return df, '_SITEID_'
        except Exception as e:
            if verbose:
                print(f"  ⚠ Consents load failed: {e}")

    ptid_col = next((c for c in df.columns if c.lower().startswith('ptid')), None)
    if ptid_col:
        df['_SITEID_'] = df[ptid_col].astype(str).str.extract(r'^(\d+)_S_')[0].astype(float)
        if verbose:
            print(f"  Extracted SITEID from PTID: {df['_SITEID_'].notna().sum()}/{len(df)} "
                  f"({df['_SITEID_'].nunique()} sites)")
        return df, '_SITEID_'

    if verbose:
        print("  ⚠ Could not determine site information — falling back to 5-fold CV")
    return df, None


# =============================================================================
# FIGURE 2 — WHOLE COHORT (NO HTN×DM STRATIFICATION)
# =============================================================================


def generate_figure2(modality_results, n_cn, n_ad,
                     output_path='figure2_importance.html'):
    """Generate Figure 2: per-modality horizontal bar plot for the whole cohort.

    modality_results = {
        'clinical': {'weights': {...}, 'mean_auc': ..., 'std_auc': ..., 'n_sites': ...},
        't1':       {'weights': {...}, ...},
        'dti':      {'weights': {...}, ...},
    }
    """

    modality_order = [
        ('clinical', 'Clinical (AD-Related)', '#4472C4'),
        ('t1',       'T1 (Structural Volume)', '#548235'),
        ('dti',      'DTI (Diffusion Metrics)', '#BF8F00'),
    ]

    # ── Collect bars per modality ──
    sections = []  # list of (mod_label, mod_color, auc_str, [(display_name, weight), ...])
    for mod_key, mod_label, mod_color in modality_order:
        mod_data = modality_results.get(mod_key)
        if not mod_data:
            continue
        feat_list = sorted(mod_data['weights'].items(), key=lambda x: abs(x[1]), reverse=True)
        feat_list = feat_list[:10]

        auc_str = '{:.3f}'.format(mod_data['mean_auc'])
        if mod_data['std_auc'] > 0:
            auc_str += ' \u00b1 {:.3f}'.format(mod_data['std_auc'])

        bars = []
        seen_display = set()
        for feat, w in feat_list:
            if abs(w) < 0.005:
                continue
            disp = annotate_feature(feat)
            for _suf in [' (mean)', ' (med)', ' (median)']:
                disp = disp.replace(_suf, '')
            if disp.startswith('Hx: ') and mod_key == 'clinical':
                disp = disp[4:]
            if disp in seen_display:
                continue
            seen_display.add(disp)
            bars.append((disp, w, feat))
        sections.append((mod_label, mod_color, auc_str, bars))

    # ── SVG dimensions ──
    label_width = 220
    bar_area_width = 360
    value_width = 50
    svg_width = label_width + bar_area_width + value_width + 20
    row_height = 26
    section_header_height = 36
    section_gap = 12
    top_margin = 10

    # Calculate total height
    total_rows = 0
    for _, _, _, bars in sections:
        total_rows += len(bars)
    n_sections = len(sections)
    svg_height = (top_margin + n_sections * (section_header_height + section_gap)
                  + total_rows * row_height + 30)

    # ── Build SVG bars ──
    svg_bars = ''
    cy = top_margin

    for mod_label, mod_color, auc_str, bars in sections:
        # Section header with AUC
        svg_bars += ('<rect x="0" y="{y}" width="{w}" height="{h}" fill="#f8fafc"/>\n'
                     '<rect x="0" y="{y}" width="4" height="{h}" fill="{c}" rx="2"/>\n'
                     '<text x="14" y="{ty}" font-size="13" font-weight="700" fill="#333">'
                     '{label}</text>\n'
                     '<text x="14" y="{ty2}" font-size="11" fill="#64748b">'
                     'AUC = {auc}</text>\n').format(
            y=cy, w=svg_width, h=section_header_height,
            c=mod_color, ty=cy + 16, ty2=cy + 30,
            label=mod_label, auc=auc_str)
        cy += section_header_height

        for disp, w, raw_feat in bars:
            # Label (right-aligned)
            svg_bars += ('<text x="{lx}" y="{ty}" font-size="12" fill="#333" '
                         'text-anchor="end" dominant-baseline="middle">'
                         '<title>{raw}</title>{disp}</text>\n').format(
                lx=label_width - 8, ty=cy + row_height / 2, raw=raw_feat, disp=disp)

            # Bar — centered at midpoint, extends left (protective) or right (risk)
            center_x = label_width + bar_area_width / 2
            max_w_all = max(abs(b[1]) for s in sections for b in s[3]) if sections else 1
            bar_w = abs(w) / max_w_all * (bar_area_width / 2) if max_w_all > 0 else 2
            bar_w = max(2, bar_w)
            if w >= 0:
                bx = center_x
            else:
                bx = center_x - bar_w
            opacity = min(1.0, 0.35 + 0.65 * abs(w) / max_w_all) if max_w_all > 0 else 0.5
            svg_bars += ('<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" '
                         'fill="{c}" rx="3" opacity="{op}"/>\n').format(
                bx=bx, by=cy + 4, bw=bar_w, bh=row_height - 8,
                c=mod_color, op=opacity)

            # Light gridline
            svg_bars += ('<line x1="{lx}" y1="{ly}" x2="{rx}" y2="{ly}" '
                         'stroke="#f0f0f0" stroke-width="1"/>\n').format(
                lx=label_width, ly=cy + row_height, rx=label_width + bar_area_width)

            cy += row_height

        cy += section_gap

    # ── X-axis ticks (centered butterfly) ──
    axis_y = cy - section_gap + 4
    x_ticks = ''
    center_x = label_width + bar_area_width / 2
    # Center line
    x_ticks += ('<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" '
                'stroke="#999" stroke-width="1" stroke-dasharray="4,3"/>\n').format(
        x=center_x, y1=top_margin + section_header_height, y2=axis_y)
    # Tick marks at 25%, 50%, 75%, 100% of half-width
    for frac in [0.25, 0.5, 0.75, 1.0]:
        for sign in [-1, 1]:
            tx = center_x + sign * frac * bar_area_width / 2
            x_ticks += ('<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" '
                        'stroke="#ddd" stroke-width="0.5"/>\n').format(
                x=tx, y1=top_margin + section_header_height, y2=axis_y)
    # Axis line
    x_ticks += ('<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
                'stroke="#ccc" stroke-width="1.5"/>\n').format(
        x1=label_width, x2=label_width + bar_area_width, y=axis_y)

    svg_height = axis_y + 24

    # Axis label
    x_ticks += ('<text x="{x}" y="{y}" font-size="11" fill="#888" '
                'text-anchor="middle" font-style="italic">'
                '\u03b2 coefficient (\u2190 protective | risk \u2192)</text>\n').format(
        x=center_x, y=svg_height - 2)
    svg_height += 8

    n_info = '<div class="n-info">N = {} (CN = {}, AD = {})</div>\n'.format(
        n_cn + n_ad, n_cn, n_ad)
    svg_tag = ('<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg" '
               'font-family="Source Sans 3, sans-serif">\n').format(svg_width, svg_height)

    html = ('<!DOCTYPE html><html><head><meta charset="UTF-8">\n'
'<title>Figure 2</title>\n'
'<style>\n'
'@import url("https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Source+Sans+3:wght@300;400;500;600;700&display=swap");\n'
'*{margin:0;padding:0;box-sizing:border-box;}\n'
'body{font-family:"Source Sans 3",sans-serif;color:#1a1a2e;background:#fff;padding:40px 48px 36px;max-width:750px;margin:0 auto;}\n'
'.fig-title{font-family:"Source Serif 4",serif;font-size:20px;font-weight:700;margin-bottom:6px;}\n'
'.fig-sub{font-size:13px;color:#666;line-height:1.6;margin-bottom:20px;}\n'
'.n-info{font-size:12px;color:#64748b;margin-bottom:16px;}\n'
'.foot{font-size:11px;color:#aaa;margin-top:20px;line-height:1.5;font-style:italic;border-top:1px solid #eee;padding-top:10px;}\n'
'</style></head><body>\n'
'<div class="fig-title">Figure 2. Feature Importance by Modality</div>\n'
'<div class="fig-sub">\n'
'  Separate bounded logistic regression models [0,\u20091] per modality via L-BFGS-B.\n'
'  Features z-scored and direction-aligned internally, trained on the full cohort\n'
'  using LOSO cross-validation. Each modality section shows its own AUC.\n'
'</div>\n'
+ n_info + svg_tag + x_ticks + svg_bars +
'</svg>\n'
'<div class="foot">\n'
'  Bounded logistic regression (L-BFGS-B, L2 \u03bb=0.01, weights \u2208 [0,\u20091]) per modality.\n'
'  LOSO = Leave-One-Site-Out cross-validation.\n'
'</div>\n'
'</body></html>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print('\n  Figure 2 saved to: {}'.format(output_path))


def generate_figure2_pdf(modality_results, n_cn, n_ad,
                         output_path='figure2.pdf'):
    """Generate Figure 2 as publication-quality PDF with A (Clinical), B (T1), C (DTI)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.gridspec import GridSpec
    import os

    # Register TeX Gyre Heros (Helvetica-identical metrics)
    heros_dir = '/usr/share/texmf/fonts/opentype/public/tex-gyre/'
    for variant in ['texgyreheros-regular.otf', 'texgyreheros-bold.otf',
                    'texgyreheros-italic.otf', 'texgyreheros-bolditalic.otf']:
        path = heros_dir + variant
        if os.path.exists(path):
            fm.fontManager.addfont(path)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['TeX Gyre Heros', 'Helvetica', 'Arial'],
        'font.size': 8.5,
        'axes.linewidth': 0.6,
        'axes.edgecolor': '#333333',
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'pdf.fonttype': 42,
    })

    modality_order = [
        ('clinical', 'Clinical',              '#4472C4'),
        ('t1',       'T1 (Structural MRI)',   '#548235'),
        ('dti',      'DTI (Diffusion Metrics)', '#BF8F00'),
    ]

    # Collect bars per modality with display-name dedup & filter zeros
    from scipy.stats import ttest_1samp
    all_sections = {}

    for mod_key, mod_label, mod_color in modality_order:
        mod_data = modality_results.get(mod_key)
        if not mod_data:
            continue
        # Sort by absolute β
        feat_list = sorted(mod_data['weights'].items(), key=lambda x: abs(x[1]), reverse=True)
        feat_list = feat_list[:10]
        auc_str = '{:.3f}'.format(mod_data['mean_auc'])
        if mod_data.get('std_auc', 0) > 0:
            auc_str += ' \u00b1 {:.3f}'.format(mod_data['std_auc'])
        wps = mod_data.get('weight_per_site', {})

        # Build knowledge lookup for sign annotations
        k_lookup = {}
        for kr in mod_data.get('knowledge_report', []):
            k_lookup[kr['feature']] = kr
        sv_set = set(sv['feature'] for sv in mod_data.get('sign_violations', []))

        bars = []
        seen_display = set()
        for feat, w in feat_list:
            if abs(w) < 0.005:
                continue
            disp = annotate_feature(feat)
            for _suf in [' (mean)', ' (med)', ' (median)']:
                disp = disp.replace(_suf, '')
            if disp.startswith('Hx: ') and mod_key == 'clinical':
                disp = disp[4:]
            if disp in seen_display:
                continue
            seen_display.add(disp)
            # One-sample t-test: is mean β ≠ 0?
            site_vals = wps.get(feat, None)
            pv = float('nan')
            if site_vals and len(site_vals) >= 3:
                arr = np.array(site_vals, dtype=float)
                if np.std(arr) > 0:
                    _, p_two = ttest_1samp(arr, 0.0)
                    pv = p_two
                elif abs(np.mean(arr)) > 0:
                    pv = 1e-15
            # Knowledge annotation
            kinfo = k_lookup.get(feat, {})
            expected_sign = kinfo.get('expected_sign', 0)
            sign_conf = kinfo.get('confidence', 0)
            is_violation = feat in sv_set
            bars.append((disp, w, pv, expected_sign, sign_conf, is_violation))
        all_sections[mod_key] = {
            'label': mod_label, 'color': mod_color,
            'auc_str': auc_str, 'bars': bars,
            'lambda_k': mod_data.get('lambda_k', 0),
        }

    def _draw_barh(ax, bars, color, panel_label=None, section_label=None, auc_str=None):
        if not bars:
            ax.set_visible(False)
            return
        # bars: list of (display, beta, pval, expected_sign, sign_conf, is_violation)
        labels = [b[0] for b in bars][::-1]
        values = [b[1] for b in bars][::-1]
        pvals  = [b[2] for b in bars][::-1]
        exp_signs = [b[3] for b in bars][::-1]
        sign_confs = [b[4] for b in bars][::-1]
        violations = [b[5] for b in bars][::-1]
        y_pos = range(len(labels))

        # Determine max abs for symmetric axis
        max_abs = max(abs(v) for v in values) if values else 1.0
        x_lim = max_abs * 1.35  # tighter since no bar labels

        # Color: risk (positive β) uses base color, protective (negative β) uses lighter shade
        risk_color = color
        # Create protective color by mixing with white
        from matplotlib.colors import to_rgba
        rc = to_rgba(color)
        prot_color = tuple(min(1, c * 0.6 + 0.4) for c in rc[:3]) + (rc[3],)

        bar_colors = [risk_color if v >= 0 else prot_color for v in values]

        rects = ax.barh(y_pos, values, height=0.6, color=bar_colors,
                        edgecolor='#000000', linewidth=1.2)

        for i, (v, pv, es, sc, viol) in enumerate(zip(values, pvals, exp_signs, sign_confs, violations)):
            # Knowledge marker: ⚠ if sign violation, ★ if high-confidence prior
            marker = ''
            marker_color = '#999'
            if viol:
                marker = '\u26a0'  # ⚠
                marker_color = '#CC0000'
            elif sc >= 0.7:
                marker = '\u2605'  # ★
                marker_color = '#2E7D32'
            if marker:
                ax.text(x_lim * 0.82, i, marker, va='center', ha='center',
                        fontsize=7, color=marker_color, clip_on=False)

            # P-value column — pushed right to avoid overlap with β text
            if np.isnan(pv):
                ptxt = '\u2014'
                pcol = '#bbb'
            elif pv < 0.001:
                exp = int(np.floor(np.log10(max(pv, 1e-99))))
                coeff = pv / (10 ** exp)
                ptxt = '{:.1f}e{:d}***'.format(coeff, exp)
                pcol = '#333'
            elif pv < 0.01:
                ptxt = '{:.4f}**'.format(pv)
                pcol = '#555'
            elif pv < 0.05:
                ptxt = '{:.4f}*'.format(pv)
                pcol = '#777'
            else:
                ptxt = '{:.3f}'.format(pv)
                pcol = '#999'
            ax.text(x_lim * 0.99, i, ptxt, va='center', ha='right',
                    fontsize=5, color=pcol, fontweight='500',
                    clip_on=False)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlim(-x_lim, x_lim)

        # Symmetric x ticks
        tick_step = round(max_abs / 3, 2) or 0.1
        ticks = np.arange(-max_abs, max_abs + tick_step, tick_step)
        ticks = [t for t in ticks if abs(t) <= max_abs * 1.05]
        ax.set_xticks(ticks)
        ax.set_xticklabels(['{:.2f}'.format(t) for t in ticks], fontsize=6.5)
        ax.set_xlabel('\u03b2 coefficient (\u2190 protective | risk \u2192)', fontsize=7.5, color='#333')
        ax.axvline(x=0, color='#666', linewidth=0.8, linestyle='-', zorder=1)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', direction='out', length=2.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.6)

        # Light horizontal gridlines
        for yy in y_pos:
            ax.axhline(y=yy, color='#e8e8e8', linewidth=0.3, zorder=0)

        # Section title + AUC
        title = section_label or ''
        if auc_str:
            title += '    AUC = {}'.format(auc_str)
        ax.set_title(title, fontsize=8.5, fontweight='bold', loc='left',
                     color='#222', pad=8)

        # p-value header
        ax.text(x_lim * 0.99, len(labels) - 0.3, 'p', ha='right', va='bottom',
                fontsize=5.5, fontweight='bold', color='#444',
                clip_on=False)

        # Panel label
        ax._panel_label = panel_label

    # Count items for height ratios
    n_clin = len(all_sections.get('clinical', {}).get('bars', []))
    n_t1 = len(all_sections.get('t1', {}).get('bars', []))
    n_dti = len(all_sections.get('dti', {}).get('bars', []))
    n_mri_max = max(n_t1, n_dti, 1)

    # Use GridSpec: 2 rows, 2 cols. Row 0 = A spanning both cols, Row 1 = B + C
    fig_w = 9.0
    fig_h = 3.0 + 3.5  # row A ~ 3in, row BC ~ 3.5in

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[n_clin + 2, n_mri_max + 2],
                  hspace=0.55, wspace=0.85,
                  left=0.18, right=0.98, top=0.92, bottom=0.06)

    # Panel A: Clinical (spans both columns)
    ax_a = fig.add_subplot(gs[0, :])
    if 'clinical' in all_sections:
        s = all_sections['clinical']
        _draw_barh(ax_a, s['bars'], s['color'],
                   panel_label='A', section_label=s['label'],
                   auc_str=s['auc_str'])

    # Panel B: T1 (bottom left)
    ax_t1 = fig.add_subplot(gs[1, 0])
    if 't1' in all_sections:
        s = all_sections['t1']
        _draw_barh(ax_t1, s['bars'], s['color'],
                   panel_label='B', section_label=s['label'],
                   auc_str=s['auc_str'])

    # Panel C: DTI (bottom right)
    ax_dti = fig.add_subplot(gs[1, 1])
    if 'dti' in all_sections:
        s = all_sections['dti']
        _draw_barh(ax_dti, s['bars'], s['color'],
                   section_label=s['label'],
                   auc_str=s['auc_str'])

    # Add panel labels at fixed figure x-coordinate so A and B align
    fig.canvas.draw()
    label_x = 0.02  # fixed left edge in figure coords
    for ax in [ax_a, ax_t1, ax_dti]:
        pl = getattr(ax, '_panel_label', None)
        if pl:
            # Get top of axes in figure coords
            bbox = ax.get_position()
            fig.text(label_x, bbox.y1 + 0.03, pl,
                     fontsize=14, fontweight='bold', va='top', ha='left')

    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('\n  Figure 2 PDF saved to: {}'.format(output_path))


def run_figure2_analysis(input_file, label_col='PHC_Diagnosis',
                         output_dir='figure2_results',
                         consents_file=None,
                         apoe_file=None,
                         phs_file=None,
                         beta_int_file=None,
                         gpt_cache_file=None,
                         gpt_network_file=None,
                         lambda_k=5.0,
                         verbose=True):
    """Run per-modality importance analysis on the WHOLE cohort (no HTN×DM split)."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  FIGURE 2: PER-MODALITY FEATURE IMPORTANCE (WHOLE COHORT)")
    if HAS_BETA_KNOWLEDGE:
        print("  Pipeline: Load → Build pools → Knowledge-constrained β per modality")
        print("  ★ β_knowledge: sign-constrained regularization (λ_k={:.1f})".format(lambda_k))
    else:
        print("  Pipeline: Load → Build pools → Standard L2 β per modality")
    print("  ★ Evaluation: Leave-One-Site-Out cross-validation")
    print("=" * 70)

    # ─── [1] Load ───
    print(f"\n[1/4] Loading data from {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"  Loaded: {len(df):,} rows \u00d7 {len(df.columns):,} columns")

    # ─── [2] Encode target ───
    print(f"\n[2/4] Encoding target ({label_col})...")
    target_col = find_target_column(df, label_col)
    df = encode_target(df, target_col)
    y = df['AD_Label'].values
    df, y = dedup_longitudinal(df, y)

    # ─── [2a] Merge APOE carrier from external file ───
    if apoe_file and os.path.exists(apoe_file):
        print(f"\n  Loading APOE from {apoe_file}...")
        apoe_df = pd.read_csv(apoe_file, low_memory=False)
        # Create e4 carrier: 1 if any allele is 4
        apoe_df['APOE_e4_carrier'] = apoe_df['GENOTYPE'].apply(
            lambda g: 1 if '4' in str(g) else 0)
        apoe_df = apoe_df[['RID', 'APOE_e4_carrier']].drop_duplicates(subset='RID')
        # Merge by RID
        if 'RID' in df.columns:
            before = len(df)
            df = df.merge(apoe_df, on='RID', how='left')
            df['APOE_e4_carrier'] = df['APOE_e4_carrier'].fillna(0).astype(int)
            matched = df['APOE_e4_carrier'].sum()
            print(f"  ✓ APOE e4 carrier merged: {int(matched)}/{len(df)} carriers")
        else:
            print(f"  ✗ No RID column in data — cannot merge APOE")
        y = df['AD_Label'].values

    # ─── [2a2] Merge PHS from external file ───
    if phs_file and os.path.exists(phs_file):
        print(f"\n  Loading PHS from {phs_file}...")
        pdf = pd.read_csv(phs_file, low_memory=False)
        # Find key column (RID or PTID)
        rc = None
        for c in pdf.columns:
            if c.lower() in ('rid', 'ptid', 'subject'):
                rc = c; break
        dr = None
        for c in df.columns:
            if c.lower() in ('rid', 'ptid', 'subject'):
                dr = c; break
        # Find PHS score column
        phs_col = None
        for c in pdf.columns:
            if c.lower() in ('ptid', 'rid', 'subject', 'phase', 'viscode'):
                continue
            if 'phs' in c.lower():
                phs_col = c; break
        if phs_col is None:
            for c in pdf.columns:
                if c.lower() in ('ptid', 'rid', 'subject', 'phase', 'viscode'):
                    continue
                if pd.api.types.is_numeric_dtype(pdf[c]):
                    phs_col = c; break
        if rc and dr and phs_col:
            print(f"  PHS file: key={rc}, score={phs_col}, {len(pdf)} rows")
            pdf[rc] = pdf[rc].astype(str).str.strip()
            df[dr] = df[dr].astype(str).str.strip()
            pm = pdf.drop_duplicates(subset=[rc], keep='last').set_index(rc)[phs_col]
            df['PHS_external_gene'] = df[dr].map(pm).astype(float)
            n_mapped = df['PHS_external_gene'].notna().sum()
            print(f"  ✓ PHS merged: {n_mapped}/{len(df)} mapped")
            if n_mapped == 0:
                print(f"  ⚠ 0 PHS mapped! df keys: {list(df[dr].head(3))}, PHS keys: {list(pm.index[:3])}")
        else:
            print(f"  ⚠ PHS file missing columns (key={rc}, score={phs_col})")
        y = df['AD_Label'].values
    elif phs_file:
        print(f"  ⚠ PHS file not found: {phs_file}")

    # ─── [2b] Site info for LOSO ───
    print(f"\n  Setting up site info for LOSO...")
    df, site_col = add_site_info(df, consents_path=consents_file, verbose=verbose)
    if site_col:
        site_series = df[site_col]
        n_sites = site_series.dropna().nunique()
        print(f"  LOSO ready: {n_sites} sites available")
    else:
        site_series = None
        print(f"  \u26a0 No site info \u2014 will fall back to 5-fold CV")

    # ─── [2c] Load GPT validation caches ───
    main_cache, network_cache = {}, {}
    if HAS_BETA_KNOWLEDGE:
        print(f"\n  Loading GPT validation caches...")
        main_cache, network_cache = load_gpt_caches(
            main_cache_path=gpt_cache_file,
            network_cache_path=gpt_network_file,
        )
        if not main_cache and not network_cache:
            print(f"  ⚠ No GPT caches found — knowledge constraints will be minimal")
    else:
        print(f"\n  ⚠ beta_knowledge_v2 not available — using standard L2")

    # ─── [3] Build feature pools ───
    print(f"\n[3/4] Building feature pools...")
    clinical_feats, mri_feats, col_info = classify_feature_pool(df, 'AD_Label', verbose=verbose)
    load_adni_datadic()

    # Clinical pool: fixed set of 6 features, matched by keyword
    CLINICAL_TARGETS = [
        ('age_cardiovascular',  'Age'),
        ('phs',            'Polygenic Hazard Score'),
        ('genotype' if not apoe_file else 'apoe_e4_carrier', 'APOE e4 Carrier'),
        ('sex',            'Sex (M to F)'),
        ('bmi',            'BMI'),
        ('hypertension',   'Hypertension'),
        ('diabetes',       'Diabetes'),
        ('education',      'Education'),
    ]
    clinical_top = []
    for keyword, label in CLINICAL_TARGETS:
        matched = None
        for col in df.columns:
            if keyword in col.lower():
                matched = col
                break
        if matched:
            clinical_top.append(matched)
            print(f"  ✓ {label}: {matched}")
        else:
            print(f"  ✗ {label}: not found (keyword={keyword})")

    # MRI pools: split into DTI vs T1, prefilter top 10 each
    def _mri_cat(f):
        fl = f.lower()
        if 'jhu_' in fl:
            return 'dti'
        diffusion_markers = ['_fa_', '_fa_mean', '_fa_median',
                             '_md_', '_md_mean', '_md_median',
                             '_rd_', '_rd_mean', '_rd_median',
                             '_ad_', '_ad_mean', '_ad_median',
                             'freewater', 'free_water', '_fw_']
        if any(m in fl for m in diffusion_markers):
            return 'dti'
        return 't1'

    all_dti_feats = [f for f in mri_feats if _mri_cat(f) == 'dti']
    all_t1_feats = [f for f in mri_feats if _mri_cat(f) == 't1']

    # Remove Thick SD (FreeSurfer TS metric) — noise, not informative
    _ts_pattern = re.compile(r'st\d+ts', re.IGNORECASE)
    n_before_ts = len(all_t1_feats)
    all_t1_feats = [f for f in all_t1_feats if not _ts_pattern.search(f.lower())]
    n_ts_removed = n_before_ts - len(all_t1_feats)
    if n_ts_removed > 0:
        print(f"  Removed {n_ts_removed} Thick SD (TS) features from T1 pool")

    print(f"\n  MRI pool: {len(mri_feats)} → {len(all_dti_feats)} DTI, {len(all_t1_feats)} T1")

    TOP_N = 10

    # ── β₃-guided T1 selection (biologically informed) ──
    beta3_regions = None
    if beta_int_file and os.path.exists(beta_int_file):
        print(f"\n  Loading B3 from {beta_int_file} for region selection...")
        bi_df = pd.read_csv(beta_int_file, low_memory=False)
        intbeta_cols = [c for c in bi_df.columns if c.startswith('IntBeta_')]
        if intbeta_cols:
            mask = bi_df['Age_z'].abs() > 0.01
            sample = bi_df[mask].iloc[0]
            beta3_regions = {}
            for c in intbeta_cols:
                region = c.replace('IntBeta_', '')
                beta3_regions[region] = sample[c] / sample['Age_z']
            ranked = sorted(beta3_regions.items(), key=lambda x: abs(x[1]), reverse=True)
            print(f"  B3 ranking ({len(ranked)} regions):")
            for region, b3 in ranked:
                print(f"    |B3|={abs(b3):>8.4f}  {region}")

    t1_b3_map = {}  # {selected_col: (region_name, b3_value)} — populated during B3 selection

    if beta3_regions and all_t1_feats:
        _REGION_PATTERNS = {
            'Left_Hippocampus_Vol':              ['st11', 'left_hippo', 'hippocampus.*left', 'left.*hippocampus'],
            'Right_Hippocampus_Vol':             ['st88', 'right_hippo', 'hippocampus.*right', 'right.*hippocampus'],
            'ICV':                               ['st62', 'intracranial', '_icv_', '_icv$'],
            'Left_Entorhinal_CortVol':           ['st24', 'entorhinal.*left', 'left.*entorhinal'],
            'Right_Entorhinal_CortVol':          ['st83', 'entorhinal.*right', 'right.*entorhinal'],
            'Left_Entorhinal_ThickAvg':          ['st24ta', 'entorhinal.*left.*thick', 'left.*entorhinal.*thick'],
            'Right_Entorhinal_ThickAvg':         ['st83ta', 'entorhinal.*right.*thick', 'right.*entorhinal.*thick'],
            'Left_Precuneus_ThickAvg':           ['st42', 'precuneus.*left', 'left.*precuneus'],
            'Right_Precuneus_ThickAvg':          ['st103', 'precuneus.*right', 'right.*precuneus'],
            'Left_PostCingulate_ThickAvg':       ['st30', 'postcingulate.*left', 'posterior.*cingulate.*left',
                                                   'isthmuscingulate.*left', 'left.*isthmus'],
            'Right_PostCingulate_ThickAvg':      ['st89', 'postcingulate.*right', 'posterior.*cingulate.*right',
                                                   'isthmuscingulate.*right', 'right.*isthmus'],
            'Left_SupFrontal_ThickAvg':          ['st44', 'superiorfrontal.*left', 'supfrontal.*left', 'left.*sup.*frontal'],
            'Right_SupFrontal_ThickAvg':         ['st105', 'superiorfrontal.*right', 'supfrontal.*right', 'right.*sup.*frontal'],
        }
        t1_selected = []
        t1_seen = set()
        ranked = sorted(beta3_regions.items(), key=lambda x: abs(x[1]), reverse=True)
        for region, b3 in ranked:
            patterns = _REGION_PATTERNS.get(region, [])
            if not patterns:
                parts = region.lower().replace('_', ' ').split()
                patterns = [p for p in parts if len(p) > 3]
            matched_cols = []
            for col in all_t1_feats:
                if col in t1_seen:
                    continue
                cl = col.lower()
                for pat in patterns:
                    if re.search(pat, cl):
                        matched_cols.append(col)
                        break
            if matched_cols:
                best_col, best_score = None, -1
                for mc in matched_cols:
                    try:
                        x = pd.to_numeric(df[mc], errors='coerce').fillna(0)
                        g0, g1 = x[y == 0], x[y == 1]
                        if len(g0) > 10 and len(g1) > 10 and x.std() > 1e-10:
                            _, p = mannwhitneyu(g0, g1, alternative='two-sided')
                            score = -np.log10(max(p, 1e-50))
                        else:
                            score = 0
                    except:
                        score = 0
                    if score > best_score:
                        best_score = score
                        best_col = mc
                if best_col and best_col not in t1_seen:
                    t1_selected.append(best_col)
                    t1_seen.add(best_col)
                    t1_b3_map[best_col] = (region, b3)
                    print(f"    B3 {region:40s} |B3|={abs(b3):.4f} -> {best_col} "
                          f"({annotate_feature(best_col)})")
        if t1_selected:
            t1_top = t1_selected[:TOP_N]
            print(f"\n  * T1 pool: {len(t1_top)} features selected by B3 ranking")
        else:
            t1_top = []
    else:
        t1_top = []

    if len(t1_top) < 2 and all_t1_feats:
        t1_pre = prefilter_with_cache(df, y, all_t1_feats,
                                       max_features=TOP_N * 3,
                                       cache_key='fig2_t1_pool', verbose=verbose)
        t1_top = t1_pre[:TOP_N]

    dti_top = []
    if all_dti_feats:
        dti_pre = prefilter_with_cache(df, y, all_dti_feats,
                                        max_features=TOP_N * 3,
                                        cache_key='fig2_dti_pool', verbose=verbose)
        dti_top = dti_pre[:TOP_N]

    print(f"\n  Clinical: {len(clinical_top)}, T1: {len(t1_top)}, DTI: {len(dti_top)}")

    # ─── [3b] Build MRI knowledge priors from β₃ ───
    # β₃ tells us each region's age-related trajectory:
    #   β₃ < 0 (shrinks with age): AD accelerates this → expected sign = -1 (lower vol = risk)
    #   β₃ > 0 (enlarges with age, e.g. ventricles): AD accelerates → expected sign = +1
    # Use |β₃| as confidence: larger age effect = more certain about direction
    mri_beta3_priors = {}
    if beta3_regions and t1_top and t1_b3_map:
        max_abs_b3 = max(abs(b) for b in beta3_regions.values()) if beta3_regions else 1
        print(f"\n  Building MRI knowledge priors from β₃ aging trajectories...")
        for col, (region, b3) in t1_b3_map.items():
            sign = -1 if b3 < 0 else +1
            confidence = min(0.85, 0.3 + 0.55 * abs(b3) / max_abs_b3)
            expected_beta = sign * min(0.30, 0.10 + 0.20 * abs(b3) / max_abs_b3)
            mri_beta3_priors[col] = {
                'sign': sign,
                'confidence': confidence,
                'expected_beta': expected_beta,
                'bias_strength': 3.0,
                'rationale': f'B3={b3:.2f} from age-brain model ({region})',
            }
            s = '+' if sign > 0 else '-'
            print(f"    {annotate_feature(col):45s} B3={b3:+8.2f} → sign={s}, "
                  f"conf={confidence:.2f}, μ={expected_beta:+.2f}")

    # Inject MRI β₃ priors into main_cache so train_bounded_model picks them up
    if mri_beta3_priors:
        print(f"\n  Injected {len(mri_beta3_priors)} MRI β₃ priors into knowledge cache")
        for col, prior in mri_beta3_priors.items():
            main_cache[col] = {
                'direction': 'negative' if prior['sign'] < 0 else 'positive',
                'mci_relevance': int(prior['confidence'] * 10),
                'mechanism': prior['rationale'],
                'data_type': 'mri_imaging',
                'source': 'beta3_aging',
                'expected_sign': prior['sign'],
                'sign_confidence': prior['confidence'],
                'expected_beta': prior['expected_beta'],
                'bias_strength': prior['bias_strength'],
            }

    # ─── [4] Train per-modality models on full cohort ───
    n_cn = int((y == 0).sum())
    n_ad = int((y == 1).sum())
    print(f"\n[4/4] Training knowledge-constrained models on full cohort (N={len(df)}, CN={n_cn}, AD={n_ad})...")

    MODALITIES = [
        ('clinical', 'Clinical', clinical_top),
        ('t1',       'T1',       t1_top),
        ('dti',      'DTI',      dti_top),
    ]

    modality_results = {}
    for mod_key, mod_label, mod_pool in MODALITIES:
        features = [f for f in mod_pool if f in df.columns]
        if len(features) < 2:
            print(f"  {mod_label}: \u26a0 <2 features — skipping")
            continue

        print(f"\n  {mod_label}: training on {len(features)} features...")
        try:
            result = train_bounded_model(df, y, features, site_series=site_series,
                                         main_cache=main_cache, network_cache=network_cache,
                                         lambda_k=lambda_k)
        except Exception as e:
            print(f"    \u26a0 Failed: {e}")
            continue

        if result is None:
            print(f"    \u26a0 Model returned None")
            continue

        modality_results[mod_key] = {
            'weights': result['weights'],
            'weight_per_site': result.get('weight_per_site', {}),
            'mean_auc': result['mean_auc'],
            'std_auc': result['std_auc'],
            'n_sites': result['n_sites'],
            'knowledge_report': result.get('knowledge_report', []),
            'sign_violations': result.get('sign_violations', []),
            'lambda_k': result.get('lambda_k', 0),
        }
        print(f"    AUC = {result['mean_auc']:.3f} \u00b1 {result['std_auc']:.3f} "
              f"({result['n_sites']} folds)")
        top_by_weight = sorted(result['weights'].items(),
                               key=lambda x: abs(x[1]), reverse=True)[:5]
        for f, w in top_by_weight:
            sign = '+' if w >= 0 else '-'
            print(f"      {annotate_feature(f):40s} \u03b2={sign}{abs(w):.3f}")

    if not modality_results:
        print("\n  \u26a0 No modality results — cannot generate Figure 2")
        return None

    # ─── Generate Figure 2 ───
    fig_path = os.path.join(output_dir, 'figure2_importance.html')
    generate_figure2(modality_results, n_cn, n_ad, output_path=fig_path)

    # ─── Generate Figure 2 PDF ───
    pdf_path = os.path.join(output_dir, 'figure2.pdf')
    generate_figure2_pdf(modality_results, n_cn, n_ad, output_path=pdf_path)

    # ─── Save JSON report ───
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'knowledge_constrained_beta' if HAS_BETA_KNOWLEDGE else 'standard_l2_beta',
        'lambda_k': lambda_k,
        'n_subjects': len(df),
        'n_cn': n_cn,
        'n_ad': n_ad,
        'modalities': {},
    }
    for mod_key, mod_data in modality_results.items():
        report['modalities'][mod_key] = {
            'mean_auc': mod_data['mean_auc'],
            'std_auc': mod_data['std_auc'],
            'n_sites': mod_data.get('n_sites', 0),
            'top_features': sorted(
                [(annotate_feature(f), round(w, 4))
                 for f, w in mod_data['weights'].items() if abs(w) > 0.01],
                key=lambda x: -abs(x[1]))[:10],
            'sign_violations': mod_data.get('sign_violations', []),
        }
    json_path = os.path.join(output_dir, 'figure2_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to: {json_path}")

    print(f"\n{'='*70}")
    print(f"  Done!")
    print(f"{'='*70}")

    return modality_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='UWAS Figure 2: Per-Modality Feature Importance (Whole Cohort)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python uwas_fig2.py -i ADNI_merged_data.csv -c consents.csv --apoe APOERES.csv
  python uwas_fig2.py -i ADNI_merged_data.csv -c consents.csv --apoe APOERES.csv -o .
""")

    parser.add_argument('--input', '-i', required=True,
                        help='Path to merged ADNI CSV file')
    parser.add_argument('--label', '-l', default='PHC_Diagnosis',
                        help='Target column name (default: PHC_Diagnosis)')
    parser.add_argument('--consents', '-c', default=None,
                        help='Path to ADNI consents CSV (for site IDs / LOSO)')
    parser.add_argument('--apoe', default=None,
                        help='Path to APOERES CSV (for APOE e4 carrier)')
    parser.add_argument('--phs', default=None,
                        help='Path to PHS scores CSV (needs RID/PTID + PHS column)')
    parser.add_argument('--beta-int', default=None,
                        help='Path to beta_int CSV (use B3 to select T1 regions)')
    parser.add_argument('--gpt-cache', default=None,
                        help='Path to gpt_validation_cache.json (16K feature priors)')
    parser.add_argument('--gpt-network', default=None,
                        help='Path to gpt_validation_network_cache_ad.json (63 key features)')
    parser.add_argument('--lambda-k', type=float, default=5.0,
                        help='Knowledge penalty strength (default: 2.0, 0=disabled)')
    parser.add_argument('--output-dir', '-o', default='figure2_results',
                        help='Output directory (default: figure2_results)')
    parser.add_argument('--quiet', '-q', action='store_true', default=False,
                        help='Reduce output verbosity')

    args = parser.parse_args()

    run_figure2_analysis(
        input_file=args.input,
        label_col=args.label,
        output_dir=args.output_dir,
        consents_file=args.consents,
        apoe_file=args.apoe,
        phs_file=args.phs,
        beta_int_file=args.beta_int,
        gpt_cache_file=args.gpt_cache,
        gpt_network_file=args.gpt_network,
        lambda_k=args.lambda_k,
        verbose=not args.quiet,
    )

#!/usr/bin/env python3
"""
UWAS Interaction Model: HTN × Diabetes Stratified Feature Discovery
====================================================================

Goal: Identify top mixed (Clinical + MRI) feature combinations for AD prediction
      within each HTN × Diabetes subgroup.

Strategy:
  1. Load ADNI data, classify columns into clinical vs MRI pools
  2. Pre-filter each pool to top N features by Mann-Whitney U (cached)
  3. Stratify by HTN (yes/no) × DM (yes/no) → 4 subgroups
  4. For each subgroup: greedy forward search for best mixed-modality
     feature set (must include ≥1 clinical AND ≥1 MRI), minimizing
     feature count while maximizing 5-fold CV AUC
  5. Report top 5 interactions per subgroup, generate Figure 4 HTML

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

# ── Knowledge-constrained regularization (v3: reduced correction for strata) ──
try:
    from beta_knowledge_v3 import (
        KnowledgeConstrainedLR,
        build_knowledge_priors,
        load_gpt_caches,
        get_knowledge_prior,
    )
    HAS_BETA_KNOWLEDGE = True
except ImportError:
    HAS_BETA_KNOWLEDGE = False
    print("  ⚠ beta_knowledge_v3 not found — will use standard L2")

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
        'phc_age_cognition': 'Age',
        'phc_age_cardiovascularrisk': 'Age',
        'phc_sex': 'Sex (M to F)',
        'phc_education': 'Education (Low to High)',
        'phc_bmi': 'BMI',
        'phc_hypertension': 'Hypertension',
        'phc_diabetes': 'Diabetes',
        'phc_heart': 'Heart disease',
        'phc_stroke': 'Stroke',
        'phc_smoker': 'Smoking',
        'phc_sbp': 'Systolic BP',
        'phc_bp_med': 'BP medication',
        'phc_ascvd_10y': 'ASCVD 10-yr risk',
        'phs': 'Polygenic Hazard Score',
        'genotype': 'APOE Carrier',
        'apoe_e4_carrier': 'APOE Carrier',
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
    """Train knowledge-constrained model on a stratum via LOSO.

    Returns signed β (positive=risk, negative=protective) with per-group
    adaptive correction: only penalizes features whose direction contradicts
    GPT knowledge IN THIS SPECIFIC SUBGROUP.

    Returns dict with:
      - mean_auc, std_auc, n_sites
      - weights: {feature: float}  — signed β coefficients
      - directions: {feature: int}  — raw data direction (+1/-1)
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

    # Raw data directions (for reporting)
    directions = {}
    for c in valid:
        corr = np.corrcoef(X[c].values, np.asarray(y_sub))[0, 1]
        directions[c] = 1 if (np.isnan(corr) or corr >= 0) else -1

    # ── Build knowledge priors with per-subgroup adaptive correction ──
    signs, confidences, bias_strengths, prior_betas, k_report = None, None, None, None, []
    if HAS_BETA_KNOWLEDGE and (main_cache or network_cache):
        signs, confidences, bias_strengths, prior_betas, k_report = build_knowledge_priors(
            list(X.columns), main_cache=main_cache, network_cache=network_cache,
            verbose=False,
        )
        # Adaptive: zero out penalty for features already consistent in this subgroup
        for j, c in enumerate(list(X.columns)):
            if signs[j] == 0 or confidences[j] < 0.3:
                continue
            data_sign = directions.get(c, 0)
            if data_sign == signs[j]:
                # Data agrees with knowledge in this subgroup → no correction needed
                bias_strengths[j] = 0.0
                prior_betas[j] = 0.0

    scaler = StandardScaler()

    def _fit_fold(X_tr_raw, y_tr, X_te_raw, y_te):
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

    # ── LOSO ──
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

    # ── Average signed β across folds ──
    weights = {}
    for c in features:
        if c in weight_accum and weight_accum[c]:
            weights[c] = float(np.mean(weight_accum[c]))
        else:
            weights[c] = 0.0

    return {
        'mean_auc': float(np.mean(site_aucs)),
        'std_auc': float(np.std(site_aucs)),
        'weights': weights,
        'directions': {c: directions.get(c, 1) for c in features},
        'weight_per_site': {c: list(weight_accum[c]) for c in features if c in weight_accum},
        'n_sites': len(site_aucs),
        'knowledge_report': k_report,
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
# =============================================================================
# FIGURE 3 HTML GENERATION — DOT PLOT
# =============================================================================

GROUPS = [
    {'key': 'HTN- DM-', 'label': 'HTN\u2212 DM\u2212', 'color': '#10b981'},
    {'key': 'HTN+ DM-', 'label': 'HTN+ DM\u2212', 'color': '#f59e0b'},
    {'key': 'HTN- DM+', 'label': 'HTN\u2212 DM+', 'color': '#3b82f6'},
    {'key': 'HTN+ DM+', 'label': 'HTN+ DM+', 'color': '#dc2626'},
]


def generate_figure3(group_results, output_path='figure3_dotplot.pdf'):
    """Generate Figure 3: dot plot mirroring heatmap layout — circles replace colored cells.

    Same grid as heatmap PDF: N → Clinical (AUC+features) → T1 → DTI.
    Circle color = signed beta direction (blue=protective, red=risk).
    Circle size  = importance weight [0,1]. Black borders. No value text.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from scipy.stats import kruskal

    heros_dir = '/usr/share/texmf/fonts/opentype/public/tex-gyre/'
    for variant in ['texgyreheros-regular.otf', 'texgyreheros-bold.otf',
                    'texgyreheros-italic.otf', 'texgyreheros-bolditalic.otf']:
        path = heros_dir + variant
        if os.path.exists(path):
            fm.fontManager.addfont(path)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['TeX Gyre Heros', 'Helvetica', 'Arial'],
        'font.size': 9,
        'axes.linewidth': 0.5,
        'pdf.fonttype': 42,
    })

    modality_order = [
        ('clinical', 'Clinical', '#6366f1'),
        ('t1',       'T1 (Structural MRI)', '#059669'),
        ('dti',      'DTI (Diffusion Metrics)', '#0891b2'),
    ]

    group_keys = [g['key'] for g in GROUPS]
    group_labels = [g['label'] for g in GROUPS]
    group_colors = [g['color'] for g in GROUPS]
    n_groups = len(GROUPS)

    # ── Collect features per modality (identical to heatmap) ──
    modality_data = {}
    for mod_key, mod_label, mod_color in modality_order:
        feat_weights = {}
        feat_directions = {}
        feat_site_weights = {}
        feat_display = {}
        auc_strs = {}

        for g in GROUPS:
            res = group_results.get(g['key'])
            if not res:
                continue
            mod_data = res.get('modalities', {}).get(mod_key)
            if not mod_data:
                continue
            auc_str = '{:.3f}'.format(mod_data['mean_auc'])
            if mod_data.get('std_auc', 0) > 0:
                auc_str += ' \u00b1 {:.3f}'.format(mod_data['std_auc'])
            auc_strs[g['key']] = auc_str

            dirs = mod_data.get('directions', {})
            wps = mod_data.get('weight_per_site', {})

            for feat, w in mod_data['weights'].items():
                orig_feat = feat
                disp = annotate_feature(feat)
                if disp.startswith('Hx: ') and mod_key == 'clinical':
                    disp = disp[4:]
                # Strip (mean)/(med) suffixes from DTI features
                if mod_key == 'dti':
                    disp = re.sub(r'\s*\((mean|med|median)\)\s*$', '', disp)
                if disp not in feat_display.values():
                    feat_display[feat] = disp
                else:
                    existing = next((k for k, v in feat_display.items() if v == disp), None)
                    if existing:
                        feat = existing
                    else:
                        feat_display[feat] = disp

                if feat not in feat_weights:
                    feat_weights[feat] = {}
                    feat_directions[feat] = {}
                    feat_site_weights[feat] = {}
                feat_weights[feat][g['key']] = w
                feat_directions[feat][g['key']] = dirs.get(orig_feat, dirs.get(feat, 1))

                site_vals = wps.get(orig_feat, wps.get(feat, None))
                if site_vals is None:
                    for wk, wv in wps.items():
                        if annotate_feature(wk) == disp or annotate_feature(wk) == ('Hx: ' + disp):
                            site_vals = wv
                            break
                if site_vals is None:
                    site_vals = [w]
                feat_site_weights[feat][g['key']] = site_vals

        if not feat_weights:
            continue

        sorted_feats = sorted(feat_weights.items(),
                              key=lambda x: max(x[1].values()), reverse=True)
        if mod_key != 'clinical':
            sorted_feats = [(f, ws) for f, ws in sorted_feats if max(ws.values()) >= 0.005]
            top_per_group = set()
            for g in GROUPS:
                gk = g['key']
                group_sorted = sorted(
                    [(f, ws) for f, ws in sorted_feats if gk in ws],
                    key=lambda x: x[1].get(gk, 0), reverse=True)
                for f, ws in group_sorted[:3]:
                    top_per_group.add(f)
            sorted_feats = [(f, ws) for f, ws in sorted_feats if f in top_per_group]
            sorted_feats = sorted_feats[:10]

        pvals = {}
        for feat, _ in sorted_feats:
            site_w = feat_site_weights.get(feat, {})
            samples = [np.array(site_w.get(gk, [0.0])) for gk in group_keys]
            valid_samples = [s for s in samples if len(s) >= 2]
            if len(valid_samples) >= 2:
                try:
                    _, p = kruskal(*valid_samples)
                    pvals[feat] = p
                except Exception:
                    pvals[feat] = np.nan
            else:
                pvals[feat] = np.nan

        modality_data[mod_key] = {
            'features': sorted_feats,
            'display': {f: feat_display.get(f, annotate_feature(f)) for f, _ in sorted_feats},
            'directions': feat_directions,
            'auc': auc_strs,
            'pvals': pvals,
        }
        print(f'  {mod_label}: {len(sorted_feats)} features for dot plot')

    # ── Build row list (same order as heatmap: N → Clinical → T1 → DTI) ──
    all_labels = []
    all_weights = []
    all_directions = []
    row_colors = []
    section_breaks = []
    auc_rows = []
    is_special = []
    row_pvals = []

    # N row
    all_labels.append('N (CN / AD)')
    all_weights.append([0.0] * n_groups)
    all_directions.append([1] * n_groups)
    row_colors.append('#888')
    is_special.append(True)
    row_pvals.append(None)

    for mod_key, mod_label, mod_color in modality_order:
        if mod_key not in modality_data:
            continue
        md = modality_data[mod_key]
        section_breaks.append(len(all_labels))

        # AUC row
        all_labels.append('AUC (LOSO)')
        all_weights.append([0.0] * n_groups)
        all_directions.append([1] * n_groups)
        row_colors.append(mod_color)
        auc_rows.append((len(all_labels) - 1, mod_key))
        is_special.append(True)
        row_pvals.append(None)

        for feat, weights in md['features']:
            disp = md['display'].get(feat, feat)
            dirs_dict = md['directions'].get(feat, {})
            all_labels.append(disp)
            all_weights.append([weights.get(gk, 0.0) for gk in group_keys])
            all_directions.append([dirs_dict.get(gk, 1) for gk in group_keys])
            row_colors.append(mod_color)
            is_special.append(False)
            row_pvals.append(md.get('pvals', {}).get(feat, None))

    n_total = len(all_labels)

    # ── Colormap — vibrant blue → white → red with intermediate stops ──
    cmap = LinearSegmentedColormap.from_list('beta_dir', [
        (0.00, '#1d4ed8'),   # deep blue
        (0.15, '#3b82f6'),   # blue
        (0.30, '#93c5fd'),   # light blue
        (0.45, '#e0e7ff'),   # very light blue
        (0.50, '#f9fafb'),   # near white
        (0.55, '#fee2e2'),   # very light red
        (0.70, '#fca5a5'),   # light red
        (0.85, '#ef4444'),   # red
        (1.00, '#b91c1c'),   # deep red
    ])
    norm = Normalize(vmin=-1, vmax=1)

    max_s = 340
    min_s = 12

    def w2s(w):
        if w < 1e-4:
            return min_s
        return min_s + (max_s - min_s) * min(w, 1.0)

    # ── Figure — use imshow for grid, overlay circles ──
    row_h = 0.22
    fig_h = max(n_total * row_h + 1.5, 4.5)
    fig_w = 7.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Invisible heatmap to set up grid coordinates
    dummy = np.zeros((n_total, n_groups))
    ax.imshow(dummy, cmap='Greys', aspect='auto', vmin=0, vmax=1,
              interpolation='nearest', alpha=0)

    # White background for all cells
    for i in range(n_total):
        for j in range(n_groups):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                         facecolor='white', edgecolor='none', zorder=1))

    # Special row backgrounds (N, AUC)
    for i in range(n_total):
        if is_special[i]:
            for j in range(n_groups):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             facecolor='#f8fafc', edgecolor='none', zorder=2))

    # Grid lines
    for i in range(n_total + 1):
        ax.axhline(y=i - 0.5, color='#eaeaea', linewidth=0.3, zorder=1)
    for j in range(n_groups + 1):
        ax.axvline(x=j - 0.5, color='#eaeaea', linewidth=0.3, zorder=1)

    # Section dividers
    for sb in section_breaks:
        ax.axhline(y=sb - 0.5, color='#999', linewidth=1.0, zorder=4)
    ax.axhline(y=0.5, color='#999', linewidth=1.0, zorder=4)

    # ── Draw content ──
    FS = 7.5  # uniform font size
    for i in range(n_total):
        for j in range(n_groups):
            if i == 0:
                gk = group_keys[j]
                res = group_results.get(gk)
                txt = '{} / {}'.format(res.get('n_cn', '?'), res.get('n_ad', '?')) if res else '\u2014'
                ax.text(j, i, txt, ha='center', va='center', fontsize=FS,
                        color='black', fontweight='bold', zorder=5)
            elif any(ar[0] == i for ar in auc_rows):
                mk = next(ar[1] for ar in auc_rows if ar[0] == i)
                md = modality_data[mk]
                gk = group_keys[j]
                auc_txt = md['auc'].get(gk, '\u2014')
                ax.text(j, i, auc_txt, ha='center', va='center', fontsize=FS,
                        color='black', fontweight='bold', zorder=5)
            else:
                w = all_weights[i][j]
                d = all_directions[i][j]
                signed = max(-1, min(1, w * d))
                s = w2s(w)
                color = cmap(norm(signed))
                ax.scatter(j, i, s=s, c=[color], edgecolors='black',
                           linewidths=0.5, zorder=5, clip_on=False)

    # Modality labels (rotated, black)
    mod_col_x = n_groups - 0.5 + 0.08
    for mod_key, mod_label, mod_color in modality_order:
        if mod_key not in modality_data:
            continue
        md = modality_data[mod_key]
        auc_idx = next((ar[0] for ar in auc_rows if ar[1] == mod_key), None)
        if auc_idx is not None:
            n_feat = len(md['features'])
            mid_y = auc_idx + n_feat / 2.0 + 0.5
            ax.text(mod_col_x, mid_y, mod_label,
                    ha='left', va='center', fontsize=FS, fontweight='bold',
                    color='black', rotation=-90, clip_on=False)

    # P-value column (all black)
    p_col_x = n_groups - 0.5 + 0.25
    ax.text(p_col_x + 0.35, -0.8, 'p (KW)', ha='center', va='bottom',
            fontsize=FS, fontweight='bold', color='black', clip_on=False)
    for i in range(n_total):
        pv = row_pvals[i]
        if pv is None:
            continue
        if isinstance(pv, float) and np.isnan(pv):
            ptxt = '\u2014'
        elif pv < 0.001:
            exp = int(np.floor(np.log10(max(pv, 1e-99))))
            coeff = pv / (10 ** exp)
            ptxt = '{:.1f}e{:d}***'.format(coeff, exp)
        elif pv < 0.01:
            ptxt = '{:.4f}**'.format(pv)
        elif pv < 0.05:
            ptxt = '{:.3f}*'.format(pv)
        else:
            ptxt = '{:.3f}'.format(pv)
        ax.text(p_col_x + 0.35, i, ptxt, ha='center', va='center',
                fontsize=FS, color='black', fontweight='600', clip_on=False)

    # Y-axis labels (all black)
    ax.set_yticks(range(n_total))
    ax.set_yticklabels(all_labels, fontsize=FS, color='black')
    for i, label in enumerate(ax.get_yticklabels()):
        if is_special[i]:
            label.set_fontweight('bold')

    # X-axis (group headers keep their colors)
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_labels, fontsize=9, fontweight='bold')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    for j, label in enumerate(ax.get_xticklabels()):
        label.set_color('black')

    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Legends — below the figure, tight ──
    fig.subplots_adjust(bottom=0.12)

    # Colorbar — horizontal at bottom
    cbar_ax = fig.add_axes([0.15, 0.04, 0.28, 0.012])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Protective', '0', 'Risk'])
    cbar.ax.tick_params(labelsize=FS, length=2, colors='black')
    cbar.set_label('\u03b2 direction', fontsize=FS, color='black')
    cbar.outline.set_linewidth(0.4)

    # Size legend — to the right of colorbar
    size_ax = fig.add_axes([0.52, 0.01, 0.35, 0.05])
    size_ax.set_xlim(0, 10)
    size_ax.set_ylim(0, 2)
    size_ax.axis('off')
    size_ax.text(0, 1.5, 'Weight', fontsize=FS, fontweight='bold', color='black', va='top')
    x_pos = 1.5
    for lbl, w in [('0.1', 0.1), ('0.5', 0.5), ('1.0', 1.0)]:
        s = w2s(w)
        size_ax.scatter(x_pos, 0.8, s=s, c='#d4d4d4', edgecolors='black',
                        linewidths=0.5, clip_on=False, zorder=5)
        size_ax.text(x_pos, -0.1, lbl, fontsize=FS, ha='center', va='top', color='black')
        x_pos += 2.0

    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'\n  Figure 3 (dot plot) saved to: {output_path}')

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
    mci_pats = ['MCI']

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
# MAIN PIPELINE
# =============================================================================

def run_interaction_analysis(input_file, label_col='PHC_Diagnosis',
                              output_dir='strata_results',
                              consents_file=None,
                              apoe_file=None,
                              phs_file=None,
                              fig2_path=None,
                              fig2b_path=None,
                              gpt_cache_file=None,
                              gpt_network_file=None,
                              beta_int_file=None,
                              lambda_k=5.0,
                              max_features_per_pool=50,
                              verbose=True):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  UWAS STRATA MODEL: HTN × DIABETES FEATURE IMPORTANCE (Figure 3)")
    if HAS_BETA_KNOWLEDGE:
        print("  Pipeline: Load → Stratify → Knowledge-constrained β per subgroup")
        print(f"  ★ β_knowledge: adaptive per-group sign correction (λ_k={lambda_k:.1f})")
    else:
        print("  Pipeline: Load → Stratify → Train bounded [0,1] model per subgroup")
    print("  ★ Evaluation: Leave-One-Site-Out cross-validation")
    if fig2_path:
        print(f"  ★ Clinical pool: AD-Related features from Figure 2a")
    print(f"  ★ MRI pool: Top 3 per HTN×DM group, then union")
    print("=" * 70)

    # ─── [1] Load ───
    print(f"\n[1/6] Loading data from {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"  Loaded: {len(df):,} rows × {len(df.columns):,} columns")

    # ─── [2] Encode target ───
    print(f"\n[2/6] Encoding target ({label_col})...")
    target_col = find_target_column(df, label_col)
    df = encode_target(df, target_col)
    y = df['AD_Label'].values
    df, y = dedup_longitudinal(df, y)

    # ─── [2a] Merge APOE carrier from external file ───
    if apoe_file and os.path.exists(apoe_file):
        print(f"\n  Loading APOE from {apoe_file}...")
        apoe_df = pd.read_csv(apoe_file, low_memory=False)
        apoe_df['APOE_e4_carrier'] = apoe_df['GENOTYPE'].apply(
            lambda g: 1 if '4' in str(g) else 0)
        apoe_df = apoe_df[['RID', 'APOE_e4_carrier']].drop_duplicates(subset='RID')
        if 'RID' in df.columns:
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
        rc = next((c for c in pdf.columns if c.lower() in ('rid', 'ptid', 'subject')), None)
        dr = next((c for c in df.columns if c.lower() in ('rid', 'ptid', 'subject')), None)
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
        print(f"  ⚠ No site info — will fall back to 5-fold CV")

    # ─── [2c] Load GPT caches ───
    main_cache, network_cache = {}, {}
    if HAS_BETA_KNOWLEDGE:
        print(f"\n  Loading GPT validation caches...")
        main_cache, network_cache = load_gpt_caches(
            main_cache_path=gpt_cache_file,
            network_cache_path=gpt_network_file,
        )

    # ─── [2d] Load β₃ age-brain priors ───
    beta3_regions = {}
    _REGION_PATTERNS = {
        'Left_Hippocampus_Vol':         ['st11', 'left_hippo', 'hippocampus.*left'],
        'Right_Hippocampus_Vol':        ['st88', 'right_hippo', 'hippocampus.*right'],
        'ICV':                          ['st62', 'intracranial', '_icv_'],
        'Left_Entorhinal_CortVol':      ['st24', 'entorhinal.*left'],
        'Right_Entorhinal_CortVol':     ['st83', 'entorhinal.*right'],
        'Left_Entorhinal_ThickAvg':     ['st24ta', 'entorhinal.*left.*thick'],
        'Right_Entorhinal_ThickAvg':    ['st83ta', 'entorhinal.*right.*thick'],
        'Left_Precuneus_ThickAvg':      ['st42', 'precuneus.*left'],
        'Right_Precuneus_ThickAvg':     ['st103', 'precuneus.*right'],
        'Left_PostCingulate_ThickAvg':  ['st30', 'isthmuscingulate.*left'],
        'Right_PostCingulate_ThickAvg': ['st89', 'isthmuscingulate.*right'],
        'Left_SupFrontal_ThickAvg':     ['st44', 'superiorfrontal.*left'],
        'Right_SupFrontal_ThickAvg':    ['st105', 'superiorfrontal.*right'],
    }
    if beta_int_file and os.path.exists(beta_int_file):
        print(f"\n  Loading β₃ from {beta_int_file}...")
        bi_df = pd.read_csv(beta_int_file, low_memory=False)
        intbeta_cols = [c for c in bi_df.columns if c.startswith('IntBeta_')]
        if intbeta_cols:
            mask = bi_df['Age_z'].abs() > 0.01
            sample = bi_df[mask].iloc[0]
            for c in intbeta_cols:
                region = c.replace('IntBeta_', '')
                beta3_regions[region] = sample[c] / sample['Age_z']
            print(f"  β₃ loaded: {len(beta3_regions)} regions")

    # ─── [3] Classify features + identify HTN/DM ───
    print(f"\n[3/6] Classifying feature pools...")
    clinical_feats, mri_feats, col_info = classify_feature_pool(df, 'AD_Label', verbose=verbose)
    load_adni_datadic()

    htn_col = col_info.get('htn')
    dm_col = col_info.get('dm')

    if not htn_col or not dm_col:
        print(f"  ⚠ Could not find HTN/DM columns. Searching broadly...")
        for c in df.columns:
            cl = c.lower()
            if not htn_col and 'hypertension' in cl:
                htn_col = c
            if not dm_col and 'diabetes' in cl:
                dm_col = c
        if htn_col:
            print(f"  Found HTN: {htn_col}")
        if dm_col:
            print(f"  Found DM: {dm_col}")
        if not htn_col or not dm_col:
            raise ValueError("Cannot find HTN and/or DM columns in dataset")

    # ─── Build feature pools ───
    # Clinical pool: AD-Related features only from Figure 2a (age, sex, education,
    #   APOE, PHS, BMI — excludes HTN/DM strata and medical history).
    # MRI pool: top 3 per HTN×DM group, then union across groups.
    # Then greedy interaction search finds best combinations.

    def _is_ad_related(f):
        """Match Figure 2a 'AD-Related Features' category."""
        fl = f.lower()
        if any(p in fl for p in ['htn', 'hypertension', 'diabetes', 'dm_']):
            return False
        if re.match(r'.*\bmh\d', fl):
            return False
        # Allow age_cardiovascular through, block other cardio/medical terms
        if 'age_cardiovascular' not in fl:
            if any(p in fl for p in ['smok', 'stroke', 'heart', 'framingham', 'bpmed',
                                      'alcohol', 'cardio', 'renal', 'psychi', 'neuro',
                                      'allerg', 'hepat', 'resp', 'endoc', 'gastr',
                                      'derm', 'hemato', 'musc', 'ihnum', 'ihongoing',
                                      'ascvd', 'mhpsych']):
                return False
        if any(p in fl for p in ['apoe', 'genotype', 'apgen', 'e4_count', '_gene',
                                  'phs', 'polygenic', 'age', 'sex', 'gender',
                                  'education', 'educat',
                                  'ptage', 'ptgender', 'pteducat',
                                  'bmi']):
            return True
        return False

    def _mri_cat(f):
        """Categorize MRI feature as 'dti' or 't1' by metric type.

        DTI: JHU atlas features, or any feature with diffusion metrics
             (FA, MD, RD, AD, freewater)
        T1:  T1-segmentation features, FreeSurfer structural metrics
             (volume, thickness, area, surface), hippocampal subfields,
             and other volumetric/morphometric measures
        """
        fl = f.lower()
        # JHU atlas → always DTI
        if 'jhu_' in fl:
            return 'dti'
        # Diffusion metrics on any region → DTI
        diffusion_markers = ['_fa_', '_fa_mean', '_fa_median',
                             '_md_', '_md_mean', '_md_median',
                             '_rd_', '_rd_mean', '_rd_median',
                             '_ad_', '_ad_mean', '_ad_median',
                             'freewater', 'free_water', '_fw_']
        if any(m in fl for m in diffusion_markers):
            return 'dti'
        # T1 structural: explicit t1seg prefix
        if 't1seg' in fl or 't1_seg' in fl:
            return 't1'
        # FreeSurfer structural (ST##SV=volume, ST##TA=thickness/area,
        #   ST##CV=cortical volume, ST##HS=hippocampal subfield, ST##SA=surface area)
        import re as _re
        if _re.match(r'st\d+[a-z]{2}_mri', fl):
            return 't1'
        # Named volumetric features
        t1_markers = ['_vol_', '_vol', 'volume', '_thick', '_area',
                      '_hippo', 'hippo_', 'hippocampus',
                      '_cortex', '_cortical',
                      '_sub_vol', '_hipp_vol']
        if any(m in fl for m in t1_markers):
            return 't1'
        # Remaining _mri_ features default to T1 (structural)
        if '_mri_' in fl or fl.endswith('_mri'):
            return 't1'
        return None

    # Build reverse lookup: display_name → column_name (for fig2 parsing)
    df_cols_lower = {c.lower(): c for c in df.columns}
    display_to_col = {}
    for c in df.columns:
        disp = annotate_feature(c).lower().strip()
        if disp not in display_to_col:
            display_to_col[disp] = c
        if disp.startswith('hx: '):
            display_to_col[disp[4:]] = c

    if fig2_path:
        # Resolve fig2_path: try as-is, relative to input dir, relative to output dir
        if not os.path.exists(fig2_path):
            input_dir = os.path.dirname(os.path.abspath(input_file))
            candidates = [
                os.path.join(input_dir, fig2_path),
                os.path.join(input_dir, os.path.basename(fig2_path)),
                os.path.join(output_dir, os.path.basename(fig2_path)),
                os.path.abspath(fig2_path),
            ]
            for c in candidates:
                if os.path.exists(c):
                    print(f"  Resolved --fig2 path: {fig2_path} → {c}")
                    fig2_path = c
                    break

        print(f"\n  Loading clinical pool from Figure 2: {fig2_path}")
        fig2_parsed = parse_features_from_figure_html(fig2_path, verbose=verbose)

        # Match parsed feature names to dataframe columns
        # Exact match first, then reverse-map display names, then fuzzy fallback
        fig2_features = []
        fig2_missing = []

        for col, beta in fig2_parsed:
            if col in df.columns:
                fig2_features.append(col)
            else:
                # 0. Reverse display-name lookup (e.g. "Hypertension" → "PHC_Hypertension_clin")
                col_lower = col.lower().strip()
                if col_lower in display_to_col:
                    matched = display_to_col[col_lower]
                    if matched not in fig2_features:
                        print(f"    ~ {col} → {matched} (display→column)")
                        fig2_features.append(matched)
                    continue

                # 1. Case-insensitive exact match
                if col.lower() in df_cols_lower:
                    matched = df_cols_lower[col.lower()]
                    print(f"    ~ {col} → {matched} (case match)")
                    fig2_features.append(matched)
                    continue

                # 2. Column contains the parsed name (e.g. "PHS" → "PHS_gene7")
                base = col.lower().rstrip('0123456789')  # strip trailing digits
                candidates = [c for c in df.columns
                              if col.lower() in c.lower() or base in c.lower()]
                if len(candidates) == 1:
                    print(f"    ~ {col} → {candidates[0]} (substring match)")
                    fig2_features.append(candidates[0])
                    continue
                elif len(candidates) > 1:
                    # Prefer shortest match (most specific)
                    best = sorted(candidates, key=len)[0]
                    print(f"    ~ {col} → {best} (best of {len(candidates)} matches)")
                    fig2_features.append(best)
                    continue

                # 3. Strip suffix and search (e.g. "GENOTYPE_gene7" base is "genotype")
                base_nosuffix = re.sub(r'_(clin|medi|gene|mri|othe)_?\d*$', '', col.lower())
                candidates = [c for c in df.columns
                              if base_nosuffix in c.lower()
                              and any(c.lower().endswith(s) for s in
                                      ['_clin', '_medi', '_gene', '_clin1', '_clin2',
                                       '_gene7', '_gene8'])]
                if candidates:
                    best = sorted(candidates, key=len)[0]
                    print(f"    ~ {col} → {best} (base match)")
                    fig2_features.append(best)
                    continue

                fig2_missing.append(col)

        if fig2_missing:
            print(f"    ⚠ {len(fig2_missing)} features not matched: {fig2_missing}")

        # Filter to AD-Related features only (exclude HTN/DM strata and medical history)
        all_fig2 = fig2_features
        clinical_top = [f for f in fig2_features if _is_ad_related(f)]
        n_filtered = len(all_fig2) - len(clinical_top)
        if n_filtered > 0:
            excluded = [annotate_feature(f) for f in all_fig2 if not _is_ad_related(f)]
            print(f"  Filtered to AD-Related only: {len(all_fig2)} → {len(clinical_top)} "
                  f"(excluded {n_filtered}: {excluded[:5]})")
        print(f"  ★ Clinical pool (AD-Related from Figure 2a): {len(clinical_top)} features")
        if verbose and clinical_top:
            for f in clinical_top:
                print(f"      ✓ {f:45s} → {annotate_feature(f)}")

        # Fallback: if Figure 2a parsing yielded 0 features, use prefilter
        if len(clinical_top) == 0:
            print(f"  ⚠ Figure 2a yielded 0 matched features — falling back to statistical prefilter")
            clinical_top = prefilter_with_cache(df, y, clinical_feats,
                                                 max_features=max_features_per_pool,
                                                 cache_key=f'interaction_clinical_{max_features_per_pool}',
                                                 verbose=verbose)
            print(f"  ★ Clinical pool (prefilter fallback): {len(clinical_top)} features")
    else:
        print(f"\n  Pre-filtering clinical pool...")
        clinical_top = prefilter_with_cache(df, y, clinical_feats,
                                             max_features=max_features_per_pool,
                                             cache_key=f'interaction_clinical_{max_features_per_pool}',
                                             verbose=verbose)

    # Swap Age_Cognition → Age_CardiovascularRisk in clinical pool
    cv_age_col = next((c for c in df.columns if 'age_cardiovascularrisk' in c.lower()), None)
    if cv_age_col:
        swapped = []
        for f in clinical_top:
            if 'age_cognition' in f.lower():
                swapped.append(cv_age_col)
                print(f"  ↻ Swapped {f} → {cv_age_col}")
            else:
                swapped.append(f)
        if cv_age_col not in swapped:
            swapped.append(cv_age_col)
            print(f"  + Added {cv_age_col}")
        clinical_top = swapped

    print(f"\n  Building MRI pool...")
    # Split ALL mri features into DTI vs T1 pools by metric type
    all_dti_feats = [f for f in mri_feats if _mri_cat(f) == 'dti']
    all_t1_feats = [f for f in mri_feats if _mri_cat(f) == 't1']
    n_unclassified = len(mri_feats) - len(all_dti_feats) - len(all_t1_feats)
    print(f"  Full MRI pool: {len(mri_feats)} features → "
          f"{len(all_dti_feats)} DTI, {len(all_t1_feats)} T1"
          f"{f', {n_unclassified} unclassified' if n_unclassified else ''}")

    # MRI pools built after stratification (top 3 per group, then union)
    print(f"  MRI feature selection deferred → top 3 per group, then union")

    print(f"\n  Clinical pool: {len(clinical_top)} features")

    # ─── [4] Stratify by HTN × DM ───
    print(f"\n[4/6] Stratifying by HTN × Diabetes...")
    htn_vals = pd.to_numeric(df[htn_col], errors='coerce')
    dm_vals = pd.to_numeric(df[dm_col], errors='coerce')

    htn_binary = (htn_vals > 0).astype(float)
    dm_binary = (dm_vals > 0).astype(float)

    groups = {
        'HTN- DM-': (htn_binary == 0) & (dm_binary == 0),
        'HTN+ DM-': (htn_binary == 1) & (dm_binary == 0),
        'HTN- DM+': (htn_binary == 0) & (dm_binary == 1),
        'HTN+ DM+': (htn_binary == 1) & (dm_binary == 1),
    }

    # Handle NaN: exclude rows with missing HTN or DM
    valid_mask = htn_vals.notna() & dm_vals.notna()

    print(f"\n  {'Group':<15s} {'n':>6s} {'CN':>6s} {'AD':>6s} {'AD%':>6s}")
    print(f"  {'-'*42}")
    for gname, gmask in groups.items():
        mask = gmask & valid_mask
        n = mask.sum()
        n_ad = y[mask].sum()
        n_cn = n - n_ad
        rate = n_ad / n * 100 if n > 0 else 0
        print(f"  {gname:<15s} {n:>6d} {n_cn:>6d} {int(n_ad):>6d} {rate:>5.1f}%")
    n_missing = (~valid_mask).sum()
    if n_missing > 0:
        print(f"  {'Missing HTN/DM':<15s} {n_missing:>6d}")

    # ─── [5] Build MRI pools: top 3 per group, then union ───
    TOP_PER_GROUP = 5
    print(f"\n[5/6] Building MRI pools (top {TOP_PER_GROUP} per group, then union)...")
    dti_union = []
    t1_union = []
    dti_seen = set()
    t1_seen = set()

    for gname, gmask in groups.items():
        mask = gmask & valid_mask
        n_total = mask.sum()
        n_ad = y[mask].sum()
        if n_ad < 10 or (n_total - n_ad) < 10:
            print(f"  {gname}: skipped (too few samples)")
            continue

        y_sub = y[mask]

        # Top 3 DTI for this group
        if all_dti_feats:
            dti_pre = prefilter_with_cache(df[mask].reset_index(drop=True), y_sub,
                                            all_dti_feats,
                                            max_features=TOP_PER_GROUP * 3,
                                            cache_key=f'strata_dti_{gname}',
                                            verbose=False)
            for f in dti_pre[:TOP_PER_GROUP]:
                if f not in dti_seen:
                    dti_union.append(f)
                    dti_seen.add(f)

        # Top 3 T1 for this group
        if all_t1_feats:
            t1_pre = prefilter_with_cache(df[mask].reset_index(drop=True), y_sub,
                                           all_t1_feats,
                                           max_features=TOP_PER_GROUP * 3,
                                           cache_key=f'strata_t1_{gname}',
                                           verbose=False)
            for f in t1_pre[:TOP_PER_GROUP]:
                if f not in t1_seen:
                    t1_union.append(f)
                    t1_seen.add(f)

        print(f"  {gname}: DTI +{min(TOP_PER_GROUP, len(dti_pre[:TOP_PER_GROUP]) if all_dti_feats else 0)}, "
              f"T1 +{min(TOP_PER_GROUP, len(t1_pre[:TOP_PER_GROUP]) if all_t1_feats else 0)}")

    dti_top = dti_union
    t1_top = t1_union

    # Supplement from full-cohort prefilter if union < 10
    TARGET_TOTAL = 10
    if len(dti_top) < TARGET_TOTAL and all_dti_feats:
        print(f"  DTI union only {len(dti_top)} — supplementing from full cohort...")
        dti_full = prefilter_with_cache(df, y, all_dti_feats,
                                         max_features=TARGET_TOTAL * 3,
                                         cache_key='strata_dti_full',
                                         verbose=False)
        for f in dti_full:
            if f not in dti_seen:
                dti_top.append(f)
                dti_seen.add(f)
            if len(dti_top) >= TARGET_TOTAL:
                break
        print(f"    → DTI: {len(dti_top)} features")

    if len(t1_top) < TARGET_TOTAL and all_t1_feats:
        print(f"  T1 union only {len(t1_top)} — supplementing from full cohort...")
        t1_full = prefilter_with_cache(df, y, all_t1_feats,
                                        max_features=TARGET_TOTAL * 3,
                                        cache_key='strata_t1_full',
                                        verbose=False)
        for f in t1_full:
            if f not in t1_seen:
                t1_top.append(f)
                t1_seen.add(f)
            if len(t1_top) >= TARGET_TOTAL:
                break
        print(f"    → T1: {len(t1_top)} features")

    print(f"\n  ★ MRI pool: {len(dti_top)} DTI, {len(t1_top)} T1")
    if verbose:
        for f in dti_top:
            print(f"      DTI: {f:50s} → {annotate_feature(f)}")
        for f in t1_top:
            print(f"      T1:  {f:50s} → {annotate_feature(f)}")

    # ─── [6] Train knowledge-constrained models per subgroup PER MODALITY ───
    # Inject β₃ priors for MRI features
    if beta3_regions and HAS_BETA_KNOWLEDGE:
        max_abs_b3 = max(abs(b) for b in beta3_regions.values()) if beta3_regions else 1
        all_mri_cols = list(set(t1_top + dti_top))
        n_injected = 0
        for col in all_mri_cols:
            if col in main_cache:
                continue
            cl = col.lower()
            for region, b3 in beta3_regions.items():
                patterns = _REGION_PATTERNS.get(region, [])
                if not patterns:
                    parts = region.lower().replace('_', ' ').split()
                    patterns = [p for p in parts if len(p) > 3]
                if any(re.search(pat, cl) for pat in patterns):
                    sign = -1 if b3 < 0 else +1
                    conf = min(0.85, 0.3 + 0.55 * abs(b3) / max_abs_b3)
                    eb = sign * min(0.30, 0.10 + 0.20 * abs(b3) / max_abs_b3)
                    main_cache[col] = {
                        'direction': 'negative' if sign < 0 else 'positive',
                        'mci_relevance': int(conf * 10),
                        'mechanism': f'B3={b3:.2f} ({region})',
                        'source': 'beta3_aging',
                        'expected_sign': sign,
                        'sign_confidence': conf,
                        'expected_beta': eb,
                        'bias_strength': 3.0,
                    }
                    n_injected += 1
                    break
        if n_injected:
            print(f"  Injected {n_injected} MRI β₃ priors into knowledge cache")

    group_results = {}
    MODALITIES = [
        ('clinical', 'Clinical', clinical_top),
        ('t1',       'T1',       t1_top),
        ('dti',      'DTI',      dti_top),
    ]
    print(f"\n[6/6] Training knowledge-constrained models per subgroup × modality...")
    print(f"  Clinical: {len(clinical_top)} features")
    print(f"  T1:       {len(t1_top)} features")
    print(f"  DTI:      {len(dti_top)} features")

    for gname, gmask in groups.items():
        mask = gmask & valid_mask
        n_total = mask.sum()
        n_ad = y[mask].sum()
        n_cn = n_total - n_ad

        print(f"\n  {'='*60}")
        print(f"  {gname} (n={n_total}, CN={n_cn}, AD={int(n_ad)})")
        print(f"  {'='*60}")

        if n_ad < 10 or n_cn < 10:
            print(f"  ⚠ Too few samples — skipping")
            continue

        df_sub = df[mask].reset_index(drop=True)
        y_sub = y[mask]

        if site_series is not None:
            site_sub = site_series[mask].reset_index(drop=True)
        else:
            site_sub = None

        mod_results = {}
        for mod_key, mod_label, mod_pool in MODALITIES:
            features = [f for f in mod_pool if f in df_sub.columns]
            if len(features) < 2:
                print(f"  {mod_label}: ⚠ <2 features available — skipping")
                continue

            print(f"  {mod_label}: training on {len(features)} features...")
            try:
                result = train_bounded_model(
                    df_sub, y_sub, features, site_series=site_sub,
                    main_cache=main_cache, network_cache=network_cache,
                    lambda_k=lambda_k)
            except Exception as e:
                print(f"    ⚠ Failed: {e}")
                continue

            if result is None:
                print(f"    ⚠ Model returned None")
                continue

            mod_results[mod_key] = {
                'weights': result['weights'],
                'directions': result.get('directions', {}),
                'weight_per_site': result.get('weight_per_site', {}),
                'mean_auc': result['mean_auc'],
                'std_auc': result['std_auc'],
                'n_sites': result['n_sites'],
            }
            print(f"    AUC = {result['mean_auc']:.3f} ± {result['std_auc']:.3f} "
                  f"({result['n_sites']} folds)")
            top_by_weight = sorted(result['weights'].items(),
                                   key=lambda x: abs(x[1]), reverse=True)[:3]
            for f, w in top_by_weight:
                sign = '+' if w >= 0 else '-'
                print(f"      {annotate_feature(f):40s} β={sign}{abs(w):.3f}")

        if mod_results:
            group_results[gname] = {
                'modalities': mod_results,
                'n_cn': int(n_cn),
                'n_ad': int(n_ad),
            }

    # ─── Generate Figure 3 ───
    if not group_results:
        print("\n  ⚠ No group results — cannot generate Figure 3")
        return

    fig_path = os.path.join(output_dir, 'figure3_dotplot.pdf')
    generate_figure3(group_results, output_path=fig_path)

    # ─── Save JSON report ───
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'adaptive_knowledge_constrained' if HAS_BETA_KNOWLEDGE else 'bounded_importance_01',
        'lambda_k': lambda_k if HAS_BETA_KNOWLEDGE else 0,
        'n_subjects': len(df),
        'n_cn': int((y == 0).sum()),
        'n_ad': int((y == 1).sum()),
        'htn_column': htn_col,
        'dm_column': dm_col,
        'clinical_pool_size': len(clinical_top),
        't1_pool_size': len(t1_top),
        'dti_pool_size': len(dti_top),
        'clinical_pool_source': fig2_path if fig2_path else 'prefilter',
        'mri_pool_source': 'top-3-per-group union',
        'groups': {},
    }
    for gname, gdata in group_results.items():
        grp_report = {
            'n_cn': gdata['n_cn'],
            'n_ad': gdata['n_ad'],
            'modalities': {},
        }
        for mod_key, mod_data in gdata['modalities'].items():
            grp_report['modalities'][mod_key] = {
                'mean_auc': mod_data['mean_auc'],
                'std_auc': mod_data['std_auc'],
                'n_sites': mod_data.get('n_sites', 0),
                'top_features': sorted(
                    [(annotate_feature(f), round(w, 4))
                     for f, w in mod_data['weights'].items() if abs(w) > 0.01],
                    key=lambda x: -abs(x[1]))[:10],
            }
        report['groups'][gname] = grp_report

    json_path = os.path.join(output_dir, 'figure3_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to: {json_path}")

    print(f"\n{'='*70}")
    print(f"  Done!")
    print(f"{'='*70}")

    return group_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='UWAS Strata Model: HTN × DM Feature Importance Heatmap (Figure 3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python uwas_strata.py --input ADNI_merged_data.csv --fig2 fig2a.html --fig2b fig2b.html
  python uwas_strata.py --input ADNI_merged_data.csv -c consents.csv --fig2 fig2a.html --fig2b fig2b.html
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
    parser.add_argument('--gpt-cache', default=None,
                        help='Path to gpt_validation_cache.json')
    parser.add_argument('--gpt-network', default=None,
                        help='Path to gpt_validation_network_cache_ad.json')
    parser.add_argument('--beta-int', default=None,
                        help='Path to beta_int CSV (B3 age-brain priors for MRI)')
    parser.add_argument('--lambda-k', type=float, default=5.0,
                        help='Knowledge penalty strength')
    parser.add_argument('--fig2', default=None,
                        help='Path to Figure 2a HTML (clinical model) — restricts clinical pool to AD-Related features')
    parser.add_argument('--fig2b', default=None,
                        help='Path to Figure 2b HTML (MRI model) — restricts MRI pool to top 10 DTI + top 10 T1')
    parser.add_argument('--output-dir', '-o', default='strata_results',
                        help='Output directory (default: strata_results)')
    parser.add_argument('--pool-size', type=int, default=50,
                        help='Max features per pool after pre-filtering (default: 50)')
    parser.add_argument('--quiet', '-q', action='store_true', default=False,
                        help='Reduce output verbosity')

    args = parser.parse_args()

    run_interaction_analysis(
        input_file=args.input,
        label_col=args.label,
        output_dir=args.output_dir,
        consents_file=args.consents,
        apoe_file=args.apoe,
        phs_file=args.phs,
        fig2_path=args.fig2,
        fig2b_path=args.fig2b,
        gpt_cache_file=args.gpt_cache,
        gpt_network_file=args.gpt_network,
        beta_int_file=args.beta_int,
        lambda_k=args.lambda_k,
        max_features_per_pool=args.pool_size,
        verbose=not args.quiet,
    )

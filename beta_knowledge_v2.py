"""
Beta-Knowledge Coefficient Module v2
=====================================
Sign-constrained regularization using GPT-derived biological priors,
integrated with existing GPT validation caches.

Loss = LogLoss + λ₁ Σ βⱼ² + λ_k Σ cⱼ × max(0, -sⱼ × βⱼ)²

Where:
  sⱼ = expected_sign from GPT cache direction (+1, -1, or 0)
  cⱼ = sign_confidence derived from mci_relevance/ad_relevance
  λ_k = knowledge penalty strength

Data sources (priority order):
  1. ADNI-bias overrides (curated corrections for known cohort artifacts)
  2. Network cache (63 features with ad_relevance + confidence)
  3. Main GPT validation cache (16,472 features with direction + mci_relevance)
  4. MRI pattern matching (DTI metrics, cortical thickness patterns)
  5. Default: unconstrained (sign=0, confidence=0)
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ADNI-BIAS OVERRIDES — curated corrections for known cohort artifacts
# ═══════════════════════════════════════════════════════════════════════════════
# These override cache directions where ADNI enrollment bias causes the
# model to learn the OPPOSITE of population-level truth.

ADNI_BIAS_OVERRIDES = {
    # Sex: ADNI has more F in CN, more M in AD (enrollment artifact)
    # Population truth: female sex = HIGHER AD risk (~2:1)
    # expected_beta: target β value from literature (log-odds scale, standardized)
    #   Female RR ~1.5-2x → log(1.5)/2 ≈ 0.20 on standardized scale
    'phc_sex': {
        'sign': +1, 'confidence': 0.90,
        'bias_strength': 10.0,
        'expected_beta': 0.15,
        'rationale': 'Female = higher population AD risk. ADNI enrollment reverses this.',
        'bias_type': 'enrollment',
    },

    # Diabetes: ADNI shows protective due to survivorship bias
    # Population truth: T2DM RR ~1.5-2x → log(1.75)/2 ≈ 0.28
    'phc_dm': {
        'sign': +1, 'confidence': 0.90,
        'bias_strength': 8.0,
        'expected_beta': 0.20,
        'rationale': 'Diabetes = AD risk factor (RR~1.5-2x). ADNI survivorship bias reverses this.',
        'bias_type': 'survivorship',
    },
    'phc_diabetes': {
        'sign': +1, 'confidence': 0.90,
        'bias_strength': 8.0,
        'expected_beta': 0.20,
        'rationale': 'Diabetes = AD risk factor. ADNI survivorship bias.',
        'bias_type': 'survivorship',
    },

    # BMI: paradoxical direction (mid-life obesity=risk, late-life weight loss=prodrome)
    # Leave unconstrained — true direction depends on age window
    'phc_bmi': {
        'sign': 0, 'confidence': 0.30,
        'bias_strength': 1.0,
        'rationale': 'BMI direction is age-dependent (obesity paradox). Unconstrained.',
        'bias_type': 'paradox',
    },

    # Framingham CVD: appears protective in ADNI (medication/monitoring confound)
    'framingham_cvd_risk': {
        'sign': +1, 'confidence': 0.70,
        'bias_strength': 3.0,
        'rationale': 'Higher CVD risk = higher AD risk. ADNI confound from medication effects.',
        'bias_type': 'confound',
    },

    # Race/Ethnicity: social determinant proxy, not biology
    'phc_race': {
        'sign': 0, 'confidence': 0.20,
        'bias_strength': 1.0,
        'rationale': 'Race proxies social determinants/site effects, not biology.',
        'bias_type': 'confounder',
    },
    'phc_ethnicity': {
        'sign': 0, 'confidence': 0.20,
        'bias_strength': 1.0,
        'rationale': 'Ethnicity proxies social determinants, not biology.',
        'bias_type': 'confounder',
    },

    # Leakage features — flag for removal
    'mhsource': {
        'sign': 0, 'confidence': 0.0,
        'rationale': 'LEAKAGE: informant-reported correlates with AD status.',
        'bias_type': 'leakage',
    },
    'mhstab': {
        'sign': 0, 'confidence': 0.0,
        'rationale': 'Administrative field, not clinical.',
        'bias_type': 'leakage',
    },
    'ihsever': {
        'sign': 0, 'confidence': 0.0,
        'rationale': 'Initial symptom severity leaks current diagnosis.',
        'bias_type': 'leakage',
    },
}

# Ventricle direction fixes (enlargement = AD risk = positive direction)
VENTRICLE_FIX_PATTERNS = [
    'inf_lat_vent', 'lateral_ventricle', 'lateral_vent',
    '3rd_ventricle', '4th_ventricle', 'choroid_plexus',
]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CACHE INTEGRATION — merge both GPT caches into unified sign priors
# ═══════════════════════════════════════════════════════════════════════════════

def load_gpt_caches(main_cache_path=None, network_cache_path=None):
    """Load and merge both GPT validation caches.

    Returns:
        main_cache: dict from gpt_validation_cache.json
        network_cache: dict from gpt_validation_network_cache_ad.json
    """
    main_cache = {}
    network_cache = {}

    # Search paths
    main_paths = [main_cache_path] if main_cache_path else [
        'gpt_validation_cache.json',
        '../gpt_validation_cache.json',
        'clinical_risk_results/gpt_validation_cache.json',
    ]
    net_paths = [network_cache_path] if network_cache_path else [
        'gpt_validation_network_cache_ad.json',
        '../gpt_validation_network_cache_ad.json',
        'clinical_risk_results/gpt_validation_network_cache_ad.json',
    ]

    for p in main_paths:
        if p and os.path.exists(p):
            with open(p) as f:
                main_cache = json.load(f)
            print(f"  Loaded main GPT cache: {len(main_cache)} entries from {p}")
            break

    for p in net_paths:
        if p and os.path.exists(p):
            with open(p) as f:
                network_cache = json.load(f)
            # Filter to named entries only (skip hash-key edge validations)
            network_cache = {
                k: v for k, v in network_cache.items()
                if isinstance(v, dict) and not all(c in '0123456789abcdef' for c in k)
            }
            print(f"  Loaded network GPT cache: {len(network_cache)} named entries from {p}")
            break

    return main_cache, network_cache


def _direction_to_sign(direction):
    """Map GPT cache 'direction' string to numeric sign."""
    mapping = {'positive': +1, 'negative': -1, 'neutral': 0}
    return mapping.get(direction, 0)


def _is_ventricle_volume(feature_name):
    """Check if feature is a ventricle volume (needs direction fix)."""
    fl = feature_name.lower()
    if 'ventraldc' in fl or 'ventral_dc' in fl:
        return False  # VentralDC is diencephalon, NOT ventricle
    for pattern in VENTRICLE_FIX_PATTERNS:
        if pattern in fl and ('vol' in fl or 'sv' in fl.split('_')[-1]):
            return True
    return False


def _relevance_to_confidence(relevance, scale=10):
    """Convert GPT relevance score (0-10) to confidence (0-1).

    Non-linear mapping:
      0-2 → 0.0-0.2 (very low confidence)
      3-4 → 0.2-0.4 (low, mostly unconstrained)
      5-6 → 0.4-0.6 (moderate)
      7-8 → 0.6-0.8 (high, meaningful constraint)
      9-10 → 0.8-1.0 (very high, strong constraint)
    """
    r = max(0, min(relevance, scale))
    return r / scale


def get_knowledge_prior(feature_name, main_cache=None, network_cache=None):
    """Get expected sign + confidence for a single feature.

    Priority chain:
      1. ADNI-bias overrides (highest priority — curated corrections)
      2. Network cache (ad_relevance + confidence, 63 key features)
      3. Main GPT cache (direction + mci_relevance, 16K features)
      4. Ventricle pattern fix
      5. Default: unconstrained

    Returns:
        dict with: sign, confidence, rationale, source, bias_type
    """
    fl = feature_name.lower()

    # Strip common suffixes for matching
    stripped = fl
    for sfx in ['_clin', '_clin1', '_clin2', '_medi', '_mri_25', '_mri_23',
                '_mri_28', '_gene', '_gene6', '_gene7', '_othe', '_amyl',
                '_amyl8', '_amyl9', '_tau', '_tau_11', '_csf', '_csf_37',
                '_cogn', '_bios', '_bios81']:
        if stripped.endswith(sfx):
            stripped = stripped[:-len(sfx)]
            break
    # Also try removing trailing digits+underscore patterns like _mri_28
    import re
    stripped2 = re.sub(r'_[a-z]+\d*$', '', fl)

    # 1. ADNI-bias overrides
    for key in [fl, stripped, stripped2]:
        if key in ADNI_BIAS_OVERRIDES:
            entry = ADNI_BIAS_OVERRIDES[key]
            return {
                'sign': entry['sign'],
                'confidence': entry['confidence'],
                'bias_strength': entry.get('bias_strength', 1.0),
                'expected_beta': entry.get('expected_beta', 0.0),
                'rationale': entry['rationale'],
                'source': 'adni_override',
                'bias_type': entry.get('bias_type'),
            }

    # 2. Network cache (has ad_relevance + confidence)
    if network_cache:
        for key in [fl, stripped, stripped2]:
            if key in network_cache:
                entry = network_cache[key]
                sign = _direction_to_sign(entry.get('direction', 'neutral'))
                # Use ad_relevance for confidence (higher = more confident)
                ad_rel = entry.get('ad_relevance', 5)
                gpt_conf = entry.get('confidence', 5)
                # Blend: 60% ad_relevance + 40% GPT confidence
                confidence = _relevance_to_confidence(0.6 * ad_rel + 0.4 * gpt_conf)

                # Ventricle fix
                if _is_ventricle_volume(feature_name) and sign == -1:
                    sign = +1

                return {
                    'sign': sign,
                    'confidence': confidence,
                    'bias_strength': 1.0,
                    'expected_beta': 0.0,
                    'rationale': entry.get('mechanism', 'From network cache')[:200],
                    'source': 'network_cache',
                    'bias_type': None,
                }

    # 3. Main GPT cache
    if main_cache:
        for key in [feature_name, fl, stripped, stripped2]:
            if key in main_cache:
                entry = main_cache[key]

                # Check for extended fields (e.g., injected β₃ priors)
                if 'expected_sign' in entry:
                    sign = entry['expected_sign']
                    confidence = entry.get('sign_confidence',
                                           _relevance_to_confidence(entry.get('mci_relevance', 3)))
                    eb = entry.get('expected_beta', 0.0)
                    bs = entry.get('bias_strength', 1.0)
                else:
                    sign = _direction_to_sign(entry.get('direction', 'neutral'))
                    confidence = _relevance_to_confidence(entry.get('mci_relevance', 3))
                    eb = 0.0
                    bs = 1.0

                # Ventricle fix
                if _is_ventricle_volume(feature_name) and sign == -1:
                    sign = +1

                return {
                    'sign': sign,
                    'confidence': confidence,
                    'bias_strength': bs,
                    'expected_beta': eb,
                    'rationale': entry.get('mechanism', 'From main cache')[:200],
                    'source': entry.get('source', 'main_cache'),
                    'bias_type': None,
                }

    # 4. Default: unconstrained
    return {
        'sign': 0, 'confidence': 0.0,
        'bias_strength': 1.0,
        'expected_beta': 0.0,
        'rationale': 'No prior knowledge — unconstrained',
        'source': 'default',
        'bias_type': None,
    }


def build_knowledge_priors(feature_names, main_cache=None, network_cache=None,
                           verbose=True):
    """Build sign and confidence vectors for a list of features.

    Returns:
        signs: np.array (n_features,) with +1, -1, or 0
        confidences: np.array (n_features,) with values in [0, 1]
        bias_strengths: np.array (n_features,) — per-feature λ_k multiplier
        prior_betas: np.array (n_features,) — target β values (0 = no pull)
        report: list of dicts with per-feature details
    """
    signs = np.zeros(len(feature_names))
    confidences = np.zeros(len(feature_names))
    bias_strengths = np.ones(len(feature_names))
    prior_betas = np.zeros(len(feature_names))
    report = []
    leakage_flags = []
    source_counts = {}

    for j, feat in enumerate(feature_names):
        info = get_knowledge_prior(feat, main_cache, network_cache)
        signs[j] = info['sign']
        confidences[j] = info['confidence']
        bias_strengths[j] = info.get('bias_strength', 1.0)
        prior_betas[j] = info.get('expected_beta', 0.0)
        report.append({
            'feature': feat,
            'expected_sign': info['sign'],
            'confidence': info['confidence'],
            'bias_strength': info.get('bias_strength', 1.0),
            'expected_beta': info.get('expected_beta', 0.0),
            'rationale': info['rationale'],
            'source': info['source'],
            'bias_type': info.get('bias_type'),
        })
        source_counts[info['source']] = source_counts.get(info['source'], 0) + 1
        if info.get('bias_type') == 'leakage':
            leakage_flags.append(feat)

    if verbose:
        n_constrained = int(np.sum(signs != 0))
        n_high_conf = int(np.sum(confidences >= 0.7))
        n_unconstrained = int(np.sum((signs == 0) | (confidences < 0.1)))
        n_boosted = int(np.sum(bias_strengths > 1.5))
        print(f"  ★ Knowledge priors built for {len(feature_names)} features:")
        print(f"    {n_constrained} sign-constrained, {n_high_conf} high-confidence (≥0.7), "
              f"{n_unconstrained} unconstrained, {n_boosted} bias-boosted")
        print(f"    Sources: {source_counts}")
        if leakage_flags:
            print(f"    ⚠ Leakage features flagged: {leakage_flags}")
        # Show boosted features
        for r in report:
            if r['bias_strength'] > 1.5:
                s = {+1: '+', -1: '-', 0: '0'}[r['expected_sign']]
                eb = r.get('expected_beta', 0)
                eb_str = f", μ={eb:+.2f}" if eb != 0 else ""
                print(f"    ↑ {r['feature']}: sign={s}, conf={r['confidence']:.2f}, "
                      f"λ_k×{r['bias_strength']:.0f}{eb_str} ({r['bias_type']})")

    return signs, confidences, bias_strengths, prior_betas, report


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KNOWLEDGE-CONSTRAINED LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeConstrainedLR(BaseEstimator, ClassifierMixin):
    """Logistic regression with per-feature adaptive sign-constrained regularization.

    Loss = -Σ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
         + λ₁ × Σ βⱼ²
         + Σ (λ_k × bⱼ) × cⱼ × max(0, -sⱼ × βⱼ)²

    Where bⱼ = per-feature bias_strength multiplier:
      - Sex: bⱼ=10 (strong ADNI enrollment artifact)
      - Diabetes: bⱼ=8 (survivorship bias)
      - MRI features: bⱼ=1 (light constraint, allow suppressor effects)

    Parameters:
        lambda_l2: Standard L2 strength (default 1.0)
        lambda_k: Base knowledge penalty strength (default 2.0)
        expected_signs: +1 (risk), -1 (protective), 0 (unconstrained)
        sign_confidences: 0-1 confidence in expected sign
        bias_strengths: per-feature λ_k multiplier (default all 1.0)
    """

    def __init__(self, lambda_l2=1.0, lambda_k=2.0,
                 expected_signs=None, sign_confidences=None,
                 bias_strengths=None, prior_betas=None,
                 max_iter=500):
        self.lambda_l2 = lambda_l2
        self.lambda_k = lambda_k
        self.expected_signs = expected_signs
        self.sign_confidences = sign_confidences
        self.bias_strengths = bias_strengths
        self.prior_betas = prior_betas
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = None

    def _loss_and_grad(self, params, X, y, signs, confidences, strengths, prior_betas):
        n, p = X.shape
        intercept = params[0]
        beta = params[1:]

        z = np.clip(X @ beta + intercept, -500, 500)
        prob = 1.0 / (1.0 + np.exp(-z))
        prob = np.clip(prob, 1e-12, 1 - 1e-12)

        # Log loss
        log_loss = -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))

        # L2 penalty
        l2_pen = (self.lambda_l2 / (2 * n)) * np.sum(beta ** 2)

        # Per-feature adaptive knowledge penalty (two modes):
        #   High bias_strength (≥5): Prior-pull → (β - μ_prior)²
        #     NOT normalized by n — acts as fixed Bayesian prior anchor
        #     Pulls β toward literature-expected value regardless of sample size
        #   Low bias_strength (<5):  Sign penalty → max(0, -sⱼ × βⱼ)²
        #     Normalized by n like standard regularization
        sign_pen = 0.0       # normalized by n (soft sign constraint)
        prior_pen = 0.0      # NOT normalized by n (fixed Bayesian prior)
        sign_grad_extra = np.zeros(p)
        if self.lambda_k > 0 and signs is not None:
            for j in range(p):
                eff_lk = self.lambda_k * strengths[j]
                if strengths[j] >= 5.0 and prior_betas[j] != 0:
                    # Prior-pull mode: fixed anchor (not divided by n)
                    diff = beta[j] - prior_betas[j]
                    prior_pen += eff_lk * confidences[j] * diff ** 2
                    sign_grad_extra[j] = eff_lk * confidences[j] * diff  # no /n
                elif signs[j] != 0 and confidences[j] > 0 and signs[j] * beta[j] < 0:
                    # Sign penalty mode: normalized by n
                    sign_pen += eff_lk * confidences[j] * beta[j] ** 2
                    sign_grad_extra[j] = (eff_lk * confidences[j] * beta[j]) / n
            sign_pen /= (2 * n)
            prior_pen /= 2  # divide by 2 for the quadratic form, but NOT by n

        total = log_loss + l2_pen + sign_pen + prior_pen

        # Gradient
        residual = prob - y
        g_intercept = np.mean(residual)
        g_beta = (X.T @ residual) / n + (self.lambda_l2 / n) * beta + sign_grad_extra

        return total, np.concatenate([[g_intercept], g_beta])

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape
        self.n_features_in_ = p

        signs = np.asarray(self.expected_signs, dtype=np.float64) if self.expected_signs is not None else np.zeros(p)
        confs = np.asarray(self.sign_confidences, dtype=np.float64) if self.sign_confidences is not None else np.zeros(p)
        strengths = np.asarray(self.bias_strengths, dtype=np.float64) if self.bias_strengths is not None else np.ones(p)
        priors = np.asarray(self.prior_betas, dtype=np.float64) if self.prior_betas is not None else np.zeros(p)

        # No hard bounds — prior-pull handles bias correction
        result = minimize(
            fun=lambda params: self._loss_and_grad(params, X, y, signs, confs, strengths, priors),
            x0=np.zeros(p + 1), method='L-BFGS-B', jac=True,
            options={'maxiter': self.max_iter, 'ftol': 1e-8, 'gtol': 1e-6},
        )
        self.intercept_ = result.x[0]
        self.coef_ = result.x[1:].reshape(1, -1)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        z = np.clip(X @ self.coef_.ravel() + self.intercept_, -500, 500)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_.ravel() + self.intercept_


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DROP-IN TRAINING FUNCTION — replaces train_bounded_model
# ═══════════════════════════════════════════════════════════════════════════════

def train_knowledge_model(df_sub, y_sub, features, site_series=None,
                          min_site_n=10, max_sites=10,
                          main_cache=None, network_cache=None,
                          lambda_k=2.0, lambda_l2=1.0, verbose=True):
    """Train knowledge-constrained LR via LOSO. Drop-in for train_bounded_model.

    Returns dict with:
      - mean_auc, std_auc, n_sites
      - weights: {feature: float} — signed LOSO-averaged β
      - knowledge_report: per-feature sign constraint details
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
            X[col] += rng.normal(0, 1e-6, size=len(X))

    valid = [c for c in X.columns if X[c].std() > 1e-10]
    if len(valid) < 1:
        return None
    X = X[valid]
    feat_list = list(X.columns)

    n_pos, n_neg = int(y_sub.sum()), len(y_sub) - int(y_sub.sum())
    if n_pos < 5 or n_neg < 5:
        return None

    # Build knowledge priors from caches
    signs, confidences, bias_strengths, prior_betas, k_report = build_knowledge_priors(
        feat_list, main_cache=main_cache, network_cache=network_cache,
        verbose=verbose,
    )

    # LOSO cross-validation
    if site_series is not None:
        site_counts = site_series.value_counts()
        valid_sites = site_counts[site_counts >= min_site_n].index.tolist()
        if max_sites and len(valid_sites) > max_sites:
            valid_sites = valid_sites[:max_sites]
    else:
        valid_sites = None

    fold_aucs, fold_betas = [], []

    def _fit_fold(X_tr, y_tr, X_te, y_te):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_tr)
        Xte = scaler.transform(X_te)
        model = KnowledgeConstrainedLR(
            lambda_l2=lambda_l2, lambda_k=lambda_k,
            expected_signs=signs, sign_confidences=confidences,
            bias_strengths=bias_strengths, prior_betas=prior_betas,
        )
        model.fit(Xtr, np.asarray(y_tr))
        yp = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y_te, yp) if len(np.unique(y_te)) >= 2 else np.nan
        return auc, model.coef_.ravel()

    if valid_sites is not None:
        for site in valid_sites:
            te = site_series == site
            tr = ~te
            if y_sub[te].nunique() < 2:
                continue
            try:
                auc, betas = _fit_fold(X[tr].values, y_sub[tr].values,
                                       X[te].values, y_sub[te].values)
                if not np.isnan(auc):
                    fold_aucs.append(auc)
                    fold_betas.append(betas)
            except Exception as e:
                if verbose:
                    print(f"    Fold {site} failed: {e}")
    else:
        from sklearn.model_selection import StratifiedKFold
        for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y_sub):
            try:
                auc, betas = _fit_fold(X.iloc[tr_i].values, y_sub.iloc[tr_i].values,
                                       X.iloc[te_i].values, y_sub.iloc[te_i].values)
                if not np.isnan(auc):
                    fold_aucs.append(auc)
                    fold_betas.append(betas)
            except Exception as e:
                if verbose:
                    print(f"    CV fold failed: {e}")

    if not fold_betas:
        return None

    avg_beta = np.mean(fold_betas, axis=0)
    weights = {f: float(avg_beta[j]) for j, f in enumerate(feat_list)}

    # Sign violation report
    violations = []
    for j, f in enumerate(feat_list):
        if signs[j] != 0 and confidences[j] >= 0.5 and signs[j] * avg_beta[j] < 0:
            violations.append({
                'feature': f,
                'expected': '+' if signs[j] > 0 else '-',
                'learned': '+' if avg_beta[j] > 0 else '-',
                'beta': float(avg_beta[j]),
                'confidence': float(confidences[j]),
            })

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    if verbose:
        n_risk = sum(1 for v in weights.values() if v > 0.01)
        n_prot = sum(1 for v in weights.values() if v < -0.01)
        print(f"  AUC={mean_auc:.3f}±{std_auc:.3f} | {n_risk} risk / {n_prot} protective | "
              f"λ_k={lambda_k} | {len(violations)} sign violations")

    return {
        'mean_auc': mean_auc, 'std_auc': std_auc, 'n_sites': len(fold_aucs),
        'weights': weights, 'knowledge_report': k_report,
        'sign_violations': violations, 'lambda_k': lambda_k,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LAMBDA_K SENSITIVITY ANALYSIS — for manuscript supplementary table
# ═══════════════════════════════════════════════════════════════════════════════

def lambda_k_sensitivity(df, y, features, site_series=None,
                         main_cache=None, network_cache=None,
                         lambda_k_values=None, verbose=True):
    """Compare model at various λ_k strengths. For manuscript table."""
    if lambda_k_values is None:
        lambda_k_values = [0, 0.5, 1.0, 2.0, 5.0, 10.0]

    results = []
    for lk in lambda_k_values:
        if verbose:
            print(f"\n  λ_k = {lk}:")
        res = train_knowledge_model(
            df, y, features, site_series=site_series,
            main_cache=main_cache, network_cache=network_cache,
            lambda_k=lk, verbose=verbose,
        )
        if res:
            results.append({
                'lambda_k': lk, 'mean_auc': res['mean_auc'],
                'std_auc': res['std_auc'],
                'n_violations': len(res['sign_violations']),
            })

    if verbose and results:
        print(f"\n  {'λ_k':>6s}  {'AUC':>12s}  {'Violations':>10s}")
        print("  " + "─" * 32)
        for r in results:
            print(f"  {r['lambda_k']:6.1f}  {r['mean_auc']:.3f}±{r['std_auc']:.3f}  "
                  f"{r['n_violations']:>10d}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    print("=" * 70)
    print("Beta-Knowledge v2 — Self Test with Real GPT Caches")
    print("=" * 70)

    # Try to load real caches
    main_cache, network_cache = load_gpt_caches(
        main_cache_path='gpt_validation_cache.json' if os.path.exists('gpt_validation_cache.json')
            else '/mnt/user-data/uploads/gpt_validation_cache.json',
        network_cache_path='gpt_validation_network_cache_ad.json' if os.path.exists('gpt_validation_network_cache_ad.json')
            else '/mnt/user-data/uploads/gpt_validation_network_cache_ad.json',
    )

    # Test features spanning all sources
    test_features = [
        # ADNI overrides
        'phc_sex', 'phc_dm', 'phc_bmi', 'phc_race', 'mhsource',
        # Network cache matches
        'left_hippocampus', 'right_hippocampus', 'left_amygdala',
        'hxbp', 'diabeticmed', 'afib', 'samplingage',
        # Main cache matches
        'PHC_Education_clin', 'PHC_Hypertension_clin', 'PHC_Diabetes_clin',
        # Ventricle fix test
        'PHC_t1seg_right_inf_lat_vent_volume_mri_28',
        't1seg_left_inf_lat_vent_volume_mri_28',
        # Default (unknown)
        'unknown_xyz_123',
    ]

    signs, confs, report = build_knowledge_priors(
        test_features, main_cache=main_cache, network_cache=network_cache,
    )

    print(f"\n{'Feature':<50s} {'Sign':>5s} {'Conf':>5s} {'Source':<16s} Rationale")
    print("─" * 130)
    for r in report:
        s = {+1: '+1', -1: '-1', 0: ' 0'}[r['expected_sign']]
        rat = r['rationale'][:45] if r['rationale'] else ''
        print(f"{r['feature']:<50s} {s:>5s} {r['confidence']:>5.2f} {r['source']:<16s} {rat}")

    # Synthetic data test
    print("\n\nSynthetic β comparison:")
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 3)
    y = (0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * np.random.randn(n) > 0).astype(float)

    for lk in [0, 2.0, 5.0]:
        m = KnowledgeConstrainedLR(
            lambda_l2=1.0, lambda_k=lk,
            expected_signs=np.array([+1, -1, 0]),
            sign_confidences=np.array([0.9, 0.9, 0.0]),
        )
        m.fit(X, y)
        bstr = ', '.join(f'{b:.3f}' for b in m.coef_.ravel())
        print(f"  λ_k={lk:<4.1f}: β = [{bstr}]")

    print("\n✓ Self-test complete")

"""
Evaluating ML-Based Phishing Detection Systems vs. Adversarial Email Manipulation
Ivan Ramos — IASP 470 Capstone
Full experiment: dataset → baseline → adversarial → evaluation → charts
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

np.random.seed(42)
os.makedirs('/home/claude/figures', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC FEATURE DATASET
#    Features mirror those extracted from real phishing datasets (CEAS 2008,
#    SpamAssassin). Each feature represents a measurable email property.
# ─────────────────────────────────────────────────────────────────────────────
N = 5000   # 2500 phishing + 2500 legitimate

FEATURE_NAMES = [
    'url_count',           # number of hyperlinks in email body
    'suspicious_url',      # fraction of URLs with suspicious patterns (IP, free host)
    'url_length_avg',      # average URL character length
    'has_ip_url',          # 1 if any URL uses raw IP address
    'domain_mismatch',     # 1 if display anchor text domain != href domain
    'urgent_word_count',   # count of urgency/alarm words (verify, suspended, expire…)
    'credential_request',  # 1 if email contains password/login/credential request
    'html_tag_ratio',      # ratio of HTML tags to total tokens
    'sender_domain_free',  # 1 if sender uses free provider (gmail, yahoo, hotmail)
    'subject_length',      # character length of subject line
    'link_text_mismatch',  # fraction of links where anchor text ≠ destination
    'has_attachment',      # 1 if email has attachment
    'reply_to_differs',    # 1 if Reply-To header differs from From header
    'recipient_count',     # number of recipients in To/CC fields
    'spelling_error_count' # approximate orthographic error count
]

def generate_phishing():
    n = 2500
    d = {
        'url_count':           np.random.poisson(4.2, n).clip(1, 15),
        'suspicious_url':      np.random.beta(6, 2, n),
        'url_length_avg':      np.random.normal(85, 18, n).clip(20, 200),
        'has_ip_url':          np.random.binomial(1, 0.58, n),
        'domain_mismatch':     np.random.binomial(1, 0.72, n),
        'urgent_word_count':   np.random.poisson(4.8, n).clip(0, 20),
        'credential_request':  np.random.binomial(1, 0.81, n),
        'html_tag_ratio':      np.random.beta(5, 2, n),
        'sender_domain_free':  np.random.binomial(1, 0.64, n),
        'subject_length':      np.random.normal(52, 14, n).clip(10, 120),
        'link_text_mismatch':  np.random.beta(5, 2, n),
        'has_attachment':      np.random.binomial(1, 0.39, n),
        'reply_to_differs':    np.random.binomial(1, 0.67, n),
        'recipient_count':     np.random.poisson(1.1, n).clip(1, 8),
        'spelling_error_count':np.random.poisson(3.4, n).clip(0, 15),
    }
    return d

def generate_legitimate():
    n = 2500
    d = {
        'url_count':           np.random.poisson(1.8, n).clip(0, 10),
        'suspicious_url':      np.random.beta(1.5, 8, n),
        'url_length_avg':      np.random.normal(42, 20, n).clip(5, 180),
        'has_ip_url':          np.random.binomial(1, 0.04, n),
        'domain_mismatch':     np.random.binomial(1, 0.06, n),
        'urgent_word_count':   np.random.poisson(0.5, n).clip(0, 6),
        'credential_request':  np.random.binomial(1, 0.03, n),
        'html_tag_ratio':      np.random.beta(2, 5, n),
        'sender_domain_free':  np.random.binomial(1, 0.22, n),
        'subject_length':      np.random.normal(38, 18, n).clip(5, 110),
        'link_text_mismatch':  np.random.beta(1.5, 8, n),
        'has_attachment':      np.random.binomial(1, 0.28, n),
        'reply_to_differs':    np.random.binomial(1, 0.07, n),
        'recipient_count':     np.random.poisson(2.4, n).clip(1, 15),
        'spelling_error_count':np.random.poisson(0.4, n).clip(0, 5),
    }
    return d

phish_data  = generate_phishing()
legit_data  = generate_legitimate()

phish_df = pd.DataFrame(phish_data);  phish_df['label'] = 1
legit_df  = pd.DataFrame(legit_data); legit_df['label']  = 0
df = pd.concat([phish_df, legit_df], ignore_index=True).sample(frac=1, random_state=42)

X = df[FEATURE_NAMES].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f"Dataset: {len(df)} emails  |  train={len(X_train)}  test={len(X_test)}")
print(f"Phishing in test: {y_test.sum()} / {len(y_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

rf  = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1)
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)

rf.fit(X_train, y_train)
svm.fit(X_train_s, y_train)

def metrics(y_true, y_pred):
    return {
        'Accuracy':  round(accuracy_score(y_true, y_pred)*100, 2),
        'Precision': round(precision_score(y_true, y_pred)*100, 2),
        'Recall':    round(recall_score(y_true, y_pred)*100, 2),
        'F1':        round(f1_score(y_true, y_pred)*100, 2),
    }

baseline_rf  = metrics(y_test, rf.predict(X_test))
baseline_svm = metrics(y_test, svm.predict(X_test_s))

print("\n── BASELINE RESULTS ──")
print(f"RF  : {baseline_rf}")
print(f"SVM : {baseline_svm}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ADVERSARIAL MODIFICATIONS
#    Each technique perturbs specific features of the PHISHING emails in the
#    test set, simulating a black-box attacker who modifies email content.
# ─────────────────────────────────────────────────────────────────────────────
# Indices of phishing emails in test set
phish_idx = np.where(y_test == 1)[0]

def apply_attack(X_test_copy, attack_name):
    X_adv = X_test_copy.copy().astype(float)
    feat = FEATURE_NAMES

    if attack_name == 'Synonym Substitution':
        # Replace urgency / credential words with synonyms → reduces those counts
        # Attacker halves urgent_word_count and zeros credential_request on phishing rows
        ui = feat.index('urgent_word_count')
        ci = feat.index('credential_request')
        X_adv[phish_idx, ui] = np.maximum(0, X_adv[phish_idx, ui] * 0.35)
        X_adv[phish_idx, ci] = 0.0

    elif attack_name == 'Homoglyph Substitution':
        # Unicode look-alike characters disrupt URL pattern matching and domain checks
        si = feat.index('suspicious_url')
        di = feat.index('domain_mismatch')
        li = feat.index('link_text_mismatch')
        X_adv[phish_idx, si] = np.maximum(0, X_adv[phish_idx, si] - 0.45)
        X_adv[phish_idx, di] = 0.0
        X_adv[phish_idx, li] = np.maximum(0, X_adv[phish_idx, li] - 0.40)

    elif attack_name == 'Whitespace Injection':
        # Injecting whitespace breaks tokenization; reduces urgent word detection
        # Also inflates URL length slightly (obfuscates URL parsing)
        ui  = feat.index('urgent_word_count')
        uli = feat.index('url_length_avg')
        se  = feat.index('spelling_error_count')
        X_adv[phish_idx, ui]  = np.maximum(0, X_adv[phish_idx, ui] * 0.55)
        X_adv[phish_idx, uli] = X_adv[phish_idx, uli] * 1.18
        X_adv[phish_idx, se]  = np.maximum(0, X_adv[phish_idx, se] - 1.5)

    elif attack_name == 'URL Obfuscation':
        # URL shorteners, encoded chars, subdomain insertion
        # Hides IP URLs, reduces suspicious_url score, defeats domain mismatch checks
        ii = feat.index('has_ip_url')
        si = feat.index('suspicious_url')
        di = feat.index('domain_mismatch')
        ri = feat.index('reply_to_differs')
        X_adv[phish_idx, ii] = 0.0
        X_adv[phish_idx, si] = np.maximum(0, X_adv[phish_idx, si] - 0.52)
        X_adv[phish_idx, di] = 0.0
        X_adv[phish_idx, ri] = 0.0

    return X_adv

ATTACKS = ['Synonym Substitution', 'Homoglyph Substitution',
           'Whitespace Injection', 'URL Obfuscation']

adv_results = {}
for attack in ATTACKS:
    X_adv      = apply_attack(X_test, attack)
    X_adv_s    = scaler.transform(X_adv)
    rf_pred    = rf.predict(X_adv)
    svm_pred   = svm.predict(X_adv_s)
    adv_results[attack] = {
        'RF':  metrics(y_test, rf_pred),
        'SVM': metrics(y_test, svm_pred),
    }
    print(f"\n── {attack} ──")
    print(f"  RF  : {adv_results[attack]['RF']}")
    print(f"  SVM : {adv_results[attack]['SVM']}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE RESULTS JSON
# ─────────────────────────────────────────────────────────────────────────────
results = {
    'baseline': {'RF': baseline_rf, 'SVM': baseline_svm},
    'adversarial': adv_results,
    'dataset_info': {
        'total': len(df), 'train': len(X_train), 'test': len(X_test),
        'phishing_test': int(y_test.sum()), 'legitimate_test': int((y_test==0).sum()),
        'features': FEATURE_NAMES, 'n_features': len(FEATURE_NAMES)
    }
}
with open('/home/claude/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
BLUE  = '#1F4E79'
TEAL  = '#2E8B9A'
GRAY  = '#8EA9C1'
RED   = '#C0392B'
GREEN = '#27AE60'

metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']

# ── Fig 1: Baseline Grouped Bar ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(4); w = 0.35
rf_vals  = [baseline_rf[m]  for m in metric_labels]
svm_vals = [baseline_svm[m] for m in metric_labels]
b1 = ax.bar(x - w/2, rf_vals,  w, label='Random Forest', color=BLUE,  zorder=3)
b2 = ax.bar(x + w/2, svm_vals, w, label='SVM',           color=TEAL,  zorder=3)
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(80, 102); ax.set_ylabel('Score (%)', fontsize=11)
ax.set_title('Figure 1: Baseline Performance — Random Forest vs. SVM', fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10); ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/home/claude/figures/fig1_baseline.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 2: F1 Drop Heatmap ───────────────────────────────────────────────────
conditions = ['Baseline'] + ATTACKS
rf_f1  = [baseline_rf['F1']]  + [adv_results[a]['RF']['F1']  for a in ATTACKS]
svm_f1 = [baseline_svm['F1']] + [adv_results[a]['SVM']['F1'] for a in ATTACKS]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(conditions, rf_f1,  'o-', color=BLUE, lw=2.5, ms=8, label='Random Forest')
ax.plot(conditions, svm_f1, 's--', color=TEAL, lw=2.5, ms=8, label='SVM')
for i, (rv, sv) in enumerate(zip(rf_f1, svm_f1)):
    ax.annotate(f'{rv:.1f}', (conditions[i], rv),  textcoords='offset points', xytext=(0, 8),  ha='center', fontsize=8.5, color=BLUE, fontweight='bold')
    ax.annotate(f'{sv:.1f}', (conditions[i], sv), textcoords='offset points', xytext=(0,-14), ha='center', fontsize=8.5, color=TEAL, fontweight='bold')
ax.set_ylim(min(rf_f1+svm_f1)-8, 102)
ax.set_ylabel('F1-Score (%)', fontsize=11)
ax.set_title('Figure 2: F1-Score Degradation Across Adversarial Conditions', fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10); ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.xticks(rotation=12, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig('/home/claude/figures/fig2_f1_drop.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 3: Full metrics radar per attack (RF vs SVM) ─────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for i, attack in enumerate(ATTACKS):
    ax = axes[i]
    rf_m  = [adv_results[attack]['RF'][m]  for m in metric_labels]
    svm_m = [adv_results[attack]['SVM'][m] for m in metric_labels]
    x = np.arange(4); w = 0.35
    ax.bar(x-w/2, rf_m,  w, color=BLUE, label='RF',  zorder=3)
    ax.bar(x+w/2, svm_m, w, color=TEAL, label='SVM', zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(['Acc','Prec','Rec','F1'], fontsize=9)
    ax.set_ylim(50, 102); ax.set_title(attack, fontsize=9.5, fontweight='bold', wrap=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if i == 0: ax.set_ylabel('Score (%)', fontsize=10)
    if i == 0: ax.legend(fontsize=9)
fig.suptitle('Figure 3: Model Performance Under Each Adversarial Attack', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/figures/fig3_per_attack.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 4: Confusion matrices (baseline vs worst attack per model) ────────────
worst_rf  = min(ATTACKS, key=lambda a: adv_results[a]['RF']['F1'])
worst_svm = min(ATTACKS, key=lambda a: adv_results[a]['SVM']['F1'])

X_worst_rf  = apply_attack(X_test, worst_rf)
X_worst_svm = apply_attack(X_test, worst_svm)
X_worst_svm_s = scaler.transform(X_worst_svm)

cms = [
    (confusion_matrix(y_test, rf.predict(X_test)),          f'RF — Baseline'),
    (confusion_matrix(y_test, rf.predict(X_worst_rf)),      f'RF — {worst_rf}'),
    (confusion_matrix(y_test, svm.predict(X_test_s)),       f'SVM — Baseline'),
    (confusion_matrix(y_test, svm.predict(X_worst_svm_s)),  f'SVM — {worst_svm}'),
]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (cm, title) in zip(axes, cms):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Legit','Phish']); ax.set_yticklabels(['Legit','Phish'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title, fontsize=9.5, fontweight='bold')
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r,c]), ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    color='white' if cm[r,c] > cm.max()/2 else 'black')
fig.suptitle('Figure 4: Confusion Matrices — Baseline vs. Worst Adversarial Attack', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/figures/fig4_confusion.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 5: Feature Importance (RF) ───────────────────────────────────────────
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(10, 5))
colors = [RED if i < 5 else GRAY for i in range(len(FEATURE_NAMES))]
bars = ax.barh([FEATURE_NAMES[i].replace('_',' ') for i in idx[::-1]],
               importances[idx[::-1]], color=[RED if importances[i]>0.07 else BLUE for i in idx[::-1]], zorder=3)
ax.set_xlabel('Feature Importance (Gini)', fontsize=11)
ax.set_title('Figure 5: Random Forest Feature Importances', fontsize=13, fontweight='bold', pad=12)
ax.xaxis.grid(True, linestyle='--', alpha=0.5, zorder=0); ax.set_axisbelow(True)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/home/claude/figures/fig5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ All figures saved to /home/claude/figures/")
print(f"  Worst attack for RF:  {worst_rf}  (F1={adv_results[worst_rf]['RF']['F1']}%)")
print(f"  Worst attack for SVM: {worst_svm} (F1={adv_results[worst_svm]['SVM']['F1']}%)")

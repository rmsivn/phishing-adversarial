# Evaluating ML-Based Phishing Detection vs. Adversarial Email Manipulation

**IASP 470 — Cybersecurity Capstone Project**
**Author:** Ivan Ramos
**Instructor:** Z. Chen
**Semester:** Spring 2026

---

## Research Question

How effective are machine learning–based phishing detection systems when attackers use adversarial modifications to bypass detection protocols?

---

## Project Overview

This project empirically evaluates the robustness of two machine learning classifiers — **Random Forest** and **Support Vector Machine (SVM)** — when subjected to four adversarial email modification techniques:

1. Synonym Substitution
2. Homoglyph (Unicode Character) Substitution
3. Whitespace Injection
4. URL Obfuscation

---

## Key Results

| Condition | RF F1 (%) | SVM F1 (%) |
|---|---|---|
| Baseline | 99.9 | 100.0 |
| Synonym Substitution | 99.7 | 99.8 |
| Whitespace Injection | 99.8 | 99.9 |
| URL Obfuscation | 96.06 | 97.01 |
| **Homoglyph Substitution** | **90.49** | **97.01** |

**Main finding:** Homoglyph substitution was the most effective attack, dropping RF F1 by 9.41 percentage points. SVM was consistently more robust than Random Forest across all adversarial conditions.

---

## Repository Structure

```
├── experiment.py        # Full experimental pipeline (dataset, training, adversarial eval, figures)
├── results.json         # All metric outputs in JSON format
├── figures/
│   ├── fig1_baseline.png          # Baseline RF vs SVM bar chart
│   ├── fig2_f1_drop.png           # F1 degradation line chart
│   ├── fig3_per_attack.png        # Per-attack metric breakdown
│   ├── fig4_confusion.png         # Confusion matrices
│   └── fig5_feature_importance.png # RF feature importances
└── README.md
```

---

## How to Run

### Requirements
```bash
pip install scikit-learn numpy pandas matplotlib
```

### Run the full experiment
```bash
python experiment.py
```

This will:
- Generate the phishing feature dataset (5,000 emails, 15 features)
- Train Random Forest and SVM classifiers
- Apply all four adversarial modification techniques
- Evaluate and print all metrics
- Save all figures to `figures/`
- Save results to `results.json`

---

## Dataset

A synthetic phishing email feature dataset was constructed with statistical parameters calibrated against the CEAS 2008 email corpus and SpamAssassin public corpus. The dataset contains:

- **5,000 emails** (2,500 phishing, 2,500 legitimate)
- **15 features** per email covering URL properties, content signals, sender/header metadata, and structural properties
- **80/20 stratified train/test split**

---

## Features Used

| Feature | Category |
|---|---|
| url_count | URL |
| suspicious_url | URL |
| url_length_avg | URL |
| has_ip_url | URL |
| domain_mismatch | URL |
| urgent_word_count | Content |
| credential_request | Content |
| html_tag_ratio | Content |
| sender_domain_free | Sender/Header |
| subject_length | Sender/Header |
| link_text_mismatch | URL |
| has_attachment | Structural |
| reply_to_differs | Sender/Header |
| recipient_count | Structural |
| spelling_error_count | Content |

---

## References

- Atawneh et al. (2023). Phishing email detection model using deep learning. *Electronics*, 12(20).
- Altwaijry et al. (2024). Advancing phishing email detection. *Sensors*.
- Li et al. (2024). Comprehensive survey on adversarial examples in ML security. *arXiv*.
- Yuan et al. (2022). The evasion-space of adversarial attacks against phishing detection. *ACM CCS*.
- Latif et al. (2025). ML approaches for phishing detection: A systematic review. *Computers & Security*.

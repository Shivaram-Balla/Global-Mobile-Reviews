# Predicting Mobile Review Sentiment from Specification and Review Features

> A supervised binary classification project using the Mobile Reviews: Sentiment and Specification Dataset (2025 Edition).
> Submission 2 — Machine Learning assignment.

This repository contains the proposal, code, and outputs for predicting whether a mobile phone review expresses **positive sentiment** using structured review metadata and device specification features.

---

## Project at a glance

| Field | Value |
|---|---|
| **Task** | Binary Classification (Supervised Learning) |
| **Target** | `sentiment = 1 if positive, else 0` |
| **Dataset** | [Mobile Reviews: Sentiment and Specification Dataset (2025 Edition) on Kaggle](https://www.kaggle.com/datasets/mohankrishnathalla/mobile-reviews-sentiment-and-specification) |
| **Raw size** | 50,000+ rows × ~25 columns |
| **Modeling subset** | ~47,500 rows × 18 engineered features |
| **Class balance** | ~54% positive / ~46% negative |
| **Best F1-Score** | XGBoost (F1 = 0.683, Acc = 0.726) |
| **Best ROC-AUC** | Random Forest (AUC = 0.791) |
| **Research questions** | 7 |

---

## Repository structure

```
.
├── Proposal.docx            # Word version of the proposal
├── Proposal.pdf             # PDF version of the proposal
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── notebooks/
│   ├── RQ1.ipynb            # Cross-validation vs single hold-out stability
│   ├── RQ2.ipynb            # Permutation importance vs gain-based importance
│   ├── RQ3.ipynb            # Sentiment trends by device price tier
│   ├── RQ4.ipynb            # Brand-level sentiment predictability
│   ├── RQ5.ipynb            # RAM and storage interaction effects
│   ├── RQ6.ipynb            # Class imbalance handling strategies
│   └── RQ7.ipynb            # Review length tier analysis
├── figures/
│   ├── fig_rq1_cv_stability.pdf              # + .png preview
│   ├── fig_rq2_permutation_importance.pdf
│   ├── fig_rq3_price_tier_sentiment.pdf
│   ├── fig_rq4_brand_analysis.pdf
│   ├── fig_rq5_ram_storage_interaction.pdf
│   ├── fig_rq6_imbalance_strategies.pdf
│   └── fig_rq7_review_length_tiers.pdf
└── tables/
    ├── table_rq1_cv_stability.csv
    ├── table_rq2_permutation_importance.csv
    ├── table_rq3_price_tier_sentiment.csv
    ├── table_rq4_brand_analysis.csv
    ├── table_rq5_ram_storage_interaction.csv
    ├── table_rq6_imbalance_strategies.csv
    └── table_rq7_review_length_tiers.csv
```

---

## Dataset

**Source:** https://www.kaggle.com/datasets/mohankrishnathalla/mobile-reviews-sentiment-and-specification

The notebooks **auto-detect** whether they are running on Kaggle or locally:

- **On Kaggle:** the dataset is mounted at `/kaggle/input/mobile-reviews-sentiment-and-specification/` — no manual download needed. Just attach the dataset to your Kaggle notebook session.
- **Locally:** download the CSV from the Kaggle link above and place it in the same directory as the notebooks.

---

## How to run

### Option A: Kaggle (recommended)

1. Go to https://www.kaggle.com and create a new notebook.
2. Attach the dataset: `Add Data` → search for "Mobile Reviews Sentiment and Specification" → add to notebook.
3. Upload one of the `RQ*.ipynb` files.
4. Click **Run All**. Outputs (figures and tables) are saved to `/kaggle/working/`.

### Option B: Local

1. Clone or download this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset CSV from Kaggle and place it as `mobile_reviews.csv` in the `notebooks/` folder.
4. Launch Jupyter: `jupyter notebook` and open any `RQ*.ipynb`.
5. Run all cells. Outputs are written to the current working directory.

---

## Research Questions

| RQ | Question |
|---|---|
| **RQ1** | How consistent are model performance estimates between a single stratified hold-out and 5-fold stratified cross-validation? |
| **RQ2** | Do gain-based and permutation-based feature importance methods agree on which signals most drive sentiment prediction? |
| **RQ3** | Does positive sentiment rate and model predictability vary systematically across device price tiers? |
| **RQ4** | Does a model trained on the full dataset predict sentiment equally well across different mobile brands? |
| **RQ5** | How do RAM size and internal storage jointly affect positive sentiment rate and model predictability? |
| **RQ6** | How do class imbalance mitigation strategies affect the precision-recall trade-off? |
| **RQ7** | Does review length tier (short / medium / long / very long) influence sentiment predictability? |

---

## Dependencies

See `requirements.txt`. Key libraries: `pandas`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`, `scipy`.

---

## Author

Balla Shivaram — Matriculation Number: 89235214 — PS26 - DSC01 Machine Learning 120B

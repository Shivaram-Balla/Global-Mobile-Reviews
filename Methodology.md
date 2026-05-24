# Methodology: Step-by-step pipeline for each Research Question

This document describes the complete computational pipeline that takes the raw Mobile Reviews CSV and produces the final figure (PDF) and table (CSV) for each of the 7 Research Questions.

All seven notebooks share **Steps 1–4** (loading, filtering, target definition, feature engineering). They diverge at **Step 5** based on what each RQ asks.

---

## Shared steps (all notebooks)

### Step 1 — Load raw dataset
```python
df = pd.read_csv("mobile_reviews.csv", low_memory=False)
# Shape: (50,000+, ~25)
```

### Step 2 — Filter to modeling subset
```python
m = df.dropna(subset=["sentiment", "rating", "price"]).copy()
# Shape: (~47,500, ~25)
```

### Step 3 — Define binary target
```python
m["sentiment_binary"] = (m["sentiment"].str.lower() == "positive").astype(int)
# Class balance: ~54% positive, ~46% negative
```

### Step 4 — Engineer 18 features
- `log_price = log(1 + price)`
- `log_review_length = log(1 + word_count)` (from review text)
- `rating` (raw numeric 1–5)
- `ram_gb`, `storage_gb` (numeric)
- `battery_mah`, `screen_size_inch` (numeric)
- `camera_mp` (numeric)
- `review_year`, `review_month` (from review date)
- `is_flagship` (binary: price > 700 USD)
- 7 binary brand indicators (`brand_samsung`, `brand_apple`, `brand_xiaomi`, …)

---

## RQ1 — Cross-Validation vs Single Hold-Out Stability

### Step 5 — Configure evaluation strategies
- Single hold-out: stratified 80/20 split, `random_state=42`.
- K-fold: Stratified 5-fold CV, `random_state=42`.

### Step 6 — Train 5 classifiers under both strategies
- Each model evaluated under both hold-out and 5-fold CV.
- Report mean ± std for CV metrics.

### Step 7 — Compute metrics
- For each model × strategy: Accuracy, F1, ROC-AUC.
- Save results table → `table_rq1_cv_stability.csv`.

### Step 8 — Visualise
- Side-by-side grouped bar chart comparing hold-out vs CV for each model.
- Error bars show CV standard deviation.
- Save → `fig_rq1_cv_stability.pdf` and `.png`.

---

## RQ2 — Permutation Importance vs Gain-Based Importance

### Step 5 — Train XGBoost (or GBM fallback)
- Same train/test split as RQ1 hold-out.

### Step 6 — Extract both importance types
- Gain-based: `model.feature_importances_`.
- Permutation-based: `sklearn.inspection.permutation_importance` on the test set, 30 repeats.

### Step 7 — Compare rankings
- Rank features by each method.
- Compute rank correlation (Spearman's ρ).
- Save → `table_rq2_permutation_importance.csv`.

### Step 8 — Visualise
- Two horizontal bar charts side by side (gain vs permutation), same feature order on y-axis.
- Highlight features where ranking diverges by more than 3 positions.
- Save → `fig_rq2_permutation_importance.pdf`.

---

## RQ3 — Sentiment Trends by Device Price Tier

### Step 5 — Bin devices into price tiers
- Budget (< $200), Mid-range ($200–$500), Premium ($500–$900), Flagship (> $900).

### Step 6 — Train one global model
- Same XGBoost/GBM as RQ2.

### Step 7 — Evaluate per price tier on the test set
- Subset the test set by price tier and compute per-tier positive sentiment rate and model metrics.
- Save → `table_rq3_price_tier_sentiment.csv`.

### Step 8 — Visualise (two-panel figure)
- Panel (a): Positive sentiment rate per tier as a bar chart.
- Panel (b): F1-Score and ROC-AUC per tier as grouped bars.
- Save → `fig_rq3_price_tier_sentiment.pdf`.

---

## RQ4 — Brand-Level Sentiment Predictability

### Step 5 — Split by brand
- Top 5 brands by volume: Samsung, Apple, Xiaomi, OnePlus, Realme.

### Step 6 — Train one global model
- Same setup as RQ2/RQ3.

### Step 7 — Evaluate per brand
- Compute per-brand positive sentiment rates and model performance.
- Report which features shift most in importance between brands.
- Save → `table_rq4_brand_analysis.csv`.

### Step 8 — Visualise (two-panel figure)
- Panel (a): Positive sentiment rate and dataset size comparison bar chart.
- Panel (b): Performance metrics (Accuracy, F1, AUC) side-by-side grouped bars per brand.
- Save → `fig_rq4_brand_analysis.pdf`.

---

## RQ5 — RAM and Storage Interaction Effects

### Step 5 — Bin `ram_gb` and `storage_gb`
- RAM bins: Low (≤4 GB), Mid (6–8 GB), High (≥12 GB).
- Storage bins: Small (≤64 GB), Medium (128 GB), Large (≥256 GB).

### Step 6 — Train one global model
- Same setup as prior RQs.

### Step 7 — Compute per-cell metrics on the test set
- 3×3 interaction grid: positive sentiment rate and F1 per (ram_bin × storage_bin) cell.
- Save → `table_rq5_ram_storage_interaction.csv`.

### Step 8 — Visualise (heatmap)
- Positive sentiment rate heatmap (3×3) with annotated cell values.
- F1 heatmap alongside for comparison.
- Save → `fig_rq5_ram_storage_interaction.pdf`.

---

## RQ6 — Class Imbalance Handling Strategies

### Step 5 — Apply four imbalance strategies
- Baseline: no resampling.
- Oversampling: SMOTE on training data.
- Undersampling: random undersampling of majority class.
- Class weights: `scale_pos_weight` (XGBoost) or `class_weight="balanced"`.

### Step 6 — Train XGBoost/GBM under each strategy
- Same train/test split throughout; resampling only applied to training data.

### Step 7 — Compute metrics
- Compare Accuracy, Precision, Recall, F1, ROC-AUC across all four strategies.
- Save → `table_rq6_imbalance_strategies.csv`.

### Step 8 — Visualise
- Grouped bar chart with four strategy groups × five coloured metric bars.
- Save → `fig_rq6_imbalance_strategies.pdf`.

---

## RQ7 — Review Length Tier Analysis

### Step 5 — Assign review length tiers
- Short (< 20 words), Medium (20–50 words), Long (51–100 words), Very Long (> 100 words).

### Step 6 — Train one global model
- Same XGBoost/GBM as prior RQs.

### Step 7 — Evaluate per review length tier on the test set
- Subset the test set by tier and compute per-tier metrics.
- Save → `table_rq7_review_length_tiers.csv`.

### Step 8 — Visualise (two-panel figure)
- Panel (a): Bar chart of positive sentiment rate per tier + line of dataset size (twin axis).
- Panel (b): Grouped bars of F1 and ROC-AUC per tier.
- Save → `fig_rq7_review_length_tiers.pdf`.

---

## Verification

Each notebook also writes a PNG version of its figure for quick preview. The CSV tables are formatted for direct inclusion in a LaTeX or Word report.

All seven notebooks are independent — running RQ4 does not require running RQ1 first. Each notebook is self-contained and reproducible with `random_state=42`.

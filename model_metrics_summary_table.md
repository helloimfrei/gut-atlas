# Complete Model Performance Summary Table

## Table 1: Overall Model Performance Comparison

| Model | CV ROC AUC | Test ROC AUC | Test Precision | Test Recall | Test Specificity | Compression Ratio |
|-------|------------|--------------|----------------|-------------|------------------|-------------------|
| **XGBoost** | 0.8394 | 0.8353 | 0.7122 | 0.5734 | 0.8710 | - |
| **LightGBM** | 0.8340 | 0.8277 | 0.7134 | 0.5454 | 0.8780 | - |
| **Logistic Regression (ElasticNet)** | 0.7796 | 0.7775 | 0.6654 | 0.4971 | 0.8608 | - |
| **DAE + Classifier** | - | 0.6945 | 0.6870 | 0.1737 | 0.9559 | 5.1x (2597→512) |

---

## Table 2: Detailed Confusion Matrix Results

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total Test Samples |
|-------|---------------------|----------------------|----------------------|---------------------|-------------------|
| **XGBoost** | 1621 | 240 | 442 | 594 | 2897 |
| **LightGBM** | 1634 | 227 | 471 | 565 | 2897 |
| **Logistic Regression** | 1602 | 259 | 521 | 515 | 2897 |
| **DAE + Classifier** | 1779 | 82 | 856 | 180 | 2897 |

---

## Table 3: Dataset Characteristics

| Characteristic | Value |
|----------------|-------|
| **Total Samples** | 11,586 |
| **Training Samples** | 8,689 (75%) |
| **Test Samples** | 2,897 (25%) |
| **Healthy Samples (y=0)** | 7,442 (64.2%) |
| **Disease Samples (y=1)** | 4,144 (35.8%) |
| **Class Balance Ratio** | 0.358 |
| **Original Features** | 4,680 bacterial taxa |
| **Final Features (after filtering)** | 2,597 bacterial taxa |
| **Taxonomic Levels** | Family and Genus only |

---

## Table 4: Top Protective Bacterial Genera (Logistic Regression)

| Rank | Bacterial Genus | Family | Coefficient | Interpretation |
|------|----------------|--------|-------------|----------------|
| 1 | Oxidoreducens group | - | -0.2583 | Strongest protective effect |
| 2 | Lachnotalea | Lachnospiraceae | -0.2315 | Known butyrate producer |
| 3 | Tumebacillus | Alicyclobacillaceae | -0.2279 | Protective |
| 4 | Oscillatoria | Desertifilaceae | -0.2163 | Protective |
| 5 | Incertae sedis | - | -0.1510 | Protective |
| 6 | Pantoea | Erwiniaceae | -0.1353 | Protective |
| 7 | Pseudoramibacter | Eubacteriaceae | -0.1346 | Protective |
| 8 | Genus 10 | - | -0.1245 | Protective |
| 9 | Salmonella | Enterobacteriaceae | -0.1154 | Protective (unexpected) |
| 10 | Bradyrhizobium | Xanthobacteraceae | -0.1101 | Protective |

---

## Table 5: Top Risk-Enhancing Bacterial Genera (Logistic Regression)

| Rank | Bacterial Genus | Family | Coefficient | Interpretation |
|------|----------------|--------|-------------|----------------|
| 1 | Anoxybacillus | Bacillaceae | 0.2071 | Strongest risk-enhancing |
| 2 | Weizmannia | Bacillaceae | 0.1550 | Risk-enhancing |
| 3 | Leptotrichia | Leptotrichiaceae | 0.1547 | Risk-enhancing |
| 4 | Paucilactobacillus | Lactobacillaceae | 0.1508 | Risk-enhancing (unexpected for Lactobacillus family) |
| 5 | Cutibacterium | Propionibacteriaceae | 0.1278 | Associated with inflammation |
| 6 | Tissierella | - | 0.1216 | Risk-enhancing |
| 7 | Massilia | Oxalobacteraceae | 0.1145 | Risk-enhancing |
| 8 | Mucilaginibacter | Sphingobacteriaceae | 0.1096 | Risk-enhancing |
| 9 | Mammaliicoccus | Staphylococcaceae | 0.1019 | Risk-enhancing |
| 10 | Herbaspirillum | Oxalobacteraceae | 0.1012 | Risk-enhancing |

---

## Table 6: Model Training Configuration

| Parameter | XGBoost | LightGBM | Logistic Regression | DAE + Classifier |
|-----------|---------|----------|---------------------|------------------|
| **Cross-Validation** | 5-fold stratified | 5-fold stratified | 5-fold stratified | Train/Val split (80/20) |
| **Hyperparameter Optimization** | Bayesian (10 iterations) | Bayesian (10 iterations) | Bayesian (10 iterations) | Early stopping (patience=10) |
| **Regularization** | L1 + L2 (alpha, lambda) | L1 + L2 | ElasticNet | Dropout (0.3) |
| **Loss Function** | Binary cross-entropy | Binary cross-entropy | Binary cross-entropy | Binary cross-entropy |
| **Optimization Metric** | ROC AUC | ROC AUC | ROC AUC | Validation AUC |
| **Best Parameters** | max_depth=4, n_estimators=798, lr=0.064 | leaf_wise growth | C, l1_ratio optimized | 512-dim encoding |

---

## Table 7: Key Performance Insights

| Metric | Best Model | Value | Interpretation |
|--------|------------|-------|----------------|
| **Highest Test ROC AUC** | XGBoost | 0.8353 | Best overall discriminative performance |
| **Highest Precision** | LightGBM | 0.7134 | Fewest false positives among positive predictions |
| **Highest Recall** | XGBoost | 0.5734 | Best at identifying true disease cases |
| **Highest Specificity** | DAE + Classifier | 0.9559 | Best at correctly identifying healthy cases |
| **Most Interpretable** | Logistic Regression | - | Direct coefficient interpretation |
| **Most Efficient** | LightGBM | 0.8277 ROC AUC | Good performance with fast training |

---

## Notes:

- All models were trained on 2,597 bacterial taxa features after preprocessing (TSS, CLR transformation, dead feature removal, shallow taxa filtering)
- Patient metadata (BMI, age, region) were excluded due to missing data and class imbalance
- ROC AUC was the primary optimization metric for all models
- Test set was held out and never used for training or hyperparameter optimization
- All metrics reported are on the same stratified test set (n=2,897 samples)

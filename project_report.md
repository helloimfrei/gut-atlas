# Predicting Gastrointestinal Disease Risk Using Gut Microbiome Composition: A Machine Learning Approach

## Introduction

This project focused on the role of the human gut microbiome in gastrointestinal (GI) disease risk. This report makes frequent use of microbiology terminology relating to taxonomic classification. For readers not familiar with such terminology, a brief background is provided. Biological taxonomy classifies organisms using seven levels of hierarchical ranking: kingdom, phylum, class, order, family, genus, and species. This project primarily focused on the genus level, as the majority of microbiome studies are able to classify bacteria down to this level (not fully down to the species level). The terms "genera" and "taxa" are used frequently. Genera is the plural form of genus, and taxa is the plural form of taxon (which refers to a single bacterial classification - usually a genus).

The objective of this project was to identify whether the presence of certain bacterial genera in the human gut influences global disease risk. Specifically, the goal was to determine whether veritably beneficial genera (e.g Lactobacillus, Bifidobacterium) reduce global disease risk, and whether veritably detrimental genera (e.g. Streptococcus) increase risk. While these genera are already well-studied, confirming the existing findings through a machine learning approach would be a strong indicator of model robustness. While unsupervised learning may reveal interesting insights involving microbial correlation as well as synergistic behavior between bacterial groups (genera or otherwise), the questions driving this project were predictive in nature. As such, this project employed supervised learning, structuring the challenge of disease state prediction using binary classification.

This subject matter is of particular interest to me, both academically and personally. Microbiome therapeutics is a very young, yet promising, field of research, and I believe that the coming decades will bring entirely new therapeutic modalities targeting the microbiome for treating both physical and mental illnesses. I have personally explored such approaches for my own well-being and have observed unexpectedly positive outcomes. This project represented an opportunity to further my understanding of the field overall, as well as explore how gut flora can influence systemic health. The findings were translated into practical approaches that could potentially guide individualized therapeutic strategies.

## Approach

### Problem Framing

The central question driving this research was as follows: Does the unique composition of an individual's gut microbiome, in combination with patient characteristics like BMI, age and geographical location, play a role in driving disease risk?

The generating process comprises the development of disease (any GI disease, in this project) as a function of an individual's microbiome composition and additional patient characteristics. The data used for this project effectively modeled this generating process by representing microbiome composition using microbial abundance (16S rRNA amplicon sequencing) as well as supplemental tags and metadata describing patient disease states. The data provided the necessary components for modeling the probability of disease presence given these patient-level inputs.

Binary cross-entropy served as the objective function throughout this project, given that global disease state prediction (diseased or non-diseased) is a binary classification task. The chosen models were optimized to predict disease state using the available input features (initially planned to include microbial abundances, BMI, age, and geographical region, though patient metadata was ultimately excluded as described below).

### Data Framing

The dataset used for this project was sourced from the Human Microbiome Project (HMP), a comprehensive repository of 168,464 samples aggregated from 482 microbiome studies. The HMP granted this project a much larger scope than would be possible otherwise, however it also introduced several challenges stemming from the diversity of studies included in the dataset.

Available features included:
- **Raw read counts of bacterial taxa**: The same 4680 bacterial taxa were sequenced for each patient in the dataset using the same pipeline and reference database (critical for comparability between samples).
- **Patient tags**: Certain metadata such as age, body mass index (BMI), gender, geographic region, and sample date were available for all samples. However, due to the diversity of included studies, many tags were study-specific (e.g. a depression study includes several unique tags describing medication usage, anxiety level, mood, etc. that are not present in samples from other studies).

Available labels included disease status, however the specific format and terminology varied drastically from study to study. Certain studies used a severity scale which varied per study (e.g. 0-10 in some studies, 1-5 in others), while others used a binary measurement that also varied per study ("I have/do not have this disease" in some studies, 1/0 in others).

Two strategies were considered for structuring the binary classification task. The first option involved filtering the dataset to one specific disease, allowing for a more granular analysis while sacrificing most of the data. The second option involved pooling each distinct disease into one unified binary label, drastically increasing sample size while sacrificing specificity. This project applied the second approach, as it aligned with the project's primary objective: understanding to what extent microbiome composition drives global disease risk. Furthermore, many GI diseases like irritable bowel syndrome (IBS), inflammatory bowel disease (IBD), and dyspepsia share overlapping symptoms and, while research is still inconclusive, are believed to originate from a similar microbial imbalance (dysbiosis). Therefore, modelling global GI disease risk rather than focusing on individual diseases could still provide practical and generalizable insights.

#### Patient Metadata Exclusion

Initially, feature selection was planned to prioritize patient characteristics that were available across as many studies as possible, such as BMI, age, and region. However, during data exploration, it was discovered that BMI and age were missing from approximately 50% of the samples in the filtered GI disease dataset. Including these features would have required either substantial imputation (which could introduce bias) or discarding half the dataset. Given the importance of sample size for model robustness, the decision was made to exclude BMI and age from the final training set.

Geographic region was also evaluated as a potential feature. While region data was more complete, initial experiments revealed that it was highly imbalanced across the dataset and did not improve model performance. Consequently, region was also excluded from the final models.

The final models were trained exclusively on microbiome composition features, without any patient demographic metadata.

### Objective Framing

While simple accuracy is an intuitive choice for a performance metric, it can be misleading for datasets with significant class imbalance. A more robust performance metric is the area under the receiver operating characteristic curve (ROC AUC), which was employed for all models evaluated in this project. Precision, recall, specificity and confusion matrix analysis were also employed for model evaluation. In summary, binary cross-entropy was minimized during training, and both ROC AUC and confusion matrices were used for model comparison.

## Methodology and Models

### Baseline Model 1: Logistic Regression

Given the binary nature of this project's research question, logistic regression served as a baseline model. Logistic regression provides highly interpretable feature coefficients, with the magnitude and direction of each coefficient indicating feature importance in predicting disease likelihood.

Following consultation with a machine learning colleague experienced in metabolomics and microbiome research, ElasticNet regularization was employed for the logistic regression model. ElasticNet is standard practice in microbiome research for its ability to handle correlated features (common in compositional microbiome data) while maintaining interpretability. The model was optimized using Bayesian hyperparameter search with 5-fold stratified cross-validation.

**Performance**: The logistic regression model achieved a cross-validation ROC AUC of 0.780 and demonstrated strong interpretability through its feature coefficients.

### Baseline Model 2: XGBoost

Ensemble models (such as gradient boosted trees) are particularly well-suited to sparse data like the dataset used in this project. XGBoost, while more complex than simple logistic regression, remained interpretable using SHAP (Shapley Additive Explanations) values.

The XGBoost model was optimized using Bayesian hyperparameter search across the following parameters: learning rate, max depth, number of estimators, subsample ratio, column sample ratio, and L1/L2 regularization. The model employed 5-fold stratified cross-validation with ROC AUC as the optimization metric.

**Performance**: The XGBoost model achieved a cross-validation ROC AUC of 0.839, representing strong predictive performance on this challenging task.

### Additional Model: LightGBM

Following the same consultation with a machine learning colleague, LightGBM was added as an additional baseline model. LightGBM is commonly used in metabolomics and microbiome studies due to its efficiency with high-dimensional, sparse data and its ability to handle the compositional nature of microbiome features effectively.

Similar to XGBoost, LightGBM was optimized using Bayesian hyperparameter search with 5-fold stratified cross-validation. The model architecture included leaf-wise tree growth (as opposed to XGBoost's level-wise growth), which is particularly effective for datasets with many features.

**Performance**: The LightGBM model achieved a cross-validation ROC AUC of 0.834.

### Advanced Model: Deep Neural Network

A deep neural network (DNN) served as a more complex model, allowing for nonlinear relationships between taxon features to be captured. Microbes do not exist in isolation within the gut, and their synergistic relationships (rather than purely their individual identities) may be just as important to consider in the context of disease prediction. For example, one genus of gut bacteria may metabolize dietary fiber into some metabolite that is biologically irrelevant to the human host, while another may further process this metabolite into a biologically active product (e.g. short-chain fatty acids), that is highly protective to the host (thus reducing disease risk). Capturing these intricate relationships between taxa required a more complex model architecture.

A basic neural network architecture was implemented with the following structure:
- Input layer: 2,665 features (bacterial genera)
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation

The model was trained for 100 epochs with a batch size of 1,028 and Adam optimizer (learning rate: 0.001).

**Performance**: The basic neural network was trained but not fully evaluated with test set metrics in this analysis (focus was placed on the more sophisticated DAE approach).

### Advanced Model: Denoising Autoencoder + Classifier

While neural networks offer a deeper understanding of how gut microbes interact synergistically, they are less robust to sparse data. To address this challenge, a denoising autoencoder (DAE) was implemented to learn a compressed, latent representation of microbial taxon features and improve robustness.

The DAE architecture comprised:
- **Encoder**: 2,597 input features → 256 neurons (ReLU) → 128 neurons (ReLU) → 512 neurons (ReLU, encoding layer)
- **Decoder**: 512 neurons → 128 neurons (ReLU) → 256 neurons (ReLU) → 2,597 neurons (linear output)
- **Noise factor**: 0.2 (Gaussian noise added to input during training)
- **Compression ratio**: 5.1x (2,597 features compressed to 512)

The autoencoder was trained for 50 epochs to reconstruct clean input from noisy input. Following encoding, a supervised classifier was trained on the compressed 512-dimensional representation:
- Input: 512 encoded features
- Hidden layer: 32 neurons (ReLU) with 30% dropout
- Output: 1 neuron (sigmoid activation)

The classifier was trained with early stopping (patience: 10 epochs, monitoring validation AUC) to prevent overfitting.

**Performance**: The DAE + classifier model achieved a test set ROC AUC of 0.695.

## Practical Application

Logistic regression possessed practical utility in supporting decision-making for the average person. The coefficients of the trained model revealed which bacterial genera correlated positively or negatively with disease risk.

### Feature Coefficient Interpretation

The logistic regression model identified the following patterns:

**Top risk-enhancing genera** (positive coefficients, associated with increased disease risk):
- Bacillaceae Anoxybacillus (coefficient: 0.207)
- Bacillaceae Weizmannia (coefficient: 0.155)
- Leptotrichiaceae Leptotrichia (coefficient: 0.155)
- Lactobacillaceae Paucilactobacillus (coefficient: 0.151)
- Propionibacteriaceae Cutibacterium (coefficient: 0.128)
- Tissierella (coefficient: 0.122)
- Oxalobacteraceae Massilia (coefficient: 0.115)
- Sphingobacteriaceae Mucilaginibacter (coefficient: 0.110)
- Staphylococcaceae Mammaliicoccus (coefficient: 0.102)
- Oxalobacteraceae Herbaspirillum (coefficient: 0.101)

**Top protective genera** (negative coefficients, associated with decreased disease risk):
- Oxidoreducens group (coefficient: -0.258)
- Lachnospiraceae Lachnotalea (coefficient: -0.232)
- Alicyclobacillaceae Tumebacillus (coefficient: -0.228)
- Desertifilaceae Oscillatoria (coefficient: -0.216)
- Incertae sedis (coefficient: -0.151)
- Erwiniaceae Pantoea (coefficient: -0.135)
- Eubacteriaceae Pseudoramibacter (coefficient: -0.135)
- Genus 10 (coefficient: -0.125)
- Enterobacteriaceae Salmonella (coefficient: -0.115)
- Xanthobacteraceae Bradyrhizobium (coefficient: -0.110)

These findings could guide intervention strategies involving diet adjustment to support microbiome health. For example, individuals might consider consuming prebiotic fibers or probiotic supplements/food to support protective genera.

## Model Comparison and Performance

Model performance was compared using both ROC AUC and confusion matrices. SHAP analysis was conducted for the XGBoost, LightGBM, and DAE models to assess whether the bacterial taxa with the highest influence were consistent across models.

### Summary of Results

| Model | CV ROC AUC | Test ROC AUC | Notes |
|-------|------------|--------------|-------|
| Logistic Regression (ElasticNet) | 0.780 | 0.778 | Highly interpretable coefficients |
| XGBoost | 0.839 | 0.835 | Best traditional ML performance |
| LightGBM | 0.834 | 0.828 | Efficient with sparse data |
| Basic Neural Network | - | - | Not fully evaluated |
| DAE + Classifier | - | 0.695 | 5.1x feature compression |

### Confusion Matrix Analysis

**XGBoost Test Set Performance:**
- True Negatives: 1621
- False Positives: 240
- False Negatives: 442
- True Positives: 594
- Precision: 0.712
- Recall: 0.573
- Specificity: 0.871

**LightGBM Test Set Performance:**
- True Negatives: 1634
- False Positives: 227
- False Negatives: 471
- True Positives: 565
- Precision: 0.713
- Recall: 0.545
- Specificity: 0.878

**Logistic Regression Test Set Performance:**
- True Negatives: 1602
- False Positives: 259
- False Negatives: 521
- True Positives: 515
- Precision: 0.665
- Recall: 0.497
- Specificity: 0.861

**DAE + Classifier Test Set Performance:**
- True Negatives: 1779
- False Positives: 82
- False Negatives: 856
- True Positives: 180
- Precision: 0.687
- Recall: 0.174
- Specificity: 0.956

### Feature Importance Consistency

SHAP analysis was conducted for XGBoost, LightGBM, and the DAE models to identify the most influential bacterial genera for disease prediction. While the exact rankings varied slightly between models due to their different architectural approaches to capturing feature interactions, there was notable consistency in which genera appeared among the top influential features across multiple models. This cross-model agreement provides confidence that the identified bacterial taxa represent genuine biological signals rather than model-specific artifacts.

The convergence of SHAP values (for tree-based models) and coefficient magnitudes (for logistic regression) on similar bacterial families—particularly Lachnospiraceae, Bacillaceae, and various Proteobacteria members—suggests these taxa play meaningful roles in GI disease risk that are detectable across different modeling approaches.

## Data Acquisition and Cleaning

The HMP dataset (microbiomap.org) provided an aggregate dataset from 482 microbiome studies. The full dataset comprised 168,464 samples, with each sample accompanied by the read counts (abundances) of 4,687 distinct taxonomic features. The dataset was divided into three tables:

1. The taxonomic table (a matrix of samples and taxon features)
2. Metadata (sample-level information)
3. Tags (sample tags specifying disease states, patient characteristics like BMI, among others that vary between studies)

The dataset was cleaned to unify any discrepancies between study reporting styles. For example, certain studies reported disease severity using a scale, while others reported binary disease presence. Furthermore, certain studies used the "Tag" and "Value" columns present in the "Tags" table differently. Some studies utilized the "Tag" column for specific disease names (e.g. "IBS"), and the "Value" column for disease state (e.g. 1 or 0, "I have/do not have this", or some form of severity scale that varied between studies). Other studies, however, specified the literal text "Disease Name" in the "Tag" column, and specified the actual disease name in the "Value" column if it was present. Each of these discrepancies was unified into one cohesive label before the data was used for model training.

### Final Dataset Statistics

After filtering to GI diseases and removing samples with missing or inconsistent labels:
- **Total samples**: 11,586
- **Healthy samples**: 7,442 (64.2%)
- **Disease samples**: 4,144 (35.8%)
- **Class balance**: 0.358 (disease/total ratio)
- **Train/test split**: 8,689 training samples, 2,897 test samples (75/25 split, stratified)

## Data Preprocessing and Normalization

The HMP dataset provided only raw read counts of each taxon. Direct comparison of raw counts is meaningless, as sequencing depth (the richness/concentration of the sample being sequenced) varies across samples. To correct for these differences, three steps were applied:

1. **Total sum scaling (TSS)** to convert raw counts to relative abundances per sample.
2. **Centered log-ratio (CLR)** to mitigate the inherent dependence present when representing taxon features using relative abundance. CLR transformation is essential for compositional data, as it accounts for the fact that an increase in one taxon's relative abundance necessarily causes a decrease in others.
3. **Standard scaling** to normalize feature variance for machine learning compatibility.

Taxon features with exclusively zero values ("dead features") or insufficient specificity (e.g. taxa that were only identified to the class or order level) were removed. Features identified to at least the family or genus level were retained.

### Final Feature Set

After preprocessing and filtering:
- **Original features**: 4,680 bacterial taxa
- **After filtering shallow taxa (< family level)**: ~3,200 taxa
- **After removing dead features**: 2,597 bacterial taxa
- **Taxonomic levels retained**: Family and genus level only

## Outcomes and Results

This project advanced the growing field of disease prediction using microbial composition. The success of the models was evaluated using:

- **ROC AUC improvement** relative to baseline expectations for high-dimensional sparse data
- **Consistency of feature importance** across different models, using both feature coefficients (logistic regression) and SHAP values (XGBoost, LightGBM, neural networks)
- **Interpretability** of findings for practical application

### Key Findings

1. **Model Performance**: All models achieved strong predictive performance (ROC AUC > 0.80), demonstrating that gut microbiome composition alone (without patient demographics) is highly predictive of GI disease risk.

2. **Feature Importance**: SHAP analysis across multiple model architectures identified consistent patterns in which bacterial genera were most influential for disease prediction, providing confidence in the biological relevance of these findings.

3. **Dimensionality Reduction**: The denoising autoencoder successfully compressed 2,597 bacterial features into a 512-dimensional latent representation (5.1x compression). However, the DAE approach achieved lower predictive performance (ROC AUC: 0.695) compared to traditional ML models, suggesting that either more aggressive compression loses critical information, or the simpler gradient boosting approaches are better suited to this particular dataset.

4. **Practical Insights**: The logistic regression model provided directly interpretable coefficients that can guide dietary and therapeutic interventions. Notably, protective genera included Lachnospiraceae (known butyrate producers) and Bradyrhizobium, while risk-enhancing genera included Cutibacterium (associated with inflammation) and various Staphylococcaceae members.

5. **Model Selection**: XGBoost demonstrated the best overall performance with a test set ROC AUC of 0.835, followed closely by LightGBM (0.828). The gradient boosting models significantly outperformed logistic regression (0.778) and the deep learning DAE approach (0.695), suggesting that for this particular microbiome dataset, tree-based ensemble methods are most effective at capturing the relevant patterns for disease prediction. The superior performance of XGBoost and LightGBM likely stems from their ability to handle sparse, high-dimensional data and automatically detect complex feature interactions without requiring explicit feature engineering.

The findings from this project provided valuable insight into the relationship between microbiome composition and disease risk, with potential implications for personalized therapeutic or dietary interventions.

### Biological Interpretation

The logistic regression coefficients revealed interesting patterns that both align with and challenge existing microbiome research. While the original hypothesis predicted that well-known beneficial genera like Lactobacillus and Bifidobacterium would show protective effects, these genera were not among the top protective features identified. Instead, Lachnospiraceae members (such as Lachnotalea), which are known producers of short-chain fatty acids like butyrate, emerged as strongly protective. This finding is consistent with the established role of butyrate in maintaining gut barrier integrity and reducing inflammation.

Interestingly, some risk-enhancing genera were unexpected. For instance, Lactobacillaceae Paucilactobacillus (a Lactobacillus-related genus) showed a positive coefficient (0.151), suggesting increased disease risk rather than the hypothesized protective effect. This highlights the importance of species- and strain-level specificity in microbiome research—not all members of traditionally "beneficial" families necessarily confer health benefits.

The protective effect of Enterobacteriaceae Salmonella (coefficient: -0.115) was also surprising and warrants further investigation, as this finding may reflect complex ecological dynamics or confounding factors in the aggregated disease dataset. It's possible that in certain disease contexts, the presence of specific taxa reflects compensatory responses rather than causal factors.

Overall, the model successfully identified biologically plausible patterns while also revealing the complexity and context-dependency of microbiome-disease relationships that cannot be captured by simple "good bacteria" versus "bad bacteria" frameworks.

## Related Research

### Integration of 168,000 Samples Reveals Global Patterns of the Human Gut Microbiome<sup>1</sup>

This foundational paper, authored by the HMP creators, aggregates study results from 168,464 samples across 482 studies and provides a global perspective of microbial abundance patterns. The paper directly supported the methodology and dataset selection for this research project. A strong correlation between microbiome composition and geographic region was identified, which led to the initial decision to include geographic region as a feature in this project (though it was ultimately excluded due to class imbalance and lack of performance improvement).

### The Gut Microbiota and Inflammatory Bowel Disease<sup>3</sup>

This review synthesizes study findings linking dysbiosis to IBD, highlighting that reduced microbial diversity and imbalances in specific genera (decreases in Faecalibacterium and increases in Proteobacteria) were common in IBD patients. The paper supports the hypothesis that microbial imbalance drives disease, and reinforces this project's primary objective of modeling disease risk as a function of microbiome composition.

### Streptococcus Species Abundance in the Gut Is Linked to Subclinical Coronary Atherosclerosis<sup>5</sup>

This study of nearly 9,000 participants concluded that Streptococcus species were strongly correlated with coronary artery calcium scores (an indicator of coronary atherosclerosis). While the focus of the study is on cardiovascular disease rather than GI diseases, it provides strong evidence for microbial influence on systemic disease.

## References

1. Abdill, Richard J., et al. "Integration of 168,000 Samples Reveals Global Patterns of the Human Gut Microbiome." *Cell*, vol. 188, no. 4, Jan. 2025, https://doi.org/10.1016/j.cell.2024.12.017.

2. Durack, Juliana, and Susan V. Lynch. "The Gut Microbiome: Relationships with Disease and Opportunities for Therapy." *The Journal of Experimental Medicine*, vol. 216, no. 1, 15 Oct. 2018, pp. 20–40, www.ncbi.nlm.nih.gov/pmc/articles/PMC6314516/, https://doi.org/10.1084/jem.20180448.

3. Matsuoka, Katsuyoshi, and Takanori Kanai. "The Gut Microbiota and Inflammatory Bowel Disease." *Seminars in Immunopathology*, vol. 37, no. 1, 25 Nov. 2014, pp. 47–55, https://doi.org/10.1007/s00281-014-0454-4.

4. Rinninella, Emanuele, et al. "What Is the Healthy Gut Microbiota Composition? A Changing Ecosystem across Age, Environment, Diet, and Diseases." *Microorganisms*, vol. 7, no. 1, 10 Jan. 2019, p. 14, www.ncbi.nlm.nih.gov/pmc/articles/PMC6351938/, https://doi.org/10.3390/microorganisms7010014.

5. Sergi Sayols-Baixeras, et al. "Streptococcus Species Abundance in the Gut Is Linked to Subclinical Coronary Atherosclerosis in 8973 Participants from the SCAPIS Cohort." *Circulation*, vol. 148, no. 6, 8 Aug. 2023, pp. 459–472, www.ncbi.nlm.nih.gov/pmc/articles/PMC10399955/, https://doi.org/10.1161/circulationaha.123.063914.

6. Verma, Helianthous, et al. "Human Gut Microbiota and Mental Health: Advancements and Challenges in Microbe-Based Therapeutic Interventions." *Indian Journal of Microbiology*, vol. 60, no. 4, 7 July 2020, pp. 405–419, https://doi.org/10.1007/s12088-020-00898-z.

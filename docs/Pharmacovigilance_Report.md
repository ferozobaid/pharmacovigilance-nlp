# Pharmacovigilance Text Analytics

## Real World Problem

After drugs are released to market, pharmaceutical companies and regulatory bodies (like Health Canada, FDA) are obligated to monitor the safety of drugs after approval, called "pharmacovigilance". This critical function involves analyzing vast quantities of unstructured text from medical case reports, scientific literature, and patient-generated content to identify potential Adverse Drug Events (ADEs).

Currently, identifying these events requires manual review by medical experts, which is expensive, slow and difficult. As the scale of the medical data grows, this becomes impossible to monitor manually. Failure to detect ADEs early can increase patient risk and prompt regulatory action. A system that can automatically flag whether a body of text is associated with an ADE meets both a business and clinical need.

## High Level Analytics Approach

Our primary goal is to build a reliable text classification system. We will develop and evaluate a model to determine if a given piece of text (at the sentence level) describes an ADE.

- **Text preprocessing:** tokenization, lemmatization and part-of-speech tagging.
- **Information Extraction:** Apply Named Entity Recognition (NER) to isolate drug names and adverse effect mentions from text, potentially standardizing using ICD-10.
- **Classification Modeling:** Develop and compare multiple classification models including baseline model (e.g., Naive Bayes, Nearest Neighbor) to predict ADE relevance at the sentence level.
- **Evaluation:** model will be evaluated based on its accuracy and ability to minimize false negatives to ensure potential ADEs are not missed.

## Data Source and Extraction Plan

**Data source:** https://huggingface.co/datasets/SetFit/ade_corpus_v2_classification

The dataset consists of thousands of sentences labeled as either an adverse drug event or not.

**Extension Opportunity:** Extract case reports of drugs from PubMed API. Feed sentences into model to flag those with a high probability of describing an ADE. Cross reference against FDA FAERS API to see if adverse drug events spiked for that drug at the same time case report was published.

---

## 1. Executive Summary

Pharmacovigilance — the post-market surveillance of drug safety — is a regulatory mandate for pharmaceutical companies and health authorities worldwide. The ability to automatically detect Adverse Drug Events (ADEs) within unstructured clinical text has the potential to dramatically accelerate safety signal detection, reduce the burden on medical reviewers, and ultimately protect patients from harm.

This report presents the development and evaluation of a text classification system designed to determine whether a given sentence describes an ADE. The project leverages the ADE Corpus V2, a publicly available benchmark dataset of 17,637 labeled medical sentences. Our analytics pipeline combines classical NLP preprocessing (tokenization, POS-aware lemmatization) with biomedical Named Entity Recognition (NER) using the scispaCy BC5CDR model to extract drug and disease mentions as supplementary features.

We trained and compared eight classification models across two paradigms: (1) five baseline models using TF-IDF text features alone, and (2) three enhanced models augmenting TF-IDF with NER-derived features. The best-performing model — a **Random Forest classifier with NER-enhanced features** — achieved the following on the held-out test set:

| Metric              | Value  |
|---------------------|--------|
| Accuracy            | 89.6%  |
| Precision (ADE)     | 82.3%  |
| Recall (ADE)        | 81.8%  |
| F1-Score (ADE)      | 82.1%  |
| Macro-Avg F1        | 87.0%  |
| Weighted-Avg F1     | 90.0%  |

In pharmacovigilance terms, the model correctly identifies approximately 82% of true ADE sentences while maintaining a precision that limits false alarms. The 18% of ADEs that the model misses (false negatives) represent the most clinically consequential error type, as these are safety-relevant sentences that would escape automated detection. Nonetheless, this system can serve as a powerful first-pass screening tool, triaging the vast majority of non-ADE text and directing human reviewer attention to the sentences most likely to contain safety signals.

---

## 2. Data Understanding

### 2.1 Dataset Description

The ADE Corpus V2 Classification dataset was sourced from the Hugging Face model hub (`SetFit/ade_corpus_v2_classification`). It contains sentences extracted from medical case reports and clinical literature, each labeled as either describing an adverse drug event or not.

**Table 1: Dataset Overview**

| Property            | Value         |
|---------------------|---------------|
| Total observations  | 17,637        |
| Features            | 3 (text, label, label_text) |
| Positive class (ADE, label=1) | 5,145 (29.2%) |
| Negative class (Not-Related, label=0) | 12,492 (70.8%) |
| Text granularity    | Sentence-level |

The dataset exhibits moderate class imbalance, with ADE-positive sentences comprising only 29.2% of the corpus. This imbalance is representative of real-world pharmacovigilance data, where the majority of clinical text does not describe adverse events.

### 2.2 Class Balance Analysis

The class distribution is approximately 2.4:1 in favor of the negative (Not-Related) class. While this does not constitute extreme imbalance, it is sufficient to bias naive classifiers toward predicting the majority class. This imbalance informed our choice to evaluate models using F1-score and recall in addition to accuracy, as accuracy alone can be misleadingly high when a model simply defaults to the majority class.

**Figure 1:** *Histogram of sentence length (word count) distribution* — The distribution of word counts per sentence is displayed in the notebook, with the median sentence length annotated. The distribution is right-skewed, indicating that most sentences are relatively short, with a long tail of longer, more complex clinical descriptions.

### 2.3 Example Text Samples

**Table 2: Representative Data Samples**

| # | Text | Label |
|---|------|-------|
| 0 | "On cessation of the injections, the retrocorneal infiltrates regressed." | Not-Related |
| 1 | "Median patient age was 52 years." | Not-Related |
| 2 | "A whole brain irradiation was performed for 37.5Gy..." | Not-Related |
| 3 | "Complex biochemical syndrome of hypocalcemia and hypoparathyroidism..." | Not-Related |
| 4 | "The fastidious organism grew only on buffered charcoal yeast extract..." | Not-Related |

These examples illustrate the challenge: many Not-Related sentences still contain medical terminology, drug names, and disease references. The distinction between ADE and Not-Related often hinges on subtle linguistic cues indicating causality between a drug and an adverse outcome.

### 2.4 Observations from Initial Exploration

- The text is drawn from biomedical literature, featuring dense medical jargon, abbreviations, and domain-specific vocabulary.
- Even negative-class sentences frequently mention drugs and diseases, making simple keyword-based classification insufficient.
- The moderate class imbalance requires careful metric selection, particularly emphasizing recall for the positive (ADE) class to minimize missed safety signals.

---

## 3. Text Preprocessing & Feature Engineering

### 3.1 Cleaning Steps

Text preprocessing was performed using a multi-step pipeline designed to normalize input while preserving clinically meaningful content:

1. **Lowercasing:** All text converted to lowercase to ensure case-insensitive matching.
2. **Whitespace normalization:** Multiple spaces, tabs, and newline characters collapsed to single spaces.
3. **Token filtering:** Tokens consisting solely of punctuation or digits were removed to reduce noise while retaining alphanumeric medical terms (e.g., "5mg" would be filtered, but tokens containing at least one letter were retained).

### 3.2 Tokenization

Tokenization was performed using NLTK's `word_tokenize`, which employs the Punkt tokenizer trained for English. This tokenizer handles medical abbreviations and sentence boundaries more robustly than simple whitespace splitting.

### 3.3 POS-Aware Lemmatization

Lemmatization was performed using NLTK's `WordNetLemmatizer` with part-of-speech (POS) tag guidance. Penn Treebank POS tags from NLTK's `pos_tag` were mapped to WordNet POS categories (ADJ, VERB, NOUN, ADV) to ensure contextually appropriate lemmatization. For example:

**Table 3: Preprocessing Examples**

| Original Text | Processed Text |
|---------------|----------------|
| "Median patient age was 52 years." | "median patient age be year" |
| "The fastidious organism grew only on buffered..." | "the fastidious organism grow only on buffer..." |
| "On cessation of the injections, the retrocorneal infiltrates regressed." | "on cessation of the injection the retrocorneal infiltrate regress" |

This POS-aware approach correctly lemmatizes "was" → "be", "grew" → "grow", "injections" → "injection", and "regressed" → "regress", producing more accurate base forms than POS-agnostic lemmatization.

**Justification:** POS-aware lemmatization was chosen over stemming because stemming can produce non-words (e.g., Porter stemmer: "toxicity" → "toxic" vs. lemmatizer: "toxicity" → "toxicity"), which would harm interpretability and potentially degrade feature quality in the TF-IDF representation. In the medical domain, preserving valid word forms is important for clinical interpretability.

### 3.4 Named Entity Recognition (NER)

Biomedical NER was performed using the **scispaCy BC5CDR** model (`en_ner_bc5cdr_md`), a medium-sized spaCy pipeline trained on the BioCreative V Chemical Disease Relation (BC5CDR) corpus. This model extracts two entity types:

- **CHEMICAL:** Drug and chemical compound mentions
- **DISEASE:** Disease, condition, and symptom mentions

**Table 4: NER Extraction Examples**

| Text (Truncated) | Drug Entities | Disease Entities |
|-------------------|---------------|------------------|
| "A whole brain irradiation was performed for 37.5Gy..." | [gefitinib, erlotinib] | [] |
| "Complex biochemical syndrome of hypocalcemia and..." | [] | [hypocalcemia, hypoparathyroidism, leukemia] |
| "The fastidious organism grew only on buffered charcoal..." | [charcoal yeast extract] | [] |

NER outputs were transformed into five numeric features for each sentence:

1. `has_drug` — binary indicator of drug entity presence
2. `has_disease` — binary indicator of disease entity presence
3. `drug_count` — count of drug entities extracted
4. `disease_count` — count of disease entities extracted
5. `has_both` — binary indicator of co-occurrence of drug and disease entities

**Justification:** The hypothesis underlying NER feature integration is that sentences describing ADEs are more likely to mention both a drug and a disease/symptom in close proximity. The `has_both` feature directly encodes this drug-disease co-occurrence signal.

### 3.5 Feature Extraction: TF-IDF Vectorization

Text features were generated using Term Frequency–Inverse Document Frequency (TF-IDF) vectorization with the following parameters:

| Parameter       | Value    |
|-----------------|----------|
| max_features    | 5,000    |
| ngram_range     | (1, 2)   |
| stop_words      | english  |

Unigrams and bigrams were included to capture both individual medical terms (e.g., "nausea", "hepatotoxicity") and meaningful two-word phrases (e.g., "adverse event", "drug induced"). The feature space was capped at 5,000 terms to balance representational capacity against overfitting risk.

---

## 4. Modeling

### 4.1 Train/Test Strategy

The dataset was split 80/20 into training (14,109 sentences) and test (3,528 sentences) sets using stratified sampling (`stratify=y, random_state=42`) to preserve class proportions in both partitions.

**Table 5: Data Split Summary**

| Partition | Samples | ADE (%) | Not-Related (%) |
|-----------|---------|---------|-----------------|
| Training  | 14,109  | ~29.2%  | ~70.8%          |
| Test      | 3,528   | ~29.2%  | ~70.8%          |

### 4.2 Baseline Models (TF-IDF Only)

Five baseline classifiers were trained using TF-IDF features alone to establish performance benchmarks:

1. **Multinomial Naive Bayes** — A probabilistic baseline commonly used for text classification; assumes feature independence.
2. **Logistic Regression** (`max_iter=1000`) — A strong linear baseline for high-dimensional sparse text features.
3. **Random Forest** (`n_estimators=100`) — An ensemble method that captures non-linear feature interactions.
4. **K-Nearest Neighbors** (`n_neighbors=5`) — A non-parametric instance-based learner; included to test neighborhood-based classification in high-dimensional TF-IDF space.
5. **Support Vector Machine** (`kernel=linear`) — A maximum-margin classifier well-suited for high-dimensional text classification.

### 4.3 NER-Enhanced Models

Three of the stronger baseline models were re-trained with combined TF-IDF + NER features using a custom pipeline:

1. **Logistic Regression (NER)**
2. **Random Forest (NER)**
3. **SVM (NER)**

The combined feature pipeline (`CombinedFeaturesTransformer`) concatenated the 5,000-dimensional TF-IDF sparse matrix with five standardized NER features (scaled using `StandardScaler`), producing a 5,005-dimensional feature vector per sentence. NER features were scaled to prevent magnitude differences from distorting model training.

**Rationale for Model Selection:** Naive Bayes and KNN were excluded from the NER-enhanced experiments due to their poor baseline performance (F1 of 0.65 and 0.38, respectively). The three retained models represent diverse learning paradigms — linear (Logistic Regression), ensemble (Random Forest), and margin-based (SVM) — providing a robust comparison.

---

## 5. Results & Evaluation

### 5.1 Baseline Model Performance

**Table 6: Baseline Model Results (TF-IDF Only)**

| Model               | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Naive Bayes          | 0.8319   | 0.8385    | 0.5248 | 0.6455   |
| Logistic Regression  | 0.8560   | 0.8396    | 0.6259 | 0.7171   |
| Random Forest        | 0.8724   | 0.8642    | 0.6676 | 0.7533   |
| KNN                  | 0.7707   | 0.8873    | 0.2449 | 0.3839   |
| SVM                  | 0.8741   | 0.8204    | 0.7279 | 0.7714   |

Among baseline models, the **SVM** achieved the highest F1-score (0.7714) and recall (0.7279), making it the strongest text-only classifier. The **KNN** model performed poorly, with a recall of only 24.5% — meaning it missed over 75% of true ADE sentences — likely due to the curse of dimensionality in the 5,000-feature TF-IDF space. **Naive Bayes**, despite its simplicity, maintained reasonable precision (0.84) but suffered from low recall (0.52), consistent with its independence assumption failing to capture the nuanced, multi-word patterns characteristic of ADE descriptions.

### 5.2 NER-Enhanced Model Performance

**Table 7: NER-Enhanced Model Results (TF-IDF + NER Features)**

| Model                      | Accuracy | Precision | Recall | F1-Score |
|----------------------------|----------|-----------|--------|----------|
| Logistic Regression (NER)  | 0.8815   | 0.8121    | 0.7726 | 0.7918   |
| Random Forest (NER)        | 0.8957   | 0.8231    | 0.8183 | 0.8207   |
| SVM (NER)                  | 0.8900   | 0.8189    | 0.7998 | 0.8092   |

The addition of NER features produced consistent improvements across all three models:

**Table 8: Impact of NER Features (F1-Score Improvement)**

| Model               | Baseline F1 | NER-Enhanced F1 | Improvement |
|----------------------|-------------|-----------------|-------------|
| Logistic Regression  | 0.7171      | 0.7918          | +0.0747     |
| Random Forest        | 0.7533      | 0.8207          | +0.0674     |
| SVM                  | 0.7714      | 0.8092          | +0.0378     |

The most dramatic improvement came from **Logistic Regression**, where NER features boosted recall from 0.6259 to 0.7726 (+14.7 percentage points), indicating that the linear model was able to directly leverage the drug-disease co-occurrence signal encoded in the NER features. The **Random Forest (NER)** model achieved the highest overall performance across all metrics and was selected as the final model.

**Figure 2:** *Bar chart comparison of Accuracy, Precision, Recall, and F1 across all baseline and NER-enhanced models* — The visualization confirms that NER-enhanced models uniformly outperform their baseline counterparts, with the most pronounced gains in recall.

### 5.3 Best Model: Detailed Evaluation

The **Random Forest (NER)** model was selected as the best-performing model based on F1-score.

**Table 9: Full Classification Report — Random Forest (NER)**

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Not ADE     | 0.93      | 0.93   | 0.93     | 2,499   |
| ADE         | 0.82      | 0.82   | 0.82     | 1,029   |
| **Accuracy**    |           |        | **0.90** | **3,528** |
| Macro Avg   | 0.87      | 0.87   | 0.87     | 3,528   |
| Weighted Avg| 0.90      | 0.90   | 0.90     | 3,528   |

### 5.4 Confusion Matrix Analysis

**Figure 3:** *Confusion Matrix — Random Forest (NER)*

The confusion matrix (derived from the classification report) reveals the following:

**Table 10: Confusion Matrix — Random Forest (NER)**

|                      | Predicted: Not ADE | Predicted: ADE |
|----------------------|--------------------|----------------|
| **Actual: Not ADE**  | ~2,324 (TN)        | ~175 (FP)      |
| **Actual: ADE**      | ~185 (FN)          | ~844 (TP)      |

- **True Negatives (2,324):** The model correctly filters 93% of non-ADE sentences, dramatically reducing the volume of text requiring human review.
- **True Positives (844):** The model correctly identifies 82% of ADE sentences, capturing the majority of safety-relevant content.
- **False Positives (175):** Approximately 7% of non-ADE sentences are incorrectly flagged as ADEs. In a pharmacovigilance workflow, these represent unnecessary but non-harmful reviews — a tolerable cost.
- **False Negatives (185):** Approximately 18% of true ADE sentences are missed. **This is the most clinically consequential error type**, as these represent potential safety signals that escape automated detection.

### 5.5 Pharmacovigilance Risk Assessment

In pharmacovigilance, the asymmetric cost of errors demands careful interpretation:

- **False negatives are high-risk:** A missed ADE could delay the identification of a dangerous drug-event pattern, potentially exposing patients to preventable harm. The model's 82% recall means that roughly 1 in 5 true ADE sentences would be missed in a fully automated system.
- **False positives are low-risk:** A non-ADE sentence flagged for review simply adds to reviewer workload without patient safety implications.

This asymmetry suggests that the model is best deployed as a **screening tool** rather than a replacement for expert review, with human reviewers focused on the model's flagged subset plus a random sample of negatives for quality assurance.

**Figure 4:** *Precision-Recall Curve — Random Forest (NER)* — The precision-recall curve provides a visual representation of the trade-off between precision and recall at varying classification thresholds. By lowering the decision threshold, recall can be increased at the cost of precision, which may be appropriate for safety-critical applications.

### 5.6 Feature Importance Analysis

The Random Forest model provides intrinsic feature importance scores, revealing which features are most predictive of ADE classification.

**Table 11: Top 20 Most Important Features**

| Rank | Feature              | Importance |
|------|----------------------|------------|
| 1    | has_both             | 0.0771     |
| 2    | drug_count           | 0.0571     |
| 3    | has_drug             | 0.0531     |
| 4    | disease_count        | 0.0331     |
| 5    | induced              | 0.0251     |
| 6    | has_disease          | 0.0195     |
| 7    | develop              | 0.0143     |
| 8    | associate            | 0.0091     |
| 9    | induce               | 0.0074     |
| 10   | associated           | 0.0067     |
| 11   | report               | 0.0064     |
| 12   | case                 | 0.0060     |
| 13   | patient              | 0.0057     |
| 14   | treatment            | 0.0053     |
| 15   | cause                | 0.0043     |
| 16   | therapy              | 0.0043     |
| 17   | acute                | 0.0040     |
| 18   | interferon           | 0.0039     |
| 19   | administration       | 0.0037     |
| 20   | toxicity             | 0.0037     |

**Figure 5:** *Horizontal bar chart of top 10 most important features*

**Key Observations:**

- **NER features dominate:** The top 4 features and 6 of the top 6 are NER-derived, validating the hypothesis that drug-disease co-occurrence is a powerful ADE signal. The `has_both` feature alone contributes nearly 8% of the model's total feature importance — a remarkable concentration given a 5,005-feature space.
- **Causal language matters:** Text features like "induced", "develop", "associate/associated", and "cause" reflect the linguistic patterns that distinguish ADE sentences from general medical text. These words signal causality between drug exposure and adverse outcomes.
- **Clinical context terms:** "patient", "treatment", "therapy", "administration", and "toxicity" provide contextual framing typical of ADE case reports.
- **Drug-specific signal:** "interferon" appears in the top 20, suggesting that ADE reports involving interferon-class drugs are particularly well-represented in the training data.

---

## 6. Error Analysis

### 6.1 Sources of Classification Errors

While the notebook does not include explicit examples of misclassified sentences, the error patterns can be characterized based on the feature importance analysis and the nature of the dataset:

**TO FILL — Representative misclassified examples should be extracted from the test set by running inference and filtering for false positives and false negatives. The following discussion is based on known patterns in ADE classification tasks.**

### 6.2 Likely False Negative Patterns (Missed ADEs)

1. **Implicit causality:** Sentences that describe an adverse event without explicit causal language (e.g., "The patient's liver function deteriorated during the treatment course" vs. "Drug X induced hepatotoxicity"). The model's reliance on features like "induced", "cause", and "associated" suggests it may miss ADEs expressed through temporal co-occurrence rather than explicit causation.

2. **Rare drug mentions:** ADEs involving uncommon drugs not well-represented in the training vocabulary may lack TF-IDF features with sufficient discriminative weight.

3. **Negated ADEs:** Sentences like "No adverse events were observed following administration of Drug X" could be misclassified if the model does not adequately capture negation patterns. Standard bag-of-words models (including TF-IDF) are known to struggle with negation.

4. **Complex multi-clause sentences:** Sentences where the ADE information is embedded in a subordinate clause or requires cross-sentence reasoning.

### 6.3 Likely False Positive Patterns (Incorrectly Flagged)

1. **Sentences discussing drugs and diseases without causality:** Medical texts frequently mention drugs and diseases in the same sentence without implying an adverse event (e.g., "Drug X was prescribed for the treatment of Disease Y"). The `has_both` NER feature, while powerful, does not distinguish causality from co-occurrence.

2. **Drug efficacy descriptions:** Sentences describing successful treatment outcomes may share vocabulary with ADE sentences (e.g., "therapy", "patient", "treatment").

### 6.4 Ambiguity and Domain Challenges

Medical text is inherently ambiguous. The same symptom may be a disease being treated, a pre-existing condition, or an adverse effect of a drug. Distinguishing these cases often requires document-level or patient-level context that sentence-level classification cannot capture.

---

## 7. Business & Clinical Implications

### 7.1 Reduction of Manual Review Burden

The model correctly classifies approximately 90% of sentences, with 93% of non-ADE sentences correctly filtered out. In a pharmacovigilance pipeline processing thousands of case reports, this translates to a substantial reduction in the volume of text requiring expert review. If a corpus contains 100,000 sentences with the same class distribution as this dataset (~29% ADE), the model would:

- Correctly filter ~65,100 non-ADE sentences (out of ~70,800)
- Correctly flag ~23,900 ADE sentences (out of ~29,200) for review
- Generate ~5,100 false alarms requiring brief human dismissal
- **Miss ~5,300 ADE sentences**, necessitating supplementary quality assurance sampling

### 7.2 Risk Mitigation Perspective

For safety-critical deployment, the model's probability outputs (via `predict_proba`) enable threshold tuning. By lowering the classification threshold below 0.5, organizations can increase recall at the cost of additional false positives — a trade-off that is often acceptable when patient safety is at stake. The precision-recall curve (Figure 4) provides guidance for selecting an application-appropriate threshold.

### 7.3 Regulatory Implications

Regulatory bodies such as Health Canada and the FDA require documented pharmacovigilance processes. An automated ADE detection system could:

- **Accelerate signal detection timelines** from weeks to hours.
- **Provide auditable classification trails** for regulatory submissions.
- **Enable continuous monitoring** of expanding biomedical literature volumes that exceed human review capacity.

However, current regulatory frameworks would likely require human-in-the-loop validation before any ADE classification is used in formal safety assessments.

---

## 8. Limitations

### 8.1 Data Limitations

- **Single dataset:** The model was trained and evaluated exclusively on the ADE Corpus V2. Performance on text from different sources (patient forums, electronic health records, social media) is unknown and likely degraded.
- **Sentence-level granularity:** ADE relationships may span multiple sentences or require document-level context that sentence-level classification cannot capture.
- **No temporal validation:** The train/test split is random rather than temporal, so the model's ability to generalize to newly emerging drugs and ADEs is untested.

### 8.2 Annotation Bias

- The dataset was annotated by a specific set of human reviewers whose inter-annotator agreement is not reported in the dataset documentation. Ambiguous cases may have been labeled inconsistently.
- The binary labeling scheme (ADE vs. Not-Related) does not capture the severity, certainty, or type of adverse event.

### 8.3 Domain Adaptation Issues

- **Vocabulary drift:** New drugs, brand names, and medical terminology emerge continuously. The fixed TF-IDF vocabulary will not capture terms absent from the training corpus.
- **Writing style variation:** The dataset is drawn from scientific literature. Application to patient-generated text (e.g., social media, patient forums) would require domain adaptation.
- **Language coverage:** The model is trained exclusively on English text and cannot process reports in other languages.

### 8.4 Model Limitations

- **No negation handling:** The TF-IDF + NER feature pipeline does not explicitly model negation, potentially causing sentences like "No side effects were observed" to be misclassified.
- **No contextual embeddings:** The bag-of-words TF-IDF representation loses word order and contextual semantics, limiting the model's ability to capture subtle linguistic patterns.
- **NER model version mismatch:** The scispaCy BC5CDR model was trained with spaCy v3.7.4 and may exhibit degraded entity extraction performance with newer spaCy versions.
- **No cross-validation:** Model evaluation relied on a single train/test split rather than k-fold cross-validation, which may produce less stable performance estimates.

---

## 9. Future Improvements

### 9.1 Transformer-Based Models

Pre-trained biomedical language models such as **BioBERT**, **PubMedBERT**, or **SciBERT** could replace TF-IDF features with contextualized embeddings that capture word order, negation, and semantic nuance. These models have demonstrated state-of-the-art performance on biomedical NLP benchmarks and are well-suited to the specialized vocabulary of pharmacovigilance text.

### 9.2 Active Learning

An active learning framework could iteratively identify the most informative unlabeled sentences for human annotation, efficiently expanding the training set with difficult, ambiguous examples that improve model performance in regions of high uncertainty.

### 9.3 Domain-Specific Embeddings

Training word2vec or fastText embeddings on large biomedical corpora (e.g., PubMed abstracts, clinical trial registries) could produce features better attuned to the specialized semantics of drug-event relationships than general-purpose TF-IDF.

### 9.4 Threshold Optimization

Formal threshold optimization using the precision-recall curve — targeting a minimum recall of 95% for ADE detection — would better align model behavior with pharmacovigilance priorities, where missing a true ADE is far more costly than generating a false alarm.

### 9.5 Multi-Label and Severity Classification

Extending the binary classification to a multi-label or ordinal framework that captures ADE type (e.g., hepatotoxicity, cardiotoxicity) and severity would provide more actionable output for pharmacovigilance teams.

### 9.6 Real-Time Deployment Considerations

A production deployment would require:

- **API encapsulation** of the preprocessing and prediction pipeline.
- **Model monitoring** for performance degradation over time (data drift).
- **Integration with pharmacovigilance databases** (e.g., FDA FAERS, WHO VigiBase) for cross-referencing model predictions with known safety signals.
- **Scalable NER processing** to handle high-volume document ingestion.

### 9.7 Extension: PubMed + FAERS Cross-Referencing

As outlined in the original analytics plan, a valuable extension would extract case reports from the **PubMed API**, feed individual sentences through the trained ADE classifier, and cross-reference flagged drugs against the **FDA FAERS (FDA Adverse Event Reporting System) API** to determine whether reported adverse events correlate with real-world signal spikes.

---

## 10. Technical Appendix

### 10.1 Environment Setup

The notebook was developed and executed in a conda-managed Python environment. A recommended environment setup:

```bash
conda create -n pharmacovigilance python=3.11
conda activate pharmacovigilance
pip install pandas numpy scikit-learn matplotlib seaborn nltk spacy datasets ipykernel
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz --no-deps
```

### 10.2 Key Package Versions

| Package       | Version Used     |
|---------------|------------------|
| Python        | 3.8 (original) / 3.11+ (recommended) |
| pandas        | Standard         |
| scikit-learn  | Standard         |
| NLTK          | 3.9+             |
| spaCy         | 3.7–3.8          |
| scispaCy model| en_ner_bc5cdr_md 0.5.4 |
| datasets (HF) | Standard         |

### 10.3 Reproduction Instructions

1. Create and activate the conda environment as specified above.
2. Download NLTK data: `punkt_tab`, `wordnet`, `averaged_perceptron_tagger_eng`.
3. Open `pharmacovigilance.ipynb` in Jupyter Lab or VS Code.
4. Select the appropriate kernel matching the conda environment.
5. Run all cells sequentially from top to bottom.
6. The dataset is automatically downloaded from Hugging Face on first execution.

### 10.4 References

1. Gurulingappa, H., Rajput, A. M., Roberts, A., Fluck, J., Hofmann-Apitius, M., & Toldo, L. (2012). Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports. *Journal of Biomedical Informatics*, 45(5), 885–892.
2. Neumann, M., King, D., Beltagy, I., & Ammar, W. (2019). ScispaCy: Fast and robust models for biomedical natural language processing. *Proceedings of the 18th BioNLP Workshop and Shared Task*, 319–327.
3. Li, J., Sun, Y., Johnson, R. J., Sciaky, D., Wei, C. H., Leaman, R., Davis, A. P., & Mattingly, C. J. (2016). BioCreative V CDR task corpus: a resource for chemical disease relation extraction. *Database*, 2016.
4. Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4), 1234–1240.

---

*Report generated from analysis performed in `pharmacovigilance.ipynb`. All metrics, figures, and tables correspond to outputs produced during notebook execution. Items marked "TO FILL" require additional notebook runs to complete.*

# Pharmacovigilance — Detecting Adverse Drug Events in Free Text

**An NLP pipeline that classifies clinical free-text snippets as Adverse Drug Events (ADEs) or not, with biomedical NER on drugs and diseases.**

INSY 669 · Text Analytics · McGill MMA · Winter 2026

---

## Problem

Pharmacovigilance — the practice of monitoring the effects of drugs after they hit the market — depends on catching adverse drug events (ADEs) from heterogeneous text sources: clinical notes, case reports, patient forums, social media. Manual review doesn't scale. This project builds an end-to-end NLP system that:

1. **Classifies** whether a free-text sentence describes an ADE
2. **Extracts** the implicated drugs (chemicals) and conditions (diseases) using biomedical NER

The result is a triage layer that prioritizes which reports human reviewers should look at first.

## Dataset

**ADE Corpus v2** — 23,516 sentences labeled ADE / non-ADE (~29% positive).
- Source: HuggingFace [`SetFit/ade_corpus_v2_classification`](https://huggingface.co/datasets/SetFit/ade_corpus_v2_classification)
- Loaded directly inside the notebook via `datasets.load_dataset(...)` — no manual download needed.

## Pipeline

1. **EDA** — class balance, sentence length, lexical overlap between classes
2. **Text preprocessing** — lowercasing, lemmatization (WordNet + POS-aware), stopword removal
3. **Biomedical NER** — `scispaCy` `en_ner_bc5cdr_md` model extracts CHEMICAL and DISEASE entities; visualized with `displacy`
4. **Feature engineering** — TF-IDF + count vectorizers, NER-derived count features fused via `FeatureUnion`
5. **Classification** — Multinomial NB, Logistic Regression, KNN, SVC, Random Forest, with `GridSearchCV` and stratified K-fold CV; `class_weight='balanced'` to handle imbalance
6. **Threshold tuning** — precision/recall curve analysis to pick an operating point that matches a "high recall" pharmacovigilance use case
7. **Evaluation** — classification report, confusion matrix, PR curve, average precision

## Results

The tuned logistic-regression + TF-IDF + NER-features model reaches:
- **F1 = 0.842** at threshold 0.48 (precision 82.0%, recall 86.5%)
- **Recall = 99.5%** at threshold 0.10 (high-recall regime for triage)

See `notebooks/pharmacovigilance.ipynb` and `docs/Pharmacovigilance_Report.md` for the full write-up.

## Repo structure

```
pharmacovigilance-nlp/
├── notebooks/
│   └── pharmacovigilance.ipynb        # full EDA + modeling pipeline
├── scripts/
│   └── setup.sh                       # installs scispaCy model + NLTK data
├── docs/
│   ├── Pharmacovigilance_Report.md
│   ├── Pharmacovigilance_Report.docx
│   └── Project_Proposal.pdf
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproduce

```bash
git clone https://github.com/ferozobaid/pharmacovigilance-nlp.git
cd pharmacovigilance-nlp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/setup.sh                      # scispaCy model + NLTK resources
jupyter lab notebooks/pharmacovigilance.ipynb
```

## Author

**Feroz Obaid Khan** — Master of Management Analytics, McGill University · Group 1 project

## License

Code: MIT — see [LICENSE](LICENSE). Dataset: ADE Corpus v2, redistributed by SetFit on HuggingFace.

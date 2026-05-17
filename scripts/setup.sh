#!/usr/bin/env bash
# Set up the pharmacovigilance NLP environment.
# The ADE corpus loads automatically via `datasets.load_dataset`,
# so no separate data download is required — this script installs the
# scispaCy biomedical NER model and NLTK resources that the notebook needs.

set -euo pipefail

echo "Installing scispaCy BC5CDR model ..."
pip install \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

echo "Downloading NLTK resources ..."
python - <<'PY'
import nltk
for pkg in ["punkt", "punkt_tab", "wordnet", "averaged_perceptron_tagger", "omw-1.4"]:
    nltk.download(pkg)
PY

echo "Done. Open notebooks/pharmacovigilance.ipynb to run the pipeline."

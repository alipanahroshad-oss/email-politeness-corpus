# A Synthetic Request–Reply Email Corpus Annotated with Document-Level Politeness and Sentence-Level Face Acts

**Roshad Alipanah, Valentin Barriere, and Jorge Baier**  
**Findings of the Association for Computational Linguistics: EMNLP 2026**

This repository contains the official implementation accompanying our work on a synthetic request–reply email corpus jointly annotated with **sentence-level Face Acts (FA)** and **document-level politeness**, both grounded in **Brown and Levinson's politeness theory**.

The repository provides the experimental pipeline and supporting analyses, including:

- Synthetic corpus generation from Enron-inspired request–reply scenarios
- Controlled generation of politeness-graded email variants using GPT-4o
- Sentence-level Face Act Classification (FAC)
- Document-level Overall Politeness Regression (OPR)
- Human annotation files and annotation guidelines
- Inter-annotator reliability analyses
- Official train/validation/test splits
- Validation analyses of the generated corpus
- Oracle analysis using gold Face Act annotations

---

# Repository Structure

```text
.
├── data/
│   ├── corpus/                     # Final sentence- and document-level datasets
│   ├── annotation/                 # Human annotation files
│   ├── validation/                 # Validation datasets
│   └── splits/                     # Official train/validation/test splits
│
├── docs/
│   ├── Face_Acts_Annotation_Guideline.pdf
│   └── Politeness_Scoring_Annotation_Guidelines.pdf
│
├── src/
│   ├── generation/                 # Corpus generation and validation
│   ├── face_act_classification/    # Face Act Classification (FAC)
│   │
│   ├── overall_politeness_regression/
│   │   └── OPR_one_sentence.py     # Overall Politeness Regression (OPR)
│   │
│   ├── reliability/
│   │   ├── krippendorff_alpha_face_acts.ipynb
│   │   └── krippendorff_alpha_politeness.ipynb
│   │
│   └── ablations/
│       └── OPR_oracle_goldfa.py
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# Repository Overview

## Corpus Generation and Validation

The `src/generation/` module contains the scripts used to construct and validate the synthetic request–reply email corpus.

The generation pipeline consists of three stages:

1. **Seed Email Generation.** Topics extracted from the Enron Email Dataset are used to generate request–reply email pairs. Requests are designed to contain direct face-threatening acts (FTAs), while replies vary in acceptance/rejection and face-saving/face-threatening strategies.

2. **Politeness-Level Generation.** GPT-4o paraphrases every request and reply into four progressively more polite variants while preserving the original communicative intent, producing five politeness levels (1–5).

3. **Corpus Validation.** Validation analyses examine whether GPT-4o's intended politeness levels align with independent human document-level annotations. Across the three document-level politeness dimensions, human scores increase monotonically from politeness level 1 to level 5, indicating that the generated variants correspond to meaningful differences in perceived politeness.

---

## Face Act Classification (FAC)

The `src/face_act_classification/` module contains the **BERT-based** Face Act Classification models evaluated in the paper.

The task is formulated as **multi-label sentence classification** over the nine Face Act categories introduced in the annotation scheme.

The repository includes implementations of:

- **One-Sentence classification**, where each sentence is classified independently.
- **Context-aware sentence classification**, using either the current email history or the full paired request–reply history as context.
- **Sequence labeling models**, which jointly predict Face Act labels for all sentences within a current email history or paired email history.

These experiments investigate the impact of contextual information and sequence modeling on sentence-level Face Act prediction.

---

## Overall Politeness Regression (OPR)

The `src/overall_politeness_regression/` module contains the implementation used for the document-level politeness prediction experiments reported in the paper.

Overall Politeness Regression predicts scores for the three document-level politeness dimensions:

- **Directness vs. Indirectness**
- **Positive Face Saving**
- **Negative Face Saving**

The released implementation includes:

- **Text-only**, which predicts document-level politeness directly from the email text.
- **PredFA-only**, which predicts document-level politeness from aggregated predicted Face Act features.
- **Text + PredFA**, which combines the textual representation with aggregated Face Act features predicted by the FAC model.

The predicted Face Act probabilities are aggregated at the document level using the **HSPT-13** representation, consisting of nine Face Act dimensions and four aggregated Face Act features.

---

## Annotation Reliability

The `src/reliability/` directory contains the inter-annotator reliability analyses for both annotation levels.

It includes:

- `krippendorff_alpha_face_acts.ipynb`  
  Computes **Krippendorff's alpha using Jaccard set distance** for the multi-label sentence-level Face Act annotations.

- `krippendorff_alpha_politeness.ipynb`  
  Computes **Krippendorff's alpha with interval distance** for the three document-level politeness dimensions:
  - Directness vs. Indirectness
  - Positive Face Saving
  - Negative Face Saving

---

## Ablation and Oracle Analysis

The `src/ablations/` directory contains additional analyses used to investigate the contribution of Face Act information to document-level politeness prediction.

The released oracle experiment is:

- `OPR_oracle_goldfa.py`  
  Replaces predicted Face Act features with **gold human-annotated Face Acts** while retaining the same HSPT-13 feature representation and OPR architecture.

The oracle comparison evaluates:

- **Text-only**
- **Text + GoldFA**

This experiment provides an upper-bound analysis of the contribution of Face Act information when gold sentence-level Face Act annotations are available.

---

# Data

## Corpus

The `data/corpus/` directory contains the final datasets used in the experiments, including:

- Sentence-level Face Act annotations
- Document-level politeness annotations

---

## Annotation

The `data/annotation/` directory contains the human annotation files used to produce the final gold labels and calculate inter-annotator reliability.

---

## Validation

The `data/validation/` directory contains the datasets used for validating the generated corpus, including the analysis of the alignment between GPT-4o intended politeness levels and independent human document-level politeness scores.

---

## Data Splits

The `data/splits/` directory contains the official train, validation, and test splits used throughout the experiments.

The splits are constructed at the seed level to prevent overlap between related generated emails across training, validation, and test sets.

---

# Annotation Guidelines

The repository includes the annotation guidelines used during corpus construction:

- `docs/Face_Acts_Annotation_Guideline.pdf`  
  Guidelines for the sentence-level Face Act annotation task.

- `docs/Politeness_Scoring_Annotation_Guidelines.pdf`  
  Guidelines for the document-level politeness scoring task.

---

# Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

---

# Experimental Pipeline

```text
Enron Topics
      │
      ▼
Synthetic Request–Reply Generation
      │
      ▼
Politeness-Level Generation (1–5)
      │
      ▼
Human Annotation
      │
      ├────────► Sentence-Level Face Acts
      │
      └────────► Document-Level Politeness
                    │
                    ▼
         Face Act Classification (FAC)
                    │
                    ▼
          Predicted Face Acts
                    │
                    ▼
     Overall Politeness Regression (OPR)

Additional Analyses
      │
      ├────────► Inter-Annotator Reliability
      │
      └────────► GoldFA Oracle Experiment
```

---

# Citation

If you use this dataset or code, please cite:

```bibtex
@inproceedings{alipanah2026synthetic,
  title     = {A Synthetic Request--Reply Email Corpus Annotated with Document-Level Politeness and Sentence-Level Face Acts},
  author    = {Alipanah, Roshad and Barriere, Valentin and Baier, Jorge},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

---

# License

This repository is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** License.

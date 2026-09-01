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
- Misaligned PredFA ablation analysis
- Saved evaluation artifacts for exact reproduction of the reported oracle and misaligned results

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
├── artifacts/
│   ├── misaligned/
│   │   ├── main_opr/
│   │   │   ├── bert_text_st/
│   │   │   │   └── metrics.json
│   │   │   └── predfa_mlp_st/
│   │   │       └── metrics.json
│   │   └── shuffled_predfa/
│   │       └── metrics.json
│   │
│   └── oracle/
│       ├── main_opr/
│       │   └── bert_text_st/
│       │       ├── metrics.json
│       │       ├── test_y.npy
│       │       └── test_pred.npy
│       └── goldfa/
│           ├── metrics.json
│           ├── test_y.npy
│           └── test_pred.npy
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
│       ├── OPR_oracle_goldfa.py
│       └── OPR_misaligned_predfa.py
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

For the **main OPR results reported in the paper, as well as the oracle and misaligned PredFA analyses, we report the single-task (ST) setting**, where an independent regression model is trained for each politeness dimension.

---

## Document-Level Politeness Dimension Names

The raw annotation files retain the original annotation column names. In the paper and repository documentation, two of these dimensions are reported using their final theoretical names:

- `Structural_Politeness_and_Politeness_Markers` → **Positive Face Saving**
- `Tone_and_Overall_Consideration` → **Negative Face Saving**

The corresponding gold target columns used by the OPR experiments are:

- `Directness_vs_Indirectness__GOLD` → **Directness vs. Indirectness**
- `Structural_Politeness_and_Politeness_Markers__GOLD` → **Positive Face Saving**
- `Tone_and_Overall_Consideration__GOLD` → **Negative Face Saving**

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

The `src/ablations/` directory contains additional analyses used to investigate the contribution and alignment of Face Act information in document-level politeness prediction.

The included analyses are:

- `OPR_oracle_goldfa.py`  
  Evaluates an oracle setting in which predicted Face Act features are replaced with **gold human-annotated Face Acts**, aggregated into email-level counts and summary features, while retaining the same OPR architecture.

- `OPR_misaligned_predfa.py`  
  Reproduces the **misaligned PredFA ablation**, in which the relationship between the text and predicted Face Act representation is disrupted. The script verifies the saved evaluation artifacts corresponding to the reported results.

For both analyses, the reported results use the **single-task (ST)** OPR setting.

### Oracle Analysis

The oracle comparison evaluates:

- **Text-only**
- **Text + GoldFA**

The GoldFA representation is constructed from sentence-level gold Face Act annotations and aggregated using the same HSPT-13 feature definition used for PredFA.

The saved artifacts used to verify the reported Oracle comparison are provided under:

```text
artifacts/oracle/
├── main_opr/
│   └── bert_text_st/
│       ├── metrics.json
│       ├── test_y.npy
│       └── test_pred.npy
└── goldfa/
    ├── metrics.json
    ├── test_y.npy
    └── test_pred.npy
```

The Oracle script verifies that the saved Text-only baseline and GoldFA test targets correspond exactly to the reconstructed test split.

### Misaligned PredFA Analysis

The misaligned PredFA analysis compares the reported results for:

- **Text-only**
- **PredFA-only**
- **Text + Misaligned PredFA**

The saved evaluation artifacts required to reproduce and verify this comparison are provided under:

```text
artifacts/misaligned/
├── main_opr/
│   ├── bert_text_st/
│   │   └── metrics.json
│   └── predfa_mlp_st/
│       └── metrics.json
└── shuffled_predfa/
    └── metrics.json
```

---

# Data

## Corpus

The `data/corpus/` directory contains the final datasets used in the experiments, including:

- Sentence-level Face Act annotations
- Document-level politeness annotations

---

## Annotation

The `data/annotation/` directory contains the human annotation files used to produce the final gold labels and calculate inter-annotator reliability.

The original annotation column names are retained in these files. In particular:

- `Structural_Politeness_and_Politeness_Markers` corresponds to **Positive Face Saving**.
- `Tone_and_Overall_Consideration` corresponds to **Negative Face Saving**.

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
              [reported ST setting]

Additional Analyses
      │
      ├────────► Inter-Annotator Reliability
      │
      ├────────► GoldFA Oracle Experiment [ST]
      │
      └────────► Misaligned PredFA Ablation [ST]
```

---

# Citation

If you use this dataset or code, please cite:

```bibtex
@inproceedings{alipanah2026synthetic,
  author    = {Roshad Alipanah and
               Valentin Barriere and
               Jorge A. Baier},
  title     = {A Synthetic Request--Reply Email Corpus Annotated with Document-Level Politeness and Sentence-Level Face Acts},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  publisher = {Association for Computational Linguistics},
  year      = {2026}
}
```

---

# License

This repository is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** License.

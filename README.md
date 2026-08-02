# A Synthetic Request–Reply Email Corpus Annotated with Document-Level Politeness and Sentence-Level Face Acts

This repository contains the official implementation accompanying our work on a synthetic request–reply email corpus jointly annotated with **sentence-level Face Acts (FA)** and **document-level politeness**, both grounded in **Brown and Levinson's politeness theory**.

The repository provides the complete experimental pipeline, including:

- Synthetic corpus generation from Enron-inspired request–reply scenarios
- Controlled generation of politeness-graded email variants using GPT-4o
- Sentence-level Face Act Classification (FAC)
- Document-level Overall Politeness Regression (OPR)
- Human annotation files and annotation guidelines
- Official train/validation/test splits
- Validation analyses of the generated corpus

The repository includes all code, datasets, and resources required to reproduce the experiments presented in the paper.

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
│   ├── Face_Act_Annotation_Guidelines.pdf
│   └── Politeness_Scoring_Annotation_Guidelines.pdf
│
├── src/
│   ├── generation/                 # Corpus generation and validation
│   ├── face_act_classification/    # Face Act Classification (FAC)
│   └── overall_politeness_regression/      # Overall Politeness Regression (OPR)
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

3. **Corpus Validation.** Validation analyses demonstrate that GPT-4o's intended politeness levels align with independent human document-level annotations. Across all three politeness dimensions, human scores increase monotonically from politeness level 1 to level 5, indicating that the generated variants correspond to meaningful differences in perceived politeness.

---

## Face Act Classification (FAC)

The `src/face_act_classification/` module contains all Face Act Classification models evaluated in the paper.

The task is formulated as **multi-label sentence classification** over the nine Face Act categories introduced in the annotation scheme. The repository includes implementations of:

- **One-Sentence classification**, where each sentence is classified independently.
- **Context-aware sentence classification**, using either the current email history or the full paired request–reply history as context.
- **Sequence labeling models**, which jointly predict Face Act labels for all sentences within a current email history or paired email history.

These experiments investigate the impact of contextual information and sequence modeling on sentence-level Face Act prediction.

---

## Overall Politeness Regression (OPR)

The `src/politeness_regression/` module contains the implementation used for the document-level politeness prediction experiments reported in the paper.

Overall Politeness Regression is formulated as a **multi-target regression** task over the three document-level politeness dimensions:

- Directness vs. Indirectness
- Positive Face Saving
- Negative Face Saving

The released implementation includes:
- **Text-only** models that predict document-level politeness directly from the email text. 
- **Text + PredFA** models that combine the textual representation with aggregated predicted Face Act features (PredFA).
The experiments evaluate whether predicted Face Act information improves document-level politeness prediction compared with text-only models.

---

# Data

## Corpus

The `data/corpus/` directory contains the final datasets used in the experiments:

- Sentence-level Face Act annotations
- Document-level politeness annotations

---

## Annotation

The `data/annotation/` directory contains the human annotation files used to produce the final gold labels.

---

## Validation

The `data/validation/` directory contains the datasets used for validating the generated corpus, including the analysis of the alignment between GPT-4o intended politeness levels and independent human document-level politeness scores.

---

## Data Splits

The `data/splits/` directory contains the official train, validation, and test splits used throughout all experiments.

---

# Annotation Guidelines

The repository includes the annotation guidelines used during corpus construction:

- `docs/Face_Act_Annotation_Guidelines.pdf`  
  Guidelines for the sentence-level Face Act annotation task.

- `docs/Document_Level_Politeness_Annotation_Guidelines.pdf`  
  Guidelines for the document-level politeness scoring task.

---

# Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

---
# Repository Status

This repository accompanies a research paper currently under review. Additional documentation and minor updates may be added following publication.

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
```

---

# Citation

If you use this repository in your research, please cite the accompanying paper once it becomes publicly available.
---


## License

This repository is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License.

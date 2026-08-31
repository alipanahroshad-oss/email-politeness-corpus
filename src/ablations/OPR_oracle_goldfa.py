#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle OPR experiment matched to the main seed-42 OPR run.

Comparison produced by this script
----------------------------------
Main OPR:   Text-only       vs. Text + PredFA
Oracle OPR: SAME Text-only  vs. Text + GoldFA

The Text-only model is deliberately NOT retrained here. Its saved metrics,
test targets, and test predictions are read from the exact seed-42 artifacts
released with this repository and copied byte-for-byte into the oracle output
directory after verification.

The saved Text + GoldFA test artifacts are also reused when valid. Their test
targets must exactly match the seed-42 test targets reconstructed from the
released corpus and split logic. If the GoldFA artifacts are missing or fail
validation, the script falls back to training Text + GoldFA with the locked
configuration below.

GoldFA is constructed from the sentence-level GoldFaceAct column. Each gold
label is represented as 0/1 at sentence level and aggregated into the same
locked HSPT-13 feature definition used for PredFA:
  9 per-label sums + sum_H + sum_S + sum_praise + sum_threat.

Expected repository layout:

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

The metric guard below verifies that the Text-only artifact is the exact
seed-42 Text-only ST baseline used for the reported comparison.
"""

from pathlib import Path

# ==================== REPOSITORY PATHS ====================
# This file is intended to live at: src/ablations/OPR_oracle_goldfa.py
REPO_ROOT = Path(__file__).resolve().parents[2]

DOC_CSV = str(
    REPO_ROOT
    / "data"
    / "corpus"
    / "email_text_gold_three_dimensions_politeness_score_with_seed_correct.csv"
)
SENT_CSV = str(
    REPO_ROOT
    / "data"
    / "corpus"
    / "sentences_with_golden_face_act.csv"
)

MAIN_OPR_OUT_DIR = str(
    REPO_ROOT / "artifacts" / "oracle" / "main_opr"
)
GOLDFA_ARTIFACT_DIR = str(
    REPO_ROOT / "artifacts" / "oracle" / "goldfa"
)

OUT_DIR = str(REPO_ROOT / "oracle_outputs_seed42")

# ==================== IMPORTS ====================
import ast
import hashlib
import json
import os
import random
import re
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# ==================== LOCKED CONFIG ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

DOC_ID_COL = "email_id"
DOC_TEXT_COL = "text_email"
SEED_COL = "seed"
SENT_IDX_COL = "sentence_idx"
FA_COL = "GoldFaceAct"

TARGETS = [
    "Directness_vs_Indirectness__GOLD",
    "Structural_Politeness_and_Politeness_Markers__GOLD",
    "Tone_and_Overall_Consideration__GOLD",
]

# Must remain identical to the main PredFA feature order.
FA_LABS = [
    "HNeg+", "HNeg-", "HPos+", "HPos-", "Neutral",
    "SNeg+", "SNeg-", "SPos+", "SPos-",
]
SUM9_ORDER = [f"sumprob_{label}" for label in FA_LABS]
SUM13_EXTRA = ["sum_H", "sum_S", "sum_praise", "sum_threat"]
SUM13_ORDER = SUM9_ORDER + SUM13_EXTRA

DOC_BERT_NAME = "bert-base-uncased"
BERT_LR = 2e-5
BERT_EPOCHS = 5
BERT_BS = 8
BERT_DROPOUT = 0.1
BERT_MAXLEN = 512

# Table 2 reports the single-task (ST) experiment.
DO_MULTI_TASK = False
DO_SINGLE_TASK = True

# Reuse the released Text+GoldFA seed-42 artifacts when they pass all alignment
# checks. If they are missing or invalid, train Text+GoldFA from scratch.
REUSE_EXISTING_GOLDFA_IF_VALID = True

# Exact unrounded Text-only values recorded by the main seed-42 Table 2 run.
# A guard checks these before copying anything into the oracle directory.
EXPECTED_TABLE2_TEXT_ST = {
    "Directness_MAE": 0.3771,
    "Markers_MAE": 0.3152,
    "Overall_MAE": 0.3428,
    "rho_D": 0.8429,
    "rho_M": 0.8973,
    "rho_O": 0.8740,
}
EXPECTED_METRIC_TOLERANCE = 5e-4


# ==================== GENERAL UTILITIES ====================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decategorize_series(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, CategoricalDtype):
        return series.astype(object)
    return series


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, CategoricalDtype):
            df[col] = df[col].astype(object)
    if DOC_ID_COL in df.columns:
        df[DOC_ID_COL] = df[DOC_ID_COL].astype(str)
    if SEED_COL in df.columns:
        df[SEED_COL] = pd.to_numeric(df[SEED_COL], errors="coerce")
    if SENT_IDX_COL in df.columns:
        df[SENT_IDX_COL] = pd.to_numeric(df[SENT_IDX_COL], errors="coerce")
    return df.reset_index(drop=True)


def grouped_split_70_15_15_index(
    groups: np.ndarray, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact copy of the main OPR split logic."""
    idx = np.arange(len(groups))
    gss1 = GroupShuffleSplit(
        n_splits=1, train_size=0.70, random_state=random_state
    )
    tr_idx, tmp_idx = next(gss1.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(
        n_splits=1, train_size=0.50, random_state=random_state
    )
    va_rel, te_rel = next(gss2.split(tmp_idx, groups=groups[tmp_idx]))
    return tr_idx, tmp_idx[va_rel], tmp_idx[te_rel]


def snapshot_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone the best epoch; model.state_dict() alone keeps live references."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ==================== GOLD FA -> HSPT-13 ====================
_FA_PATTERN = re.compile(
    "(" + "|".join(re.escape(label) for label in FA_LABS) + ")"
)


def parse_gold_faceacts(value) -> List[str]:
    """Parse a scalar/list representation of a possibly multi-label gold FA."""
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, (list, tuple, set, np.ndarray)):
        candidates = [str(item).strip() for item in value]
    else:
        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "null"}:
            return []
        candidates = []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                candidates = [str(item).strip() for item in parsed]
            elif isinstance(parsed, str):
                candidates = [parsed.strip()]
        except (ValueError, SyntaxError):
            pass
        if not candidates:
            candidates = _FA_PATTERN.findall(text)

    labels: List[str] = []
    for candidate in candidates:
        if candidate in FA_LABS and candidate not in labels:
            labels.append(candidate)
        else:
            for match in _FA_PATTERN.findall(candidate):
                if match not in labels:
                    labels.append(match)
    return labels


def sentence_gold_binary(df_sent_split: pd.DataFrame) -> pd.DataFrame:
    required = [DOC_ID_COL, SENT_IDX_COL, FA_COL]
    missing = [col for col in required if col not in df_sent_split.columns]
    if missing:
        raise ValueError(f"Sentence CSV is missing columns: {missing}")

    out = df_sent_split[required].copy()
    out[DOC_ID_COL] = _decategorize_series(out[DOC_ID_COL]).astype(str)
    out[SENT_IDX_COL] = (
        pd.to_numeric(_decategorize_series(out[SENT_IDX_COL]), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    parsed = out[FA_COL].map(parse_gold_faceacts)
    invalid_mask = parsed.map(len).eq(0)
    if invalid_mask.any():
        examples = out.loc[invalid_mask, FA_COL].astype(str).head(10).tolist()
        raise ValueError(
            f"Could not parse GoldFaceAct for {int(invalid_mask.sum())} sentences. "
            f"Examples: {examples}"
        )

    for label in FA_LABS:
        out[f"prob_{label}"] = parsed.map(
            lambda labels, wanted=label: float(wanted in labels)
        )

    return out[
        [DOC_ID_COL, SENT_IDX_COL] + [f"prob_{label}" for label in FA_LABS]
    ]


def aggregate_hspt13(binary_fa: pd.DataFrame) -> pd.DataFrame:
    """Aggregate binary sentence-level GoldFA using the main HSPT-13 formula."""
    rows = []
    binary_fa = binary_fa.sort_values(
        [DOC_ID_COL, SENT_IDX_COL], kind="mergesort", ignore_index=True
    )

    for doc_id, group in binary_fa.groupby(DOC_ID_COL, sort=False):
        row = {DOC_ID_COL: str(doc_id)}
        for label in FA_LABS:
            row[f"sumprob_{label}"] = float(group[f"prob_{label}"].sum())

        row["sum_H"] = float(
            row["sumprob_HNeg+"]
            + row["sumprob_HNeg-"]
            + row["sumprob_HPos+"]
            + row["sumprob_HPos-"]
        )
        row["sum_S"] = float(
            row["sumprob_SNeg+"]
            + row["sumprob_SNeg-"]
            + row["sumprob_SPos+"]
            + row["sumprob_SPos-"]
        )
        row["sum_praise"] = float(
            row["sumprob_HPos+"]
            + row["sumprob_HNeg+"]
            + row["sumprob_SPos+"]
            + row["sumprob_SNeg+"]
        )
        row["sum_threat"] = float(
            row["sumprob_HPos-"]
            + row["sumprob_HNeg-"]
            + row["sumprob_SPos-"]
            + row["sumprob_SNeg-"]
        )
        rows.append(row)

    return pd.DataFrame(rows)[[DOC_ID_COL] + SUM13_ORDER]


def align_gold_to_split(
    gold_features: pd.DataFrame, df_doc: pd.DataFrame, idx: np.ndarray
) -> np.ndarray:
    aligned = df_doc.loc[idx, [DOC_ID_COL]].copy()
    aligned[DOC_ID_COL] = aligned[DOC_ID_COL].astype(str)
    aligned = aligned.merge(
        gold_features.astype({DOC_ID_COL: str}),
        on=DOC_ID_COL,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_docs = aligned.loc[aligned["_merge"] != "both", DOC_ID_COL].tolist()
    if missing_docs:
        raise ValueError(
            "GoldFA sentences are missing for document IDs: "
            + ", ".join(missing_docs[:10])
        )
    return aligned[SUM13_ORDER].to_numpy(dtype=np.float32)


def zscore_fit_transform(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd == 0] = 1.0
    return (
        (x_train - mu) / sd,
        (x_val - mu) / sd,
        (x_test - mu) / sd,
        mu,
        sd,
    )


# ==================== DATASET AND MODEL ====================
class DocBertGoldDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        goldfa: np.ndarray,
    ):
        if len(df) != len(goldfa):
            raise ValueError("GoldFA rows do not align with document rows.")
        enc = tokenizer(
            df[DOC_TEXT_COL].astype(str).tolist(),
            padding=True,
            truncation=True,
            max_length=BERT_MAXLEN,
            return_tensors="pt",
        )
        self.ids = enc["input_ids"]
        self.attn = enc["attention_mask"]
        self.targets = torch.tensor(df[TARGETS].to_numpy(np.float32))
        self.goldfa = torch.tensor(goldfa, dtype=torch.float32)

    def __len__(self) -> int:
        return self.ids.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.ids[idx],
            "attention_mask": self.attn[idx],
            "targets": self.targets[idx],
            "goldfa": self.goldfa[idx],
        }


class BertDocRegressor(nn.Module):
    """Exact text/fusion architecture used by the supplied main OPR code."""
    def __init__(self, n_targets: int):
        super().__init__()
        self.enc = AutoModel.from_pretrained(DOC_BERT_NAME)
        hidden_size = self.enc.config.hidden_size
        self.dropout = nn.Dropout(BERT_DROPOUT)
        self.hidden = nn.Sequential(
            nn.Linear(hidden_size + len(SUM13_ORDER), 128),
            nn.ReLU(),
            nn.Dropout(BERT_DROPOUT),
        )
        self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(n_targets)])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        goldfa: torch.Tensor,
    ) -> torch.Tensor:
        output = self.enc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls = (
            output.pooler_output
            if getattr(output, "pooler_output", None) is not None
            else output.last_hidden_state[:, 0, :]
        )
        z = torch.cat([cls, goldfa], dim=-1)
        hidden = self.hidden(self.dropout(z))
        return torch.cat([head(hidden) for head in self.heads], dim=-1)


# ==================== METRICS AND BASELINE REUSE ====================
def mae_spearman(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]
) -> Dict:
    output = {"per_target": {}}
    correlations = []
    for idx, name in enumerate(target_names):
        true_values = y_true[:, idx]
        predictions = y_pred[:, idx]
        mae = float(np.mean(np.abs(true_values - predictions)))
        rho, _ = spearmanr(true_values, predictions)
        rho = float(rho) if rho is not None and not np.isnan(rho) else 0.0
        output["per_target"][name] = {"MAE": mae, "Spearman": rho}
        correlations.append(rho)
    output["rho_macro"] = float(np.mean(correlations))
    return output


def save_metrics(
    outdir: str,
    variant: str,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    y_val: np.ndarray,
    pred_val: np.ndarray,
) -> None:
    ensure_dir(outdir)
    payload = {
        "best_cfg": {
            "variant": variant,
            "fusion": "concat_128_relu_1",
            "feature_source": "GoldFaceAct",
            "feature_definition": "HSPT-13 binary gold-label sums",
            "random_state": RANDOM_STATE,
        },
        "val": mae_spearman(y_val, pred_val, TARGETS),
        "test": mae_spearman(y_test, pred_test, TARGETS),
        "targets": TARGETS,
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    np.save(os.path.join(outdir, "test_y.npy"), y_test)
    np.save(os.path.join(outdir, "test_pred.npy"), pred_test)


def validate_table2_text_st_metrics(metrics: Dict) -> None:
    """Reject a Text-only result that is not the one reported in Table 2."""
    test = metrics.get("test", {})
    per_target = test.get("per_target", {})
    try:
        observed = {
            "Directness_MAE": float(per_target[TARGETS[0]]["MAE"]),
            "Markers_MAE": float(per_target[TARGETS[1]]["MAE"]),
            "Overall_MAE": float(per_target[TARGETS[2]]["MAE"]),
            "rho_D": float(per_target[TARGETS[0]]["Spearman"]),
            "rho_M": float(per_target[TARGETS[1]]["Spearman"]),
            "rho_O": float(per_target[TARGETS[2]]["Spearman"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "The selected main metrics.json does not have the expected OPR "
            "test-metric structure."
        ) from error

    mismatches = []
    for name, expected in EXPECTED_TABLE2_TEXT_ST.items():
        value = observed[name]
        if abs(value - expected) > EXPECTED_METRIC_TOLERANCE:
            mismatches.append(
                f"{name}: found {value:.6f}, expected approximately {expected:.4f}"
            )
    if mismatches:
        raise ValueError(
            "WRONG MAIN TEXT BASELINE. MAIN_OPR_OUT_DIR does not contain the "
            "seed-42 Text-only ST run used in Table 2:\n  "
            + "\n  ".join(mismatches)
            + "\nSelect the exact Table 2 output directory; do not continue with "
            "this oracle comparison."
        )


def reuse_main_text_baseline(
    variant: str, expected_test_y: np.ndarray
) -> Dict[str, np.ndarray]:
    """Copy and verify the main Text-only result; never retrain it."""
    folder = f"bert_text_{variant}"
    source_dir = os.path.join(MAIN_OPR_OUT_DIR, folder)
    dest_dir = os.path.join(OUT_DIR, folder)
    required = ["metrics.json", "test_y.npy", "test_pred.npy"]
    missing = [
        os.path.join(source_dir, name)
        for name in required
        if not os.path.isfile(os.path.join(source_dir, name))
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot guarantee an identical Text-only baseline because these "
            f"main-run files are missing: {missing}"
        )

    with open(os.path.join(source_dir, "metrics.json"), "r") as handle:
        metrics = json.load(handle)
    if metrics.get("targets") != TARGETS:
        raise ValueError(
            f"Target order mismatch in {source_dir}/metrics.json. "
            f"Expected {TARGETS}, found {metrics.get('targets')}"
        )
    if variant == "st":
        validate_table2_text_st_metrics(metrics)

    saved_y = np.load(os.path.join(source_dir, "test_y.npy"))
    saved_pred = np.load(os.path.join(source_dir, "test_pred.npy"))
    if saved_y.shape != expected_test_y.shape or not np.array_equal(
        saved_y, expected_test_y
    ):
        max_diff = None
        if saved_y.shape == expected_test_y.shape:
            max_diff = float(np.max(np.abs(saved_y - expected_test_y)))
        raise ValueError(
            "Main Text-only test_y does not match this script's seed-42 test "
            f"split/target order. Shapes: {saved_y.shape} vs "
            f"{expected_test_y.shape}; max difference: {max_diff}. "
            "Do not compare these runs."
        )
    if saved_pred.shape != saved_y.shape:
        raise ValueError(
            f"Text-only prediction shape {saved_pred.shape} does not match "
            f"test target shape {saved_y.shape}."
        )

    ensure_dir(dest_dir)
    hashes = {}
    for filename in required:
        source = os.path.join(source_dir, filename)
        destination = os.path.join(dest_dir, filename)
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        destination_hash = sha256_file(destination)
        if source_hash != destination_hash:
            raise IOError(f"Copied baseline file failed hash check: {filename}")
        hashes[filename] = source_hash

    with open(os.path.join(dest_dir, "reused_from_main.json"), "w") as handle:
        json.dump(
            {
                "source_directory": source_dir,
                "identity_verified": True,
                "verification": "byte-identical SHA-256 plus exact test_y match",
                "sha256": hashes,
            },
            handle,
            indent=2,
        )

    print(f"[IDENTICAL TEXT] Reused and verified: {source_dir}")
    return {"Y": saved_y, "P": saved_pred}


def can_reuse_existing_goldfa(
    variant: str, expected_test_y: np.ndarray
) -> bool:
    """
    Reuse the released Text+GoldFA artifact when it exactly aligns with the
    reconstructed seed-42 test targets.

    The repository releases the reported single-task (ST) GoldFA artifact at:
      artifacts/oracle/goldfa/

    Multi-task mode has no released GoldFA artifact and therefore falls back to
    the normal training path if enabled.
    """
    if variant != "st":
        return False

    source_dir = GOLDFA_ARTIFACT_DIR
    dest_dir = os.path.join(OUT_DIR, f"bert_text_goldfa_{variant}")
    required = ["metrics.json", "test_y.npy", "test_pred.npy"]

    missing = [
        os.path.join(source_dir, name)
        for name in required
        if not os.path.isfile(os.path.join(source_dir, name))
    ]
    if missing:
        print(
            "[REUSE] Released Text+GoldFA artifacts are incomplete; "
            "falling back to training. Missing:",
            missing,
        )
        return False

    metrics_path = os.path.join(source_dir, "metrics.json")
    y_path = os.path.join(source_dir, "test_y.npy")
    pred_path = os.path.join(source_dir, "test_pred.npy")

    try:
        with open(metrics_path, "r") as handle:
            metrics = json.load(handle)
        saved_y = np.load(y_path)
        saved_pred = np.load(pred_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(
            "[REUSE] Could not validate released Text+GoldFA artifacts; "
            f"falling back to training: {error}"
        )
        return False

    if metrics.get("targets") != TARGETS:
        print(
            "[REUSE] GoldFA target order does not match; "
            "falling back to training."
        )
        return False

    if saved_y.shape != expected_test_y.shape:
        print(
            "[REUSE] GoldFA test_y shape does not match the reconstructed "
            "test split; falling back to training."
        )
        return False

    if not np.array_equal(saved_y, expected_test_y):
        max_diff = float(np.max(np.abs(saved_y - expected_test_y)))
        print(
            "[REUSE] GoldFA test_y does not exactly match the reconstructed "
            f"seed-42 test targets (max difference={max_diff}); "
            "falling back to training."
        )
        return False

    if saved_pred.shape != saved_y.shape:
        print(
            "[REUSE] GoldFA prediction shape does not match test_y; "
            "falling back to training."
        )
        return False

    ensure_dir(dest_dir)
    hashes = {}
    for filename in required:
        source = os.path.join(source_dir, filename)
        destination = os.path.join(dest_dir, filename)
        shutil.copy2(source, destination)

        source_hash = sha256_file(source)
        destination_hash = sha256_file(destination)
        if source_hash != destination_hash:
            raise IOError(
                f"Copied GoldFA artifact failed SHA-256 check: {filename}"
            )
        hashes[filename] = source_hash

    with open(
        os.path.join(dest_dir, "reused_from_artifact.json"), "w"
    ) as handle:
        json.dump(
            {
                "source_directory": source_dir,
                "identity_verified": True,
                "verification": (
                    "byte-identical SHA-256 plus exact test_y match"
                ),
                "sha256": hashes,
            },
            handle,
            indent=2,
        )

    print(
        f"[REUSE] Verified released Text+GoldFA ({variant}) artifacts: "
        f"{source_dir}"
    )
    return True


# ==================== ORACLE TRAINING ====================
def predict_loader(
    model: nn.Module, loader: DataLoader, target_idx: Optional[int]
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_y, all_pred = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["goldfa"].to(DEVICE),
            )
            targets = batch["targets"]
            if target_idx is not None:
                targets = targets[:, target_idx].unsqueeze(1)
            all_y.append(targets.numpy())
            all_pred.append(pred.cpu().numpy())
    return np.concatenate(all_y, axis=0), np.concatenate(all_pred, axis=0)


def train_one_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_targets: int,
    target_idx: Optional[int],
) -> nn.Module:
    model = BertDocRegressor(n_targets=n_targets).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BERT_LR, weight_decay=0.01
    )
    total_steps = len(train_loader) * BERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )
    loss_fn = nn.SmoothL1Loss()

    best_rho = -1e9
    best_state = None
    for epoch in range(1, BERT_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            targets = batch["targets"].to(DEVICE)
            if target_idx is not None:
                targets = targets[:, target_idx].unsqueeze(1)
            pred = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["goldfa"].to(DEVICE),
            )
            loss = loss_fn(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        val_y, val_pred = predict_loader(model, val_loader, target_idx)
        if target_idx is None:
            rho = mae_spearman(val_y, val_pred, TARGETS)["rho_macro"]
        else:
            rho, _ = spearmanr(val_y[:, 0], val_pred[:, 0])
            rho = float(rho) if rho is not None and not np.isnan(rho) else 0.0
        print(
            f"  epoch={epoch} val_rho={rho:.6f} "
            f"({'MT' if target_idx is None else TARGETS[target_idx]})"
        )
        if rho > best_rho:
            best_rho = rho
            best_state = snapshot_state_dict(model)

    if best_state is None:
        raise RuntimeError("No best model state was selected.")
    model.load_state_dict(best_state)
    return model


def run_text_plus_goldfa(
    df_doc: pd.DataFrame,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    gold_features: Tuple[np.ndarray, np.ndarray, np.ndarray],
    multi_task: bool,
) -> Dict[str, np.ndarray]:
    variant = "mt" if multi_task else "st"
    outdir = os.path.join(OUT_DIR, f"bert_text_goldfa_{variant}")
    ensure_dir(outdir)

    tokenizer = AutoTokenizer.from_pretrained(DOC_BERT_NAME, use_fast=True)
    df_train = df_doc.loc[tr_idx].reset_index(drop=True)
    df_val = df_doc.loc[va_idx].reset_index(drop=True)
    df_test = df_doc.loc[te_idx].reset_index(drop=True)
    x_train, x_val, x_test = gold_features

    train_ds = DocBertGoldDataset(df_train, tokenizer, x_train)
    val_ds = DocBertGoldDataset(df_val, tokenizer, x_val)
    test_ds = DocBertGoldDataset(df_test, tokenizer, x_test)
    train_loader = DataLoader(train_ds, batch_size=BERT_BS, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BERT_BS, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BERT_BS, shuffle=False)

    if multi_task:
        set_seed(RANDOM_STATE)
        model = train_one_model(train_loader, val_loader, 3, None)
        y_test, pred_test = predict_loader(model, test_loader, None)
        y_val, pred_val = predict_loader(model, val_loader, None)
        save_metrics(outdir, "multi_task", y_test, pred_test, y_val, pred_val)
        return {"Y": y_test, "P": pred_test}

    pred_test = np.zeros((len(test_ds), 3), dtype=np.float32)
    pred_val = np.zeros((len(val_ds), 3), dtype=np.float32)
    y_test = df_test[TARGETS].to_numpy(np.float32)
    y_val = df_val[TARGETS].to_numpy(np.float32)

    for target_idx in range(3):
        # Gives each target a deterministic seed-42 run.
        set_seed(RANDOM_STATE)
        model = train_one_model(train_loader, val_loader, 1, target_idx)
        _, target_test_pred = predict_loader(model, test_loader, target_idx)
        _, target_val_pred = predict_loader(model, val_loader, target_idx)
        pred_test[:, target_idx] = target_test_pred[:, 0]
        pred_val[:, target_idx] = target_val_pred[:, 0]

    save_metrics(outdir, "single_task", y_test, pred_test, y_val, pred_val)
    return {"Y": y_test, "P": pred_test}


def metrics_row(metrics_path: str) -> Dict:
    with open(metrics_path, "r") as handle:
        metrics = json.load(handle)
    test = metrics["test"]
    per_target = test["per_target"]
    return {
        "rho_macro": test["rho_macro"],
        "Directness_MAE": per_target[TARGETS[0]]["MAE"],
        "Markers_MAE": per_target[TARGETS[1]]["MAE"],
        "Overall_MAE": per_target[TARGETS[2]]["MAE"],
        "rho_D": per_target[TARGETS[0]]["Spearman"],
        "rho_M": per_target[TARGETS[1]]["Spearman"],
        "rho_O": per_target[TARGETS[2]]["Spearman"],
    }


def print_row(label: str, row: Dict) -> None:
    print(
        f"{label:<24} | MAE ↓  "
        f"D:{row['Directness_MAE']:.4f} "
        f"M:{row['Markers_MAE']:.4f} "
        f"O:{row['Overall_MAE']:.4f} | "
        f"rho_D:{row['rho_D']:.4f} "
        f"rho_M:{row['rho_M']:.4f} "
        f"rho_O:{row['rho_O']:.4f}"
    )


# ==================== MAIN ====================
def main() -> None:
    ensure_dir(OUT_DIR)
    set_seed(RANDOM_STATE)

    df_doc = sanitize_dataframe(pd.read_csv(DOC_CSV))
    df_sent = sanitize_dataframe(pd.read_csv(SENT_CSV))

    required_doc = [DOC_ID_COL, DOC_TEXT_COL] + TARGETS
    missing_doc = [col for col in required_doc if col not in df_doc.columns]
    if missing_doc:
        raise ValueError(f"Document CSV is missing columns: {missing_doc}")

    if SEED_COL not in df_doc.columns:
        if SEED_COL not in df_sent.columns:
            raise ValueError(f"Neither CSV supplies the required '{SEED_COL}' column.")
        seed_map = (
            df_sent[[DOC_ID_COL, SEED_COL]]
            .dropna()
            .drop_duplicates(subset=[DOC_ID_COL, SEED_COL])
            .groupby(DOC_ID_COL)[SEED_COL]
            .agg(lambda values: values.mode().iloc[0] if len(values.mode()) else values.iloc[0])
            .reset_index()
        )
        df_doc = sanitize_dataframe(
            df_doc.merge(seed_map, on=DOC_ID_COL, how="left")
        )

    df_doc = df_doc.dropna(subset=TARGETS).reset_index(drop=True)
    if df_doc[SEED_COL].isna().any():
        raise ValueError("Some documents have no valid split seed.")

    groups = df_doc[SEED_COL].to_numpy().astype(int)
    tr_idx, va_idx, te_idx = grouped_split_70_15_15_index(
        groups, random_state=RANDOM_STATE
    )
    train_seeds = set(groups[tr_idx])
    val_seeds = set(groups[va_idx])
    test_seeds = set(groups[te_idx])
    if train_seeds & val_seeds or train_seeds & test_seeds or val_seeds & test_seeds:
        raise AssertionError("Seed leakage across train/validation/test splits.")

    split_manifest = pd.concat(
        [
            df_doc.loc[tr_idx, [DOC_ID_COL, SEED_COL]].assign(split="train"),
            df_doc.loc[va_idx, [DOC_ID_COL, SEED_COL]].assign(split="validation"),
            df_doc.loc[te_idx, [DOC_ID_COL, SEED_COL]].assign(split="test"),
        ],
        ignore_index=True,
    )
    split_manifest.to_csv(os.path.join(OUT_DIR, "split_manifest.csv"), index=False)

    y_all = df_doc[TARGETS].to_numpy(np.float32)
    y_test = y_all[te_idx]

    # Reuse the main Text-only baseline before training any oracle model.
    variants = []
    if DO_MULTI_TASK:
        reuse_main_text_baseline("mt", y_test)
        variants.append("mt")
    if DO_SINGLE_TASK:
        reuse_main_text_baseline("st", y_test)
        variants.append("st")

    ids_by_split = {
        "train": set(df_doc.loc[tr_idx, DOC_ID_COL].astype(str)),
        "validation": set(df_doc.loc[va_idx, DOC_ID_COL].astype(str)),
        "test": set(df_doc.loc[te_idx, DOC_ID_COL].astype(str)),
    }
    sent_by_split = {
        name: df_sent[df_sent[DOC_ID_COL].astype(str).isin(doc_ids)].reset_index(drop=True)
        for name, doc_ids in ids_by_split.items()
    }

    print("[GOLD FA] Building sentence-level binary GoldFA and HSPT-13 features...")
    aggregate_by_split = {}
    for name, sent_split in sent_by_split.items():
        binary = sentence_gold_binary(sent_split)
        binary.to_csv(
            os.path.join(OUT_DIR, f"goldfa_sentence_binary_{name}.csv"),
            index=False,
        )
        aggregate = aggregate_hspt13(binary)
        aggregate.to_csv(
            os.path.join(OUT_DIR, f"goldfa_hspt13_raw_{name}.csv"),
            index=False,
        )
        aggregate_by_split[name] = aggregate

    x_train_raw = align_gold_to_split(aggregate_by_split["train"], df_doc, tr_idx)
    x_val_raw = align_gold_to_split(aggregate_by_split["validation"], df_doc, va_idx)
    x_test_raw = align_gold_to_split(aggregate_by_split["test"], df_doc, te_idx)
    x_train, x_val, x_test, mu, sd = zscore_fit_transform(
        x_train_raw, x_val_raw, x_test_raw
    )

    pd.DataFrame(
        {"feature": SUM13_ORDER, "mu": mu.tolist(), "sd": sd.tolist()}
    ).to_csv(
        os.path.join(OUT_DIR, "goldfa_hspt13_zscore_params.csv"), index=False
    )
    with open(os.path.join(OUT_DIR, "goldfa_hspt13_feature_order.json"), "w") as handle:
        json.dump(SUM13_ORDER, handle, indent=2)

    gold_features = (x_train, x_val, x_test)
    if DO_MULTI_TASK:
        reuse_mt = (
            REUSE_EXISTING_GOLDFA_IF_VALID
            and can_reuse_existing_goldfa("mt", y_test)
        )
        if not reuse_mt:
            print("\n[ORACLE] Training Text + GoldFA (multi-task)...")
            run_text_plus_goldfa(
                df_doc, tr_idx, va_idx, te_idx, gold_features, multi_task=True
            )
    if DO_SINGLE_TASK:
        reuse_st = (
            REUSE_EXISTING_GOLDFA_IF_VALID
            and can_reuse_existing_goldfa("st", y_test)
        )
        if not reuse_st:
            print("\n[ORACLE] Training Text + GoldFA (single-task)...")
            run_text_plus_goldfa(
                df_doc, tr_idx, va_idx, te_idx, gold_features, multi_task=False
            )

    comparison = {
        "random_state": RANDOM_STATE,
        "text_baseline_retrained": False,
        "text_baseline_identity_verified": True,
        "text_baseline_source": MAIN_OPR_OUT_DIR,
        "goldfa_artifact_source": GOLDFA_ARTIFACT_DIR,
        "rows": {},
    }
    for variant in variants:
        text_key = f"BERT_text_{variant}"
        gold_key = f"BERT_plus_GoldFA_{variant}"
        comparison["rows"][text_key] = metrics_row(
            os.path.join(OUT_DIR, f"bert_text_{variant}", "metrics.json")
        )
        comparison["rows"][gold_key] = metrics_row(
            os.path.join(OUT_DIR, f"bert_text_goldfa_{variant}", "metrics.json")
        )

    compare_path = os.path.join(OUT_DIR, "oracle_compare_table.json")
    with open(compare_path, "w") as handle:
        json.dump(comparison, handle, indent=2)

    print("\n=== ORACLE TEST COMPARISON ===")
    print("Text rows below are byte-identical copies of the main OPR Text rows.")
    for variant in variants:
        print_row(
            f"BERT_text ({variant})",
            comparison["rows"][f"BERT_text_{variant}"],
        )
        print_row(
            f"BERT+GoldFA ({variant})",
            comparison["rows"][f"BERT_plus_GoldFA_{variant}"],
        )
    print(f"\n[DONE] Saved: {compare_path}")


if __name__ == "__main__":
    main()

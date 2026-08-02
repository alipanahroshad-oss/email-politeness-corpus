#!/usr/bin/env python3
# -*- coding: utf-8 -*-
print("seq lan req or rep bert")
import os
import re
import ast
import gc
import json
import math
import random
import shutil
import inspect
import textwrap
import itertools
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

print("SeqLabel + ASL — Sentence-ID mask — validation micro-F1 grid search [PATCHED + LOSSFP32 + LOGSIG]")

CFG = {
    "train_csv": "data/splits/train_seed42.csv",
    "val_csv":   "data/splits/val_seed42.csv",
    "test_csv":  "data/splits/test_seed42.csv",

    "TEXT_COL":     "covered_text",
    "LABEL_COL":    "GoldFaceAct",
    "EMAIL_COL":    "email_id",
    "SENTIDX_COL":  "sentence_idx",
    "ISREQ_COL":    "is_request",
    "SEED_COL":     "seed",
    "PAIR_COL":     "pair_idx",

    "filter_is_request": None,

    "LABELS": ["HNeg+","HNeg-","HPos+","HPos-","Neutral","SNeg+","SNeg-","SPos+","SPos-"],

    "model_name": "bert-base-uncased",
    "max_length": 512,

    "output_dir": "bert_seqlabel_sentid_req_or_rep_asl_gridsearch_microf1_fixed_nrowsproblem_patched_letsee",

    "lr_grid": [1e-5, 2e-5, 3e-5],
    "epochs_grid": [3, 5, 8],
    "batch_grid": [2, 4],
    "dropout_grid": [0.1, 0.2],

    "weight_decay": 0.01,
    "early_stopping_patience": 2,

    "use_pos_weight": False,
    "asl_gamma_pos": 0.0,
    "asl_gamma_neg": 4.0,
    "asl_clip": 0.05,

    "tau_grid_start": 0.05,
    "tau_grid_end": 0.95,
    "tau_grid_steps": 91,

    "seed": 42,
    "debug_sequences": 3,
    "save_test_predictions": True,
    "save_trial_models": False,
}

os.makedirs(CFG["output_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])

LABELS = CFG["LABELS"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for i,l in enumerate(LABELS)}

_TRAINER_PARAMS = set(inspect.signature(Trainer.__init__).parameters.keys())
if "processing_class" in _TRAINER_PARAMS:
    TOK_KW = "processing_class"
elif "tokenizer" in _TRAINER_PARAMS:
    TOK_KW = "tokenizer"
else:
    TOK_KW = None
print(f"[INFO] Trainer tokenizer kwarg detected: {TOK_KW}")

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    np.clip(x, -50.0, 50.0, out=x)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

def parse_labels(cell, _row_idx: int = -1) -> List[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            arr = ast.literal_eval(s)
            return [str(x).strip() for x in arr]
        except Exception as e:
            print(f"[WARN] parse_labels: literal_eval failed for row {_row_idx}: {s!r} ({e}); falling back to comma split")
    return [t.strip() for t in re.split(r'[;,]', s) if t.strip()]

def to_multi_hot(names: List[str]) -> np.ndarray:
    vec = np.zeros(len(LABELS), dtype=np.float32)
    for n in names:
        if n in label2id:
            vec[label2id[n]] = 1.0
    return vec

def load_split(csv_path: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = [cfg["TEXT_COL"], cfg["LABEL_COL"], cfg["EMAIL_COL"], cfg["SENTIDX_COL"],
            cfg["ISREQ_COL"], cfg["SEED_COL"], cfg["PAIR_COL"]]
    for c in need:
        assert c in df.columns, f"Column '{c}' not found in {csv_path}"
    if cfg["filter_is_request"] in (0, 1):
        df = df[df[cfg["ISREQ_COL"]] == cfg["filter_is_request"]].copy()
    for c in [cfg["EMAIL_COL"], cfg["SENTIDX_COL"], cfg["ISREQ_COL"], cfg["SEED_COL"], cfg["PAIR_COL"]]:
        df[c] = df[c].astype(int)
    df["gold_list"] = [parse_labels(v, i) for i, v in enumerate(df[cfg["LABEL_COL"]].tolist())]
    df["y_vec"] = df["gold_list"].map(to_multi_hot)
    unknown_rows = sum(1 for names in df["gold_list"] if any(n not in label2id for n in names))
    if unknown_rows > 0:
        print(f"[WARN] {unknown_rows} rows in {os.path.basename(csv_path)} contained at least one unknown label name (silently dropped from y_vec)")
    return df.reset_index(drop=True)

df_train = load_split(CFG["train_csv"], CFG)
df_val   = load_split(CFG["val_csv"],   CFG)
df_test  = load_split(CFG["test_csv"],  CFG)
print(f"[INFO] Loaded rows: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

def check_split_integrity(df_train, df_val, df_test, seed_col):
    s_tr = set(df_train[seed_col].unique())
    s_va = set(df_val[seed_col].unique())
    s_te = set(df_test[seed_col].unique())
    print("\n[SEED GROUPS]")
    print(f"Train ({len(s_tr)}): {sorted(list(s_tr))}")
    print(f"Val   ({len(s_va)}): {sorted(list(s_va))}")
    print(f"Test  ({len(s_te)}): {sorted(list(s_te))}")
    print("\n[Overlap checks]")
    for a,b,name in [(s_tr,s_va,"Train ∩ Val"),(s_tr,s_te,"Train ∩ Test"),(s_va,s_te,"Val ∩ Test")]:
        ov = sorted(list(a & b))
        print(f"{name} ({len(ov)}): {ov}")
    if (s_tr & s_va) or (s_tr & s_te) or (s_va & s_te):
        print("[LEAKAGE WARNING] Some seeds appear in multiple splits!")
    else:
        print("[OK] No seeds in common across splits.")

def print_is_request_counts(df_train, df_val, df_test, col):
    print("\n[TRAIN is_request counts]\n", df_train[col].value_counts(dropna=False))
    print("[VAL   is_request counts]\n", df_val[col].value_counts(dropna=False))
    print("[TEST  is_request counts]\n", df_test[col].value_counts(dropna=False))

check_split_integrity(df_train, df_val, df_test, CFG["SEED_COL"])
print_is_request_counts(df_train, df_val, df_test, CFG["ISREQ_COL"])

tok = AutoTokenizer.from_pretrained(CFG["model_name"], use_fast=True)
assert tok.is_fast, "Use a fast tokenizer."

def build_sequences_with_ranges(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    exs = []
    for eid, grp in df.sort_values([cfg["EMAIL_COL"], cfg["SENTIDX_COL"]]).groupby(cfg["EMAIL_COL"], sort=False):
        texts = grp[cfg["TEXT_COL"]].astype(str).tolist()
        labels = [np.array(v, dtype=np.float32) for v in grp["y_vec"].tolist()]
        sentence_idx_list = grp[cfg["SENTIDX_COL"]].astype(int).tolist()
        parts, ranges, pos = [], [], 0
        for i, s in enumerate(texts):
            if i > 0:
                parts.append(" ")
                pos += 1
            start = pos
            parts.append(s)
            pos += len(s)
            ranges.append((start, pos))
        concat_text = "".join(parts)
        exs.append({
            "email_id": int(eid),
            "is_request": int(grp[cfg["ISREQ_COL"]].iloc[0]),
            "seed": int(grp[cfg["SEED_COL"]].iloc[0]),
            "pair_idx": int(grp[cfg["PAIR_COL"]].iloc[0]),
            "sentence_idx_list": sentence_idx_list,
            "labels": labels,
            "concat_text": concat_text,
            "char_ranges": ranges,
        })
    return exs

train_seq = build_sequences_with_ranges(df_train, CFG)
val_seq   = build_sequences_with_ranges(df_val,   CFG)
test_seq  = build_sequences_with_ranges(df_test,  CFG)

def encode_with_sentence_ids(concat_text: str, char_ranges: List[Tuple[int,int]], tokenizer, max_length: int):
    enc = tokenizer(
        concat_text,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        padding=False,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    offsets = enc["offset_mapping"]
    sp_mask = enc["special_tokens_mask"]

    L = len(input_ids)
    token_type_ids = [0] * L

    S = len(char_ranges)
    sentence_ids = [-1] * L
    sent_token_counts = [0] * S

    for t_idx, ((a, b), is_special, m) in enumerate(zip(offsets, sp_mask, attn)):
        if m == 0 or is_special or a == b:
            continue
        for s_idx, (sa, se) in enumerate(char_ranges):
            if not (b <= sa or a >= se):
                sentence_ids[t_idx] = s_idx
                sent_token_counts[s_idx] += 1
                break

    kept_indices = [i for i, c in enumerate(sent_token_counts) if c > 0]
    if not kept_indices:
        return input_ids, attn, token_type_ids, sentence_ids, kept_indices

    remap = {old:i for i, old in enumerate(kept_indices)}
    sentence_ids = [remap[s] if s in remap else -1 for s in sentence_ids]
    return input_ids, attn, token_type_ids, sentence_ids, kept_indices

class SeqLabelSentenceIDDataset(Dataset):
    """
    [FIX-TRUNC]
    Tracks both:
      - sequences fully dropped (zero surviving sentences)
      - sequences with PARTIAL truncation (some sentences lost)
      - total sentences dropped due to truncation
    """
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, cfg: Dict[str, Any], split_name: str = ""):
        self.examples = []
        self.split_name = split_name
        self._fully_dropped_seqs = 0
        self._partial_truncated_seqs = 0
        self._sentences_dropped_total = 0
        self._sentences_input_total = 0
        self._sentences_kept_total = 0

        for ex in examples:
            n_in = len(ex["labels"])
            self._sentences_input_total += n_in

            ids, attn, tt, sids, kept = encode_with_sentence_ids(
                ex["concat_text"], ex["char_ranges"], tokenizer, cfg["max_length"]
            )
            n_kept = len(kept)
            self._sentences_kept_total += n_kept
            self._sentences_dropped_total += (n_in - n_kept)

            if n_kept == 0:
                self._fully_dropped_seqs += 1
                continue
            if n_kept < n_in:
                self._partial_truncated_seqs += 1

            labels_kept = [ex["labels"][i] for i in kept]
            sentidx_kept = [ex["sentence_idx_list"][i] for i in kept]
            self.examples.append({
                "input_ids": ids,
                "attention_mask": attn,
                "token_type_ids": tt,
                "sentence_ids": sids,
                "labels": labels_kept,
                "email_id": ex["email_id"],
                "is_request": ex["is_request"],
                "seed": ex["seed"],
                "pair_idx": ex["pair_idx"],
                "sentence_idx_list": sentidx_kept,
                "concat_text": ex["concat_text"],
            })

        if self._fully_dropped_seqs:
            print(f"[WARN][{split_name}] Dropped {self._fully_dropped_seqs} sequences with zero surviving sentences after truncation.")
        if self._partial_truncated_seqs:
            print(f"[WARN][{split_name}] {self._partial_truncated_seqs} sequences had partial sentence truncation; "
                  f"{self._sentences_dropped_total} sentences total were lost out of {self._sentences_input_total} "
                  f"({100.0 * self._sentences_dropped_total / max(1, self._sentences_input_total):.2f}%).")

    def truncation_report(self) -> Dict[str, int]:
        return {
            "split": self.split_name,
            "n_sequences_kept": len(self.examples),
            "n_sequences_fully_dropped": int(self._fully_dropped_seqs),
            "n_sequences_partial_truncation": int(self._partial_truncated_seqs),
            "sentences_input_total": int(self._sentences_input_total),
            "sentences_kept_total": int(self._sentences_kept_total),
            "sentences_dropped_total": int(self._sentences_dropped_total),
            "pct_sentences_dropped": float(
                100.0 * self._sentences_dropped_total / max(1, self._sentences_input_total)
            ),
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        labels = torch.tensor(np.stack(e["labels"], axis=0), dtype=torch.float32)
        return {
            "input_ids": torch.tensor(e["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(e["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(e["token_type_ids"], dtype=torch.long),
            "sentence_ids": torch.tensor(e["sentence_ids"], dtype=torch.long),
            "labels": labels,
        }

train_ds = SeqLabelSentenceIDDataset(train_seq, tok, CFG, split_name="train")
val_ds   = SeqLabelSentenceIDDataset(val_seq,   tok, CFG, split_name="val")
test_ds  = SeqLabelSentenceIDDataset(test_seq,  tok, CFG, split_name="test")

trunc_report = {
    "train": train_ds.truncation_report(),
    "val":   val_ds.truncation_report(),
    "test":  test_ds.truncation_report(),
    "max_length": CFG["max_length"],
    "model_name": CFG["model_name"],
}
with open(os.path.join(CFG["output_dir"], "truncation_report.json"), "w", encoding="utf-8") as f:
    json.dump(trunc_report, f, indent=2)
print(f"[INFO] Truncation report written to {os.path.join(CFG['output_dir'], 'truncation_report.json')}")

class DataCollatorSeqLabelSentenceID:
    def __init__(self, pad_token_id: int, label_dim: int):
        self.pad_token_id = pad_token_id
        self.label_dim = label_dim
    def __call__(self, batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)
        input_ids, attention_mask, token_type_ids, sentence_ids = [], [], [], []
        for x in batch:
            pad = max_len - x["input_ids"].shape[0]
            input_ids.append(F.pad(x["input_ids"], (0, pad), value=self.pad_token_id))
            attention_mask.append(F.pad(x["attention_mask"], (0, pad), value=0))
            token_type_ids.append(F.pad(x["token_type_ids"], (0, pad), value=0))
            sentence_ids.append(F.pad(x["sentence_ids"], (0, pad), value=-1))
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        token_type_ids = torch.stack(token_type_ids, dim=0)
        sentence_ids = torch.stack(sentence_ids, dim=0)

        max_S = max(x["labels"].shape[0] for x in batch)
        label_grid = []
        for x in batch:
            S = x["labels"].shape[0]
            if S < max_S:
                pad_rows = torch.full((max_S - S, self.label_dim), -1.0, dtype=torch.float32)
                label_grid.append(torch.cat([x["labels"], pad_rows], dim=0))
            else:
                label_grid.append(x["labels"])
        labels = torch.stack(label_grid, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "sentence_ids": sentence_ids,
            "labels": labels,
        }

collator = DataCollatorSeqLabelSentenceID(
    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else 0,
    label_dim=len(LABELS)
)

pos_weight_tensor = None
if CFG["use_pos_weight"]:
    y = np.stack(df_train["y_vec"].values)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).astype(np.float32)
    pos_weight_tensor = torch.tensor(pos_weight)

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gp = gamma_pos
        self.gn = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets, pos_weight=None):
        logits_f = logits.float().clamp(-30.0, 30.0)
        targets_f = targets.float()

        # Stable building blocks. [FIX-LOGSIG]
        log_p_pos = F.logsigmoid(logits_f)
        log_p_neg_full = F.logsigmoid(-logits_f)

        p = torch.sigmoid(logits_f)

        log_pos = log_p_pos
        if self.gp > 0:
            log_pos = log_pos * (1.0 - p) ** self.gp

        if self.clip is not None and self.clip > 0:
            m = (p - self.clip).clamp(min=0.0)
            log_neg = torch.log1p(-m.clamp(max=1.0 - self.eps))
            x_neg = 1.0 - m
        else:
            log_neg = log_p_neg_full
            x_neg = 1.0 - p

        if self.gn > 0:
            log_neg = log_neg * (1.0 - x_neg) ** self.gn

        loss = -(targets_f * log_pos + (1.0 - targets_f) * log_neg)
        if pos_weight is not None:
            loss = loss * (1.0 + targets_f * (pos_weight.to(loss.device).float() - 1.0))

        # [FIX-NAN] safety net.
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss.mean()

class BertSeqLabelSentenceID(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float,
                 pos_weight: torch.Tensor = None,
                 asl_gamma_pos: float = 0.0, asl_gamma_neg: float = 4.0, asl_clip: float = 0.05):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.pos_weight = pos_weight
        self.asl = AsymmetricLoss(gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip)

    @staticmethod
    def _mean_pool_by_sentence_id(H_b, attn_b, sid_b, S_b, Hdim):
        """[FIX-FP32] sum in fp32 for determinism under fp16."""
        valid = (attn_b > 0) & (sid_b >= 0)
        if valid.sum() == 0 or S_b == 0:
            return torch.zeros(S_b, Hdim, device=H_b.device, dtype=H_b.dtype)
        idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        sid = sid_b[idx]
        feats = H_b[idx, :].float()
        sums = torch.zeros(S_b, Hdim, device=H_b.device, dtype=torch.float32)
        sums.index_add_(0, sid, feats)
        counts = torch.bincount(sid, minlength=S_b).unsqueeze(1).clamp_min(1).to(torch.float32)
        out = sums / counts
        return out.to(H_b.dtype)

    def forward(self, input_ids, attention_mask, token_type_ids=None, sentence_ids=None, labels=None):
        H = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        B, _, Hdim = H.shape
        S_max = labels.shape[1] if labels is not None else int(max(int(sentence_ids.max().item()) + 1, 0))

        sent_embs = []
        for b in range(B):
            if labels is not None:
                S_b = int((labels[b, :, 0] != -1).sum().item())
            else:
                S_b = int(max(int(sentence_ids[b].max().item()) + 1, 0))
            means_b = self._mean_pool_by_sentence_id(H[b], attention_mask[b], sentence_ids[b], S_b, Hdim)
            if S_b < S_max:
                pad = torch.zeros(S_max - S_b, Hdim, device=H.device, dtype=H.dtype)
                means_b = torch.cat([means_b, pad], dim=0)
            sent_embs.append(means_b)
        sent_embs = torch.stack(sent_embs, dim=0)
        logits = self.classifier(self.dropout(sent_embs))

        loss = None
        if labels is not None:
            mask = (labels[..., 0] != -1)
            if mask.any():
                y = labels[mask, :]
                z = logits[mask, :]
                loss = self.asl(z, y, pos_weight=self.pos_weight)
            else:
                loss = torch.zeros([], device=logits.device, dtype=torch.float32)
        return {"loss": loss, "logits": logits}

def load_custom_checkpoint(model: nn.Module, checkpoint_dir: str):
    """[FIX-LOAD] tolerant load with explicit warnings on key mismatch."""
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    safe_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    elif os.path.exists(safe_path):
        if safe_load_file is None:
            raise RuntimeError(f"Found {safe_path} but safetensors is not available.")
        state_dict = safe_load_file(safe_path, device="cpu")
    else:
        files = sorted(os.listdir(checkpoint_dir)) if os.path.isdir(checkpoint_dir) else []
        raise FileNotFoundError(
            f"No checkpoint weights found in {checkpoint_dir}. "
            f"Expected pytorch_model.bin or model.safetensors. Found: {files}"
        )
    incompat = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompat, "missing_keys", []) or [])
    unexpected = list(getattr(incompat, "unexpected_keys", []) or [])
    benign_unexpected = {"_extra_state"}
    real_unexpected = [k for k in unexpected if k not in benign_unexpected]
    if missing:
        print(f"[WARN][load_custom_checkpoint] Missing keys when loading {checkpoint_dir}: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if real_unexpected:
        print(f"[WARN][load_custom_checkpoint] Unexpected keys when loading {checkpoint_dir}: {real_unexpected[:8]}{' ...' if len(real_unexpected) > 8 else ''}")
    if missing:
        param_names = {n for n, _ in model.named_parameters()}
        missing_params = [k for k in missing if k in param_names]
        if missing_params:
            raise RuntimeError(
                f"[load_custom_checkpoint] Required parameters missing from checkpoint: {missing_params[:8]}"
            )
    return model

def flatten_valid(pred_logits: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred_logits = np.asarray(pred_logits)
    labels = np.asarray(labels)
    padded_mask = (labels == -1).all(axis=-1)
    keep_mask = ~padded_mask
    probs2d = pred_logits.reshape(-1, pred_logits.shape[-1])[keep_mask.reshape(-1)]
    gold2d = labels.reshape(-1, labels.shape[-1])[keep_mask.reshape(-1)]
    gold2d = (gold2d > 0).astype(int)
    return probs2d, gold2d

def flatten_valid_np(pred_logits: np.ndarray, labels: np.ndarray):
    probs2d_raw, gold2d = flatten_valid(pred_logits, labels)
    return sigmoid_stable(probs2d_raw), gold2d

def build_valid_counts_from_labels(labels_tensor: np.ndarray) -> np.ndarray:
    L = np.asarray(labels_tensor)
    valid_mask_per_ex = ~(L == -1).all(axis=-1)
    counts = valid_mask_per_ex.sum(axis=1)
    return counts.astype(int)

def compute_micro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }

def search_best_global_tau_microf1(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    probs, gold2d = flatten_valid_np(logits, labels)
    best_tau = 0.50
    best = None
    tau_grid = np.linspace(CFG["tau_grid_start"], CFG["tau_grid_end"], CFG["tau_grid_steps"])
    for tau in tau_grid:
        pred = (probs >= float(tau)).astype(int)
        m = compute_micro_metrics(gold2d, pred)
        row = {"tau": float(tau), **m}
        if best is None:
            best_tau = float(tau)
            best = row
        else:
            if row["micro_f1"] > best["micro_f1"]:
                best_tau = float(tau); best = row
            elif math.isclose(row["micro_f1"], best["micro_f1"], rel_tol=0.0, abs_tol=1e-12):
                if row["micro_precision"] > best["micro_precision"]:
                    best_tau = float(tau); best = row
                elif math.isclose(row["micro_precision"], best["micro_precision"], rel_tol=0.0, abs_tol=1e-12):
                    if row["micro_recall"] > best["micro_recall"]:
                        best_tau = float(tau); best = row
    return best_tau, best

def compute_metrics_epoch(eval_pred):
    logits, labels = eval_pred
    best_tau, best = search_best_global_tau_microf1(logits, labels)
    return {
        "micro/f1_tau_best": best["micro_f1"],
        "micro/precision_tau_best": best["micro_precision"],
        "micro/recall_tau_best": best["micro_recall"],
        "best_tau": best_tau,
    }

class TableLoggerCallback(TrainerCallback):
    """[FIX-DEDUP] dedupe on epoch only."""
    def __init__(self):
        self.rows = []
        self.last_train_loss = None
        self._printed = False
        self._seen_epochs = set()
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = float(logs["loss"])
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        g = lambda k: metrics.get(f"eval_{k}", metrics.get(k))
        epoch = int(round(state.epoch or 0)) if state.epoch is not None else (len(self.rows)+1)
        if epoch in self._seen_epochs:
            return
        self._seen_epochs.add(epoch)
        row = {
            "epoch": epoch,
            "train_loss": self.last_train_loss,
            "val_loss": g("loss"),
            "micro/f1_tau_best": g("micro/f1_tau_best"),
            "micro/precision_tau_best": g("micro/precision_tau_best"),
            "micro/recall_tau_best": g("micro/recall_tau_best"),
            "best_tau": g("best_tau"),
        }
        self.rows.append(row)
        def fmt(x): return "—" if x is None else f"{x:.6f}"
        if not self._printed:
            print("\nEpoch\tTraining Loss\tValidation Loss\tMicro/f1_bestTau\tMicro/precision_bestTau\tMicro/recall_bestTau\tBestTau")
            self._printed = True
        print(f"{row['epoch']}\t{fmt(row['train_loss'])}\t{fmt(row['val_loss'])}\t{fmt(row['micro/f1_tau_best'])}\t{fmt(row['micro/precision_tau_best'])}\t{fmt(row['micro/recall_tau_best'])}\t{fmt(row['best_tau'])}")
    def on_train_end(self, args, state, control, **kwargs):
        if self.rows:
            out_csv = os.path.join(args.output_dir, "epoch_log_summary.csv")
            pd.DataFrame(self.rows).to_csv(out_csv, index=False)

def classwise_table(gold: np.ndarray, pred: np.ndarray, subset_name: str) -> pd.DataFrame:
    rows = []
    for j, lab in enumerate(LABELS):
        yt = gold[:, j].astype(int)
        yp = pred[:, j].astype(int)
        rows.append({
            "subset": subset_name,
            "label": lab,
            "support_gold": int(yt.sum()),
            "support_pred": int(yp.sum()),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "gold_absent": bool(yt.sum() == 0),
            "pred_absent": bool(yp.sum() == 0),
        })
    return pd.DataFrame(rows)

def debug_print_sequences(ds: SeqLabelSentenceIDDataset, n=3):
    if n <= 0:
        return
    chosen = list(range(min(n, len(ds))))
    print("\n" + "="*72)
    print(f"[DEBUG] Showing {len(chosen)} sequences. LABEL ORDER: {LABELS}")
    print("="*72)
    for i in chosen:
        ex = ds.examples[i]
        ids = ex["input_ids"]
        sids = ex["sentence_ids"]
        toks = tok.convert_ids_to_tokens(ids)
        S = len(ex["labels"])
        counts = [sum(1 for sid in sids if sid == j) for j in range(S)]
        print("-"*72)
        print(f"[Seq {i}] email_id={ex['email_id']} | is_request={ex['is_request']} | seed={ex['seed']} | pair_idx={ex['pair_idx']}")
        print("concat_text:", textwrap.shorten(ex["concat_text"], width=140, placeholder="…"))
        print("first 120 TOKENS:", " ".join(toks[:120]))
        print("sentence token counts:", counts, " (num sentences:", S, ")")
        print("sentence_idx_list:", ex["sentence_idx_list"])
    print("="*72 + "\n")

debug_print_sequences(train_ds, CFG["debug_sequences"])

def make_trainer_kwargs(extra: Dict[str, Any]) -> Dict[str, Any]:
    """[FIX-VER] inject the right tokenizer kwarg for the installed transformers."""
    if TOK_KW is not None:
        extra[TOK_KW] = tok
    return extra

grid_rows = []
best_trial = None
best_trial_dir = None
trial_id = 0

for lr, epochs, batch_size, dropout in itertools.product(
    CFG["lr_grid"], CFG["epochs_grid"], CFG["batch_grid"], CFG["dropout_grid"]
):
    trial_id += 1
    trial_name = f"trial_{trial_id:03d}_lr{lr}_ep{epochs}_bs{batch_size}_do{dropout}"
    trial_dir = os.path.join(CFG["output_dir"], "grid_trials", trial_name)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"\n[GRID] {trial_name}")

    cleanup_memory()
    model = BertSeqLabelSentenceID(
        CFG["model_name"],
        num_labels=len(LABELS),
        dropout=dropout,
        pos_weight=pos_weight_tensor if CFG["use_pos_weight"] else None,
        asl_gamma_pos=CFG["asl_gamma_pos"],
        asl_gamma_neg=CFG["asl_gamma_neg"],
        asl_clip=CFG["asl_clip"],
    )

    args = TrainingArguments(
        output_dir=trial_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=CFG["weight_decay"],
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_micro/f1_tau_best",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=CFG["seed"],
        report_to=[],
        logging_steps=50,
        remove_unused_columns=False,
        save_total_limit=2,       
        save_safetensors=False,
    )

    trainer_kwargs = make_trainer_kwargs(dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_epoch,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG["early_stopping_patience"]),
                   TableLoggerCallback()],
    ))
    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    best_checkpoint_dir = trainer.state.best_model_checkpoint or ""
    if not best_checkpoint_dir:
        raise RuntimeError(f"No best checkpoint found for {trial_name}")

    best_model = BertSeqLabelSentenceID(
        CFG["model_name"],
        num_labels=len(LABELS),
        dropout=dropout,
        pos_weight=pos_weight_tensor if CFG["use_pos_weight"] else None,
        asl_gamma_pos=CFG["asl_gamma_pos"],
        asl_gamma_neg=CFG["asl_gamma_neg"],
        asl_clip=CFG["asl_clip"],
    )
    best_model = load_custom_checkpoint(best_model, best_checkpoint_dir)

    eval_args = TrainingArguments(
        output_dir=os.path.join(trial_dir, "eval_tmp"),
        per_device_eval_batch_size=batch_size,
        report_to=[],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    eval_trainer_kwargs = make_trainer_kwargs(dict(
        model=best_model,
        args=eval_args,
        data_collator=collator,
    ))
    eval_trainer = Trainer(**eval_trainer_kwargs)

    val_out = eval_trainer.predict(val_ds)
    val_logits = val_out.predictions
    val_labels = val_out.label_ids
    val_tau_best, val_best = search_best_global_tau_microf1(val_logits, val_labels)

    trial_row = {
        "trial_id": trial_id,
        "trial_name": trial_name,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout": dropout,
        "weight_decay": CFG["weight_decay"],
        "checkpoint_metric_micro_f1_best_tau": float(trainer.state.best_metric) if trainer.state.best_metric is not None else np.nan,
        "val_best_tau": float(val_tau_best),
        "val_micro_f1_best_tau": float(val_best["micro_f1"]),
        "val_micro_precision_best_tau": float(val_best["micro_precision"]),
        "val_micro_recall_best_tau": float(val_best["micro_recall"]),
        "best_checkpoint_dir": best_checkpoint_dir,
    }
    grid_rows.append(trial_row)

    if best_trial is None:
        best_trial = trial_row
        best_trial_dir = trial_dir
    else:
        if trial_row["val_micro_f1_best_tau"] > best_trial["val_micro_f1_best_tau"]:
            best_trial = trial_row; best_trial_dir = trial_dir
        elif math.isclose(trial_row["val_micro_f1_best_tau"], best_trial["val_micro_f1_best_tau"], rel_tol=0.0, abs_tol=1e-12):
            if trial_row["val_micro_precision_best_tau"] > best_trial["val_micro_precision_best_tau"]:
                best_trial = trial_row; best_trial_dir = trial_dir
            elif math.isclose(trial_row["val_micro_precision_best_tau"], best_trial["val_micro_precision_best_tau"], rel_tol=0.0, abs_tol=1e-12):
                if trial_row["val_micro_recall_best_tau"] > best_trial["val_micro_recall_best_tau"]:
                    best_trial = trial_row; best_trial_dir = trial_dir

    pd.DataFrame(grid_rows).to_csv(
        os.path.join(CFG["output_dir"], "grid_search_results_partial.csv"),
        index=False,
    )

    del trainer, eval_trainer, model, best_model, val_out, val_logits, val_labels
    cleanup_memory()

grid_df = pd.DataFrame(grid_rows).sort_values(
    ["val_micro_f1_best_tau", "val_micro_precision_best_tau", "val_micro_recall_best_tau"],
    ascending=[False, False, False]
).reset_index(drop=True)
grid_df.to_csv(os.path.join(CFG["output_dir"], "grid_search_results.csv"), index=False)

with open(os.path.join(CFG["output_dir"], "best_config.json"), "w", encoding="utf-8") as f:
    json.dump(best_trial, f, indent=2)

print(f"\n[BEST] {best_trial['trial_name']}")
print(json.dumps(best_trial, indent=2))

best_ckpt = best_trial["best_checkpoint_dir"]
best_tau = float(best_trial["val_best_tau"])
best_bs = int(best_trial["batch_size"])
best_dropout = float(best_trial["dropout"])

best_model = BertSeqLabelSentenceID(
    CFG["model_name"],
    num_labels=len(LABELS),
    dropout=best_dropout,
    pos_weight=pos_weight_tensor if CFG["use_pos_weight"] else None,
    asl_gamma_pos=CFG["asl_gamma_pos"],
    asl_gamma_neg=CFG["asl_gamma_neg"],
    asl_clip=CFG["asl_clip"],
)
best_model = load_custom_checkpoint(best_model, best_ckpt)

final_args = TrainingArguments(
    output_dir=os.path.join(CFG["output_dir"], "final_eval_tmp"),
    per_device_eval_batch_size=best_bs,
    report_to=[],
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
)
final_trainer_kwargs = make_trainer_kwargs(dict(
    model=best_model,
    args=final_args,
    data_collator=collator,
))
final_trainer = Trainer(**final_trainer_kwargs)

test_out = final_trainer.predict(test_ds)
test_logits = test_out.predictions
test_labels = test_out.label_ids

def flatten_pred_gold_isreq_per_example(ds: SeqLabelSentenceIDDataset, logits, tau: float):
    logits = np.asarray(logits)

    preds_list, golds_list, isreq_list, meta_rows = [], [], [], []

    for i, ex in enumerate(ds.examples):
        S_b = len(ex["labels"])
        if S_b == 0:
            continue

        probs_b = sigmoid_stable(logits[i][:S_b])
        pred_b = (probs_b >= tau).astype(int)
        gold_b = (np.stack(ex["labels"][:S_b]) > 0.5).astype(int)

        is_req_b = np.array([int(ex["is_request"])] * S_b, dtype=int)

        preds_list.append(pred_b)
        golds_list.append(gold_b)
        isreq_list.append(is_req_b)

        for j in range(S_b):
            meta_rows.append({
                "email_id": int(ex["email_id"]),
                "is_request": int(ex["is_request"]),
                "seed": int(ex["seed"]),
                "pair_idx": int(ex["pair_idx"]),
                "sentence_idx": int(ex["sentence_idx_list"][j]),
            })

    pred2d = np.vstack(preds_list) if preds_list else np.zeros((0, len(LABELS)), dtype=int)
    gold2d = np.vstack(golds_list) if golds_list else np.zeros((0, len(LABELS)), dtype=int)
    is_req_flags = np.concatenate(isreq_list) if isreq_list else np.zeros((0,), dtype=int)
    meta_df = pd.DataFrame(meta_rows)

    assert pred2d.shape == gold2d.shape
    assert pred2d.shape[0] == is_req_flags.shape[0] == len(meta_df)

    return pred2d, gold2d, is_req_flags, meta_df


pred_all, gold_all, is_req_flags, meta_df = flatten_pred_gold_isreq_per_example(
    test_ds, test_logits, best_tau
)

diag = {
    "df_test_rows": int(len(df_test)),
    "df_test_request_rows": int((df_test[CFG["ISREQ_COL"]].astype(int) == 1).sum()),
    "df_test_reply_rows": int((df_test[CFG["ISREQ_COL"]].astype(int) == 0).sum()),
    "test_ds_sequences": int(len(test_ds.examples)),
    "evaluated_sentence_rows": int(len(gold_all)),
    "evaluated_request_rows": int((is_req_flags == 1).sum()),
    "evaluated_reply_rows": int((is_req_flags == 0).sum()),
    "best_tau": float(best_tau),
    "test_truncation_report": test_ds.truncation_report(),
}
with open(os.path.join(CFG["output_dir"], "row_count_diagnostics.json"), "w", encoding="utf-8") as f:
    json.dump(diag, f, indent=2)

meta_df.to_csv(os.path.join(CFG["output_dir"], "test_eval_sentence_metadata.csv"), index=False)

mask_req = is_req_flags == 1
mask_rep = is_req_flags == 0

rows = []
class_rows = []
for subset_name, mask in [
    ("OVERALL", np.ones(len(gold_all), dtype=bool)),
    ("REQUEST", mask_req),
    ("REPLY", mask_rep),
]:
    y_true = gold_all[mask]
    y_pred = pred_all[mask]
    m = compute_micro_metrics(y_true, y_pred)
    rows.append({
        "subset": subset_name,
        "n_rows": int(mask.sum()),
        "tau_used": float(best_tau),
        "micro_f1": float(m["micro_f1"]),
        "micro_precision": float(m["micro_precision"]),
        "micro_recall": float(m["micro_recall"]),
    })
    class_rows.append(classwise_table(y_true, y_pred, subset_name))

test_table = pd.DataFrame(rows)
class_df = pd.concat(class_rows, ignore_index=True)

test_table.to_csv(os.path.join(CFG["output_dir"], "test_micro_metrics_overall_request_reply.csv"), index=False)
class_df.to_csv(os.path.join(CFG["output_dir"], "test_classwise_f1_support_for_appendix.csv"), index=False)

tex_lines = [
    "\\begin{tabular}{lcccc}",
    "\\toprule",
    "\\textbf{Subset} & \\textbf{N} & \\textbf{mF1} & \\textbf{Micro-P} & \\textbf{Micro-R} \\\\",
    "\\midrule",
]
for _, r in test_table.iterrows():
    tex_lines.append(
        f"{r['subset'].title()} & {int(r['n_rows'])} & {r['micro_f1']:.3f} & {r['micro_precision']:.3f} & {r['micro_recall']:.3f} \\\\"
    )
tex_lines += ["\\bottomrule", "\\end{tabular}"]
with open(os.path.join(CFG["output_dir"], "test_micro_metrics_overall_request_reply.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(tex_lines) + "\n")

if CFG["save_test_predictions"]:
    rows_pred = []
    logits_np = np.asarray(test_logits)

    for i, ex in enumerate(test_ds.examples):
        S_b = len(ex["labels"])
        if S_b == 0:
            continue

        probs_S = sigmoid_stable(logits_np[i][:S_b])
        pred_S = (probs_S >= best_tau).astype(int)
        gold_S = (np.stack(ex["labels"][:S_b]) > 0.5).astype(int)

        S_ok = min(gold_S.shape[0], probs_S.shape[0], len(ex["sentence_idx_list"]))

        for j in range(S_ok):
            row = {
                "email_id": int(ex["email_id"]),
                "is_request": int(ex["is_request"]),
                "seed": int(ex["seed"]),
                "pair_idx": int(ex["pair_idx"]),
                "sentence_idx": int(ex["sentence_idx_list"][j]),
                "gold_labels": ",".join([l for k, l in enumerate(LABELS) if gold_S[j][k] == 1]),
                "best_val_tau": float(best_tau),
                f"pred_labels_tau_{best_tau:.2f}": ",".join([l for k, l in enumerate(LABELS) if pred_S[j][k] == 1]),
            }
            for k, lbl in enumerate(LABELS):
                row[f"pred_prob_{lbl}"] = float(probs_S[j][k])
            rows_pred.append(row)

    out_csv = os.path.join(CFG["output_dir"], "test_predictions_per_sentence_best_validation_tau.csv")
    pd.DataFrame(rows_pred).to_csv(out_csv, index=False)

manifest = {
    "selection_metric_for_hyperparameters": "validation micro-F1",
    "checkpoint_metric_within_each_run": "validation micro-F1 at best validation-learned tau",
    "threshold_learning": "best global validation tau maximizing validation micro-F1",
    "best_trial": best_trial,
    "truncation_report": trunc_report,
}
with open(os.path.join(CFG["output_dir"], "run_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
with open(os.path.join(CFG["output_dir"], "run_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

if not CFG["save_trial_models"]:
    trial_root = os.path.join(CFG["output_dir"], "grid_trials")
    if os.path.isdir(trial_root):
        for name in os.listdir(trial_root):
            p = os.path.join(trial_root, name)
            if os.path.abspath(p) != os.path.abspath(best_trial_dir):
                shutil.rmtree(p, ignore_errors=True)

print("[DONE] Wrote:")
print(" - grid_search_results.csv")
print(" - best_config.json")
print(" - test_micro_metrics_overall_request_reply.csv")
print(" - test_micro_metrics_overall_request_reply.tex")
print(" - test_classwise_f1_support_for_appendix.csv")
print(" - test_predictions_per_sentence_best_validation_tau.csv")
print(" - truncation_report.json")
print(" - row_count_diagnostics.json")
print(" - run_manifest.json")

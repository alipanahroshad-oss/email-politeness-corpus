#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# FAC_seqlabel_req_or_rep.py
print("SeqLabel + ASL req or rep")

import os, re, ast, json, math, random, textwrap, inspect, gc
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer, AutoModel, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
)
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

CFG = {
    "train_csv": "data/splits/train_seed42.csv",
    "val_csv":   "data/splits/val_seed42.csv",
    "test_csv":  "data/splits/test_seed42.csv",

    # Column names:
    "TEXT_COL":     "covered_text",
    "LABEL_COL":    "GoldFaceAct",
    "EMAIL_COL":    "email_id",
    "SENTIDX_COL":  "sentence_idx",
    "ISREQ_COL":    "is_request",   # 1=request, 0=reply
    "SEED_COL":     "seed",
    "PAIR_COL":     "pair_idx",     # request & reply share the same pair_idx

    # Optional filter for training: 1=requests only; 0=replies only; None=both
    "filter_is_request": None,

    "LABELS": ["HNeg+","HNeg-","HPos+","HPos-","Neutral","SNeg+","SNeg-","SPos+","SPos-"],

    # Model & tokenization
    "model_name": "bert-base-uncased",
    "max_length": 512,

    # Training
    "output_dir": os.environ.get(
    "OUTDIR",
    "bert_seqlabel_sentid_req_or_rep_asl"
),
    "epochs": 5,
    "batch_size": 4,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "early_stopping_patience": 2,

    # ASL
    "use_pos_weight": False,
    "asl_gamma_pos": 0.0,
    "asl_gamma_neg": 4.0,
    "asl_clip": 0.05,

    "debug_sequences": 3,

    "save_test_predictions": True,

    "seed": int(os.environ.get("RUN_SEED", "42")),

    "FORCE_TAU_F1": 0.60,         # value used for fixed-τ test eval & per-reply F1
    "USE_VAL_TAU": False,         # if True, val-learned τ is primary in the test report
    "TAU_GRID_START": 0.05,
    "TAU_GRID_END":   0.95,
    "TAU_GRID_STEPS": 91,

    "MODEL_TAG": "SequenceLabeling",
    "USES_REQUEST_CONTEXT": False,
    "COMPARE_WITH_CSV": "",
    "DOC_SCORES_CSV": "",
    "DOC_SCORE_COL": "overall_politeness",
}

os.makedirs(CFG["output_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
if torch.cuda.is_available(): torch.cuda.manual_seed_all(CFG["seed"])
print(f"[INFO] Training seed = {CFG['seed']}")

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
    x = np.clip(x, -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

# Labels / parsing
LABELS = CFG["LABELS"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for i,l in enumerate(LABELS)}

def parse_labels(cell) -> List[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)): return []
    s = str(cell).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            arr = ast.literal_eval(s)
            return [str(x).strip() for x in arr]
        except Exception:
            pass
    return [t.strip() for t in re.split(r'[;,]', s) if t.strip()]

def to_multi_hot(names: List[str]) -> np.ndarray:
    vec = np.zeros(len(LABELS), dtype=np.float32)
    for n in names:
        if n in label2id: vec[label2id[n]] = 1.0
    return vec

# Load splits
def load_split(csv_path: str, CFG):
    df = pd.read_csv(csv_path)
    need = [CFG["TEXT_COL"], CFG["LABEL_COL"], CFG["EMAIL_COL"], CFG["SENTIDX_COL"],
            CFG["ISREQ_COL"], CFG["SEED_COL"], CFG["PAIR_COL"]]
    for c in need: assert c in df.columns, f"Column '{c}' not found in {csv_path}."

    if CFG["filter_is_request"] in (0,1):
        df = df[df[CFG["ISREQ_COL"]] == CFG["filter_is_request"]].copy()

    for c in [CFG["EMAIL_COL"], CFG["SENTIDX_COL"], CFG["ISREQ_COL"], CFG["SEED_COL"], CFG["PAIR_COL"]]:
        df[c] = df[c].astype(int)

    df["gold_list"] = df[CFG["LABEL_COL"]].map(parse_labels)
    df["y_vec"]     = df["gold_list"].map(to_multi_hot)
    return df.reset_index(drop=True)

df_train = load_split(CFG["train_csv"], CFG)
df_val   = load_split(CFG["val_csv"],   CFG)
df_test  = load_split(CFG["test_csv"],  CFG)

print("[INFO] Loaded rows: train={}, val={}, test={}".format(len(df_train), len(df_val), len(df_test)))

# Integrity checks
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
    if (s_tr & s_va) or (s_tr & s_te) or (s_va & s_te): print("[LEAKAGE WARNING] Some seeds appear in multiple splits!")
    else: print("[OK] No seeds in common across splits.")

def print_is_request_counts(df_train, df_val, df_test, col):
    print("\n[TRAIN is_request counts]\n", df_train[col].value_counts(dropna=False))
    print("[VAL   is_request counts]\n", df_val[col].value_counts(dropna=False))
    print("[TEST  is_request counts]\n", df_test[col].value_counts(dropna=False))

check_split_integrity(df_train, df_val, df_test, CFG["SEED_COL"])
print_is_request_counts(df_train, df_val, df_test, CFG["ISREQ_COL"])

for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    uniqs = np.unique(np.concatenate(df["y_vec"].values)).tolist()
    print(f"[SANITY:{name}] unique label values: {uniqs}")

# Tokenizer 
tok = AutoTokenizer.from_pretrained(CFG["model_name"], use_fast=True)
assert tok.is_fast, "Use a *fast* tokenizer."

# Build per-email sequences + char ranges
def build_sequences_with_ranges(df: pd.DataFrame, CFG) -> List[Dict[str, Any]]:
    exs = []
    for eid, grp in df.sort_values([CFG["EMAIL_COL"], CFG["SENTIDX_COL"]]).groupby(CFG["EMAIL_COL"], sort=False):
        texts = grp[CFG["TEXT_COL"]].astype(str).tolist()
        labels = [np.array(v, dtype=np.float32) for v in grp["y_vec"].tolist()]
        sentence_idx_list = grp[CFG["SENTIDX_COL"]].astype(int).tolist()
        parts, ranges, pos = [], [], 0
        for i, s in enumerate(texts):
            if i > 0:
                parts.append(" "); pos += 1
            start = pos
            parts.append(s); pos += len(s)
            ranges.append((start, pos))
        concat_text = "".join(parts)
        exs.append({
            "email_id": int(eid),
            "is_request": int(grp[CFG["ISREQ_COL"]].iloc[0]),
            "seed": int(grp[CFG["SEED_COL"]].iloc[0]),
            "pair_idx": int(grp[CFG["PAIR_COL"]].iloc[0]),
            "sentence_idx_list": sentence_idx_list,
            "texts": texts,
            "labels": labels,
            "concat_text": concat_text,
            "char_ranges": ranges,
        })
    return exs

train_seq = build_sequences_with_ranges(df_train, CFG)
val_seq   = build_sequences_with_ranges(df_val,   CFG)
test_seq  = build_sequences_with_ranges(df_test,  CFG)

# Encode with sentence_ids
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
    attn      = enc["attention_mask"]
    offsets   = enc["offset_mapping"]
    sp_mask   = enc["special_tokens_mask"]
    L = len(input_ids)
    token_type_ids = [0] * L

    S = len(char_ranges)
    sentence_ids = [-1] * L
    sent_token_counts = [0] * S

    for t_idx, ((a,b), is_special, m) in enumerate(zip(offsets, sp_mask, attn)):
        if m == 0 or is_special or a == b:
            continue
        for s_idx, (sa,se) in enumerate(char_ranges):
            if not (b <= sa or a >= se):
                sentence_ids[t_idx] = s_idx
                sent_token_counts[s_idx] += 1
                break

    kept_indices = [i for i,c in enumerate(sent_token_counts) if c > 0]
    if not kept_indices:
        return input_ids, attn, token_type_ids, sentence_ids, kept_indices
    remap = {old:i for i,old in enumerate(kept_indices)}
    sentence_ids = [remap[s] if s in remap else -1 for s in sentence_ids]
    return input_ids, attn, token_type_ids, sentence_ids, kept_indices

class SeqLabelSentenceIDDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, cfg):
        self.cfg = cfg
        self.tok = tokenizer
        self.examples = []
        self._skipped = 0
        for ex in examples:
            ids, attn, tt, sids, kept = encode_with_sentence_ids(ex["concat_text"], ex["char_ranges"], tokenizer, cfg["max_length"])
            if len(kept) == 0:
                self._skipped += 1
                continue
            labels_kept  = [ex["labels"][i] for i in kept]
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
        if self._skipped:
            print(f"[WARN] Dropped {self._skipped} sequences with zero surviving sentences after truncation.")

    def __len__(self): return len(self.examples)

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

train_ds = SeqLabelSentenceIDDataset(train_seq, tok, CFG)
val_ds   = SeqLabelSentenceIDDataset(val_seq,   tok, CFG)
test_ds  = SeqLabelSentenceIDDataset(test_seq,  tok, CFG)

class DataCollatorSeqLabelSentenceID:
    def __init__(self, pad_token_id: int, label_dim: int):
        self.pad_token_id = pad_token_id
        self.label_dim = label_dim
    def __call__(self, batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)
        input_ids, attention_mask, token_type_ids, sentence_ids = [], [], [], []
        for x in batch:
            pad = max_len - x["input_ids"].shape[0]
            input_ids.append(      F.pad(x["input_ids"],      (0,pad), value=self.pad_token_id))
            attention_mask.append( F.pad(x["attention_mask"], (0,pad), value=0))
            token_type_ids.append( F.pad(x["token_type_ids"], (0,pad), value=0))
            sentence_ids.append(   F.pad(x["sentence_ids"],   (0,pad), value=-1))
        input_ids      = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        token_type_ids = torch.stack(token_type_ids, dim=0)
        sentence_ids   = torch.stack(sentence_ids, dim=0)

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

# Optional pos_weight for ASL
pos_weight_tensor = None
if CFG["use_pos_weight"]:
    y = np.stack(df_train["y_vec"].values)
    pos = y.sum(axis=0); neg = y.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).astype(np.float32)
    pos_weight_tensor = torch.tensor(pos_weight)
    print("[INFO] pos_weight:", {lbl: float(round(w,2)) for lbl,w in zip(LABELS, pos_weight)})

# ASL 
class AsymmetricLoss(nn.Module):
    
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gp, self.gn, self.clip, self.eps = gamma_pos, gamma_neg, clip, eps

    def forward(self, logits, targets, pos_weight=None):
        logits_f = logits.float().clamp(-30.0, 30.0)
        targets_f = targets.float()

        log_p_pos = F.logsigmoid(logits_f)
        log_p_neg_full = F.logsigmoid(-logits_f)

        p = torch.sigmoid(logits_f)

        # Positive branch: -(1 - p)^gp * log(sigmoid(x))
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

        # [FIX-NAN]
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss.mean()

# Model
class BertSeqLabelSentenceID(nn.Module):
    def __init__(self, model_name: str, num_labels: int,
                 pos_weight: torch.Tensor = None,
                 asl_gamma_pos: float = 0.0, asl_gamma_neg: float = 4.0, asl_clip: float = 0.05):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
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
        H = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        B, L, Hdim = H.shape
        S_max = labels.shape[1] if labels is not None else (int(sentence_ids.max().item()) + 1)

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

model = BertSeqLabelSentenceID(
    CFG["model_name"],
    num_labels=len(LABELS),
    pos_weight=pos_weight_tensor if CFG["use_pos_weight"] else None,
    asl_gamma_pos=CFG["asl_gamma_pos"],
    asl_gamma_neg=CFG["asl_gamma_neg"],
    asl_clip=CFG["asl_clip"],
)

# Flatten & metrics
def flatten_valid(pred_logits: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred_logits = np.asarray(pred_logits)
    labels = np.asarray(labels)
    padded_mask = (labels == -1).all(axis=-1)
    keep_mask = ~padded_mask
    probs2d = pred_logits.reshape(-1, pred_logits.shape[-1])[keep_mask.reshape(-1)]
    gold2d  = labels.reshape(-1, labels.shape[-1])[keep_mask.reshape(-1)]
    gold2d = (gold2d > 0).astype(int)
    return probs2d, gold2d

def metrics_from_bin(gold: np.ndarray, pred: np.ndarray) -> Dict[str,float]:
    hamming = np.not_equal(pred, gold).mean()
    return {
        "micro/f1":        f1_score(gold, pred, average="micro",  zero_division=0),
        "macro/f1":        f1_score(gold, pred, average="macro",  zero_division=0),
        "micro/precision": precision_score(gold, pred, average="micro", zero_division=0),
        "micro/recall":    recall_score(gold, pred, average="micro",  zero_division=0),
        "jaccard/micro":   jaccard_score(gold, pred, average="micro",  zero_division=0),
        "jaccard/macro":   jaccard_score(gold, pred, average="macro",  zero_division=0),
        "jaccard/samples": jaccard_score(gold, pred, average="samples", zero_division=0),
        "hamming_loss":    float(hamming),
        "avg_true_k":      float(gold.sum(axis=1).mean()),
        "avg_pred_k":      float(pred.sum(axis=1).mean()),
    }

def flatten_valid_np(pred_logits: np.ndarray, labels: np.ndarray):
    padded = (labels == -1).all(axis=-1)
    keep = ~padded
    probs2d = pred_logits.reshape(-1, pred_logits.shape[-1])[keep.reshape(-1)]
    gold2d  = labels.reshape(-1, labels.shape[-1])[keep.reshape(-1)]
    gold2d = (gold2d > 0).astype(int)
    return sigmoid_stable(probs2d), gold2d

def search_global_tau_for_microf1(logits, labels):
    """[FIX-TAU] full fine-grained tau search (matches grid-search scripts)."""
    probs, gold2d = flatten_valid_np(logits, labels)
    grid = np.linspace(CFG["TAU_GRID_START"], CFG["TAU_GRID_END"], CFG["TAU_GRID_STEPS"])
    best_tau, best_m, best_metrics = 0.5, -1.0, {}
    for tau in grid:
        pred = (probs >= float(tau)).astype(int)
        m = metrics_from_bin(gold2d, pred)
        # tie-break: F1, then precision, then recall
        if m["micro/f1"] > best_m + 1e-12:
            best_m, best_tau, best_metrics = m["micro/f1"], float(tau), m
        elif math.isclose(m["micro/f1"], best_m, abs_tol=1e-12):
            if m["micro/precision"] > best_metrics.get("micro/precision", -1):
                best_tau, best_metrics = float(tau), m
            elif math.isclose(m["micro/precision"], best_metrics.get("micro/precision", -1), abs_tol=1e-12):
                if m["micro/recall"] > best_metrics.get("micro/recall", -1):
                    best_tau, best_metrics = float(tau), m
    return best_tau, best_metrics

def search_global_tau_for_jaccard(logits, labels):
    probs, gold2d = flatten_valid_np(logits, labels)
    grid = np.linspace(CFG["TAU_GRID_START"], CFG["TAU_GRID_END"], CFG["TAU_GRID_STEPS"])
    best_tau, best_m, best_metrics = 0.5, -1.0, {}
    for tau in grid:
        pred = (probs >= float(tau)).astype(int)
        m = metrics_from_bin(gold2d, pred)
        if m["jaccard/samples"] > best_m + 1e-12:
            best_m, best_tau, best_metrics = m["jaccard/samples"], float(tau), m
    return best_tau, best_metrics

def per_label_scut_f1(logits, labels):
    probs, gold2d = flatten_valid_np(logits, labels)
    C = probs.shape[1]; grid = np.linspace(CFG["TAU_GRID_START"], CFG["TAU_GRID_END"], CFG["TAU_GRID_STEPS"])
    tau_vec = np.zeros(C, np.float32)
    for c in range(C):
        y, p = gold2d[:,c], probs[:,c]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(y, (p>=t).astype(int), zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, float(t)
        tau_vec[c] = best_t
    pred = (probs >= tau_vec.reshape(1,-1)).astype(int)
    return tau_vec, metrics_from_bin(gold2d, pred)

def pcut_match_cardinality(logits, labels):
    probs, gold2d = flatten_valid_np(logits, labels)
    target_k = gold2d.sum(axis=1).mean()
    grid = np.linspace(CFG["TAU_GRID_START"], CFG["TAU_GRID_END"], CFG["TAU_GRID_STEPS"])
    best_tau, best_gap, best_metrics = 0.5, 1e9, {}
    for tau in grid:
        pred = (probs >= tau).astype(int)
        gap = abs(pred.sum(axis=1).mean() - target_k)
        if gap < best_gap: best_tau, best_gap, best_metrics = float(tau), gap, metrics_from_bin(gold2d, pred)
    return best_tau, best_metrics

def eval_with_tau(logits, labels, tau: Union[float, np.ndarray]):
    probs, gold2d = flatten_valid_np(logits, labels)
    if isinstance(tau, np.ndarray):
        pred = (probs >= tau.reshape(1,-1)).astype(int)
    else:
        pred = (probs >= float(tau)).astype(int)
    return metrics_from_bin(gold2d, pred), pred, gold2d, probs

# Epoch metric (τ=0.50, for progress only)
def training_metrics_fixed_tau(eval_pred):
    logits, labels = eval_pred
    probs2d, gold2d = flatten_valid(logits, labels)
    probs = sigmoid_stable(probs2d)
    preds = (probs >= 0.50).astype(int)
    return {
        "micro/f1":        f1_score(gold2d, preds, average="micro",  zero_division=0),
        "macro/f1":        f1_score(gold2d, preds, average="macro",  zero_division=0),
        "micro/precision": precision_score(gold2d, preds, average="micro", zero_division=0),
        "micro/recall":    recall_score(gold2d, preds, average="micro",  zero_division=0),
        "jaccard/micro":   jaccard_score(gold2d, preds, average="micro",  zero_division=0),
        "jaccard/macro":   jaccard_score(gold2d, preds, average="macro",  zero_division=0),
        "jaccard/samples": jaccard_score(gold2d, preds, average="samples", zero_division=0),
    }

class TableLoggerCallback(TrainerCallback):
    def __init__(self): self.rows=[]; self.last_train_loss=None; self._printed=False; self._seen=set()
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs: self.last_train_loss = float(logs["loss"])
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        g = lambda k: metrics.get(f"eval_{k}", metrics.get(k))
        epoch = int(round(state.epoch or 0)) if state.epoch is not None else (len(self.rows)+1)
        if epoch in self._seen: return
        self._seen.add(epoch)
        row = {
            "epoch": epoch, "train_loss": self.last_train_loss, "val_loss": g("loss"),
            "micro/f1": g("micro/f1"), "macro/f1": g("macro/f1"),
            "micro/precision": g("micro/precision"), "micro/recall": g("micro/recall"),
            "jaccard/micro": g("jaccard/micro"), "jaccard/macro": g("jaccard/macro"),
            "jaccard/samples": g("jaccard/samples"),
        }
        self.rows.append(row)
        def fmt(x): return "—" if x is None else f"{x:.6f}"
        if not self._printed:
            print("\nEpoch\tTraining Loss\tValidation Loss\tMicro/f1\tMacro/f1\tMicro/precision\tMicro/recall\tJaccard/micro\tJaccard/macro\tJaccard/samples")
            self._printed=True
        print(f"{row['epoch']}\t{fmt(row['train_loss'])}\t{fmt(row['val_loss'])}\t{fmt(row['micro/f1'])}\t{fmt(row['macro/f1'])}\t{fmt(row['micro/precision'])}\t{fmt(row['micro/recall'])}\t{fmt(row['jaccard/micro'])}\t{fmt(row['jaccard/macro'])}\t{fmt(row['jaccard/samples'])}")
    def on_train_end(self, args, state, control, **kwargs):
        if self.rows:
            out_csv = os.path.join(args.output_dir, "epoch_log_summary.csv")
            pd.DataFrame(self.rows).to_csv(out_csv, index=False)
            print(f"[LOG] Wrote epoch summary → {out_csv}")

# Debug printer
def debug_print_sequences(ds: SeqLabelSentenceIDDataset, n=3):
    if n <= 0: return
    chosen = list(range(min(n, len(ds))))
    print("\n" + "="*72)
    print(f"[DEBUG] Showing {len(chosen)} sequences. LABEL ORDER: {LABELS}")
    print("="*72)
    for i in chosen:
        ex = ds.examples[i]
        ids = ex["input_ids"]; sids = ex["sentence_ids"]
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

def make_trainer_kwargs(extra: Dict[str, Any]) -> Dict[str, Any]:
    if TOK_KW is not None:
        extra[TOK_KW] = tok
    return extra

# Training
args = TrainingArguments(
    output_dir=CFG["output_dir"],
    per_device_train_batch_size=CFG["batch_size"],
    per_device_eval_batch_size=CFG["batch_size"],
    learning_rate=CFG["lr"],
    weight_decay=CFG["weight_decay"],
    num_train_epochs=CFG["epochs"],
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="micro/f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    seed=CFG["seed"],
    report_to=[],
    logging_steps=50,
    remove_unused_columns=False,
    save_total_limit=2,           
    save_safetensors=False,
)

trainer = Trainer(**make_trainer_kwargs(dict(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    compute_metrics=training_metrics_fixed_tau,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG["early_stopping_patience"]),
               TableLoggerCallback()],
)))

debug_print_sequences(train_ds, n=CFG["debug_sequences"])

print("[INFO] Starting training…")
trainer.train()

# VAL — compute multi-τ strategies, then pick primary per [FIX-TAU]
print("[INFO] Evaluating on VAL (multi τ strategies)…")
val_out = trainer.predict(val_ds)
val_logits = val_out.predictions
val_labels = val_out.label_ids

val_tau_f1_learned, val_f1_metrics = search_global_tau_for_microf1(val_logits, val_labels)
tau_jacc,           val_jacc_metrics = search_global_tau_for_jaccard(val_logits, val_labels)
tau_vec,            val_scut         = per_label_scut_f1(val_logits, val_labels)
tau_pcut,           val_pcut         = pcut_match_cardinality(val_logits, val_labels)

tau_f1_forced = float(CFG["FORCE_TAU_F1"])

# Primary τ for TEST report:
if CFG["USE_VAL_TAU"]:
    tau_primary = val_tau_f1_learned
    tau_primary_name = "Val-learned τ@F1"
else:
    tau_primary = tau_f1_forced
    tau_primary_name = f"Forced τ@F1={tau_f1_forced:.2f}"

print(f"[VAL][Val-learned τ@F1] τ={val_tau_f1_learned:.2f}  metrics=")
print(json.dumps(eval_with_tau(val_logits, val_labels, val_tau_f1_learned)[0], indent=2))
print(f"[VAL][Forced  τ@F1] τ={tau_f1_forced:.2f}  metrics=")
print(json.dumps(eval_with_tau(val_logits, val_labels, tau_f1_forced)[0], indent=2))
print(f"[VAL][Global τ@Jaccard] τ={tau_jacc:.2f}  metrics=")
print(json.dumps(val_jacc_metrics, indent=2))
print(f"[VAL][Per-label SCut-F1] τ_vec={[float(round(x,2)) for x in tau_vec.tolist()]}  metrics=")
print(json.dumps(val_scut, indent=2))
print(f"[VAL][PCut] τ={tau_pcut:.2f}  metrics=")
print(json.dumps(val_pcut, indent=2))
print(f"\n[INFO] PRIMARY τ for TEST report: {tau_primary_name} (τ={tau_primary:.2f})")

# TEST — report BOTH forced and val-learned τ for completeness
test_out = trainer.predict(test_ds)
test_logits = test_out.predictions
test_labels = test_out.label_ids

print(f"\n[TEST][Val-learned τ@F1={val_tau_f1_learned:.2f}]")
test_metrics_val_tau, _, _, _ = eval_with_tau(test_logits, test_labels, val_tau_f1_learned)
print(json.dumps(test_metrics_val_tau, indent=2))

print(f"\n[TEST][Forced τ@F1={tau_f1_forced:.2f}]")
test_metrics_forced_tau, test_bin, gold2d_all, test_probs = eval_with_tau(test_logits, test_labels, tau_f1_forced)
print(json.dumps(test_metrics_forced_tau, indent=2))

# Persist both for the manifest
test_metrics_dump = {
    "val_learned_tau": float(val_tau_f1_learned),
    "test_metrics_at_val_learned_tau": test_metrics_val_tau,
    "forced_tau": float(tau_f1_forced),
    "test_metrics_at_forced_tau": test_metrics_forced_tau,
    "primary_tau_used_for_supervisor_analyses": float(tau_primary),
    "primary_tau_strategy": tau_primary_name,
}

# (A) REQUEST vs REPLY metrics — using primary τ
def build_valid_counts_from_labels(labels_tensor: np.ndarray) -> np.ndarray:
    L = np.asarray(labels_tensor)
    valid_mask_per_ex = ~(L == -1).all(axis=-1)
    counts = valid_mask_per_ex.sum(axis=1)
    return counts.astype(int)

def flatten_valid_with_flags_from_labels(ds: SeqLabelSentenceIDDataset, logits, labels, tau: float):
    probs2d_raw, gold2d = flatten_valid(logits, labels)
    probs2d = sigmoid_stable(probs2d_raw)
    pred2d  = (probs2d >= tau).astype(int)

    valid_counts = build_valid_counts_from_labels(labels)

    is_req_flags = []
    email_ids = []
    pair_idxs = []

    if len(valid_counts) != len(ds.examples):
        raise RuntimeError(f"[ALIGN ERROR] #examples mismatch: counts={len(valid_counts)} vs ds={len(ds.examples)}")

    for ex, vcount in zip(ds.examples, valid_counts):
        is_req_flags.extend([bool(ex["is_request"] == 1)] * int(vcount))
        email_ids.extend([ex["email_id"]] * int(vcount))
        pair_idxs.extend([ex["pair_idx"]] * int(vcount))

    is_req_flags = np.asarray(is_req_flags, dtype=bool)
    email_ids    = np.asarray(email_ids)
    pair_idxs    = np.asarray(pair_idxs)

    if not (len(is_req_flags) == probs2d.shape[0] == gold2d.shape[0] == pred2d.shape[0]):
        raise RuntimeError(
            f"[ALIGN ERROR] flags={len(is_req_flags)} vs probs={probs2d.shape[0]} vs gold={gold2d.shape[0]} vs pred={pred2d.shape[0]}"
        )

    return probs2d, gold2d, pred2d, is_req_flags, email_ids, pair_idxs

def metrics_from_mask(gold2d, pred2d, mask_bool, title, tau):
    m = metrics_from_bin(gold2d[mask_bool], pred2d[mask_bool])
    print(f"[TEST][{title}] τ={tau:.2f} metrics=")
    print(json.dumps(m, indent=2))
    return m

probs2d_all, gold2d_all, pred2d_all, is_req_flags, email_ids_flat, pair_idxs_flat = \
    flatten_valid_with_flags_from_labels(test_ds, test_logits, test_labels, tau_primary)

m_req = metrics_from_mask(gold2d_all, pred2d_all,  is_req_flags, "REQUEST", tau_primary)
m_rep = metrics_from_mask(gold2d_all, pred2d_all, ~is_req_flags, "REPLY",   tau_primary)

# (B) Per-reply micro-F1 and CSV — using primary τ
def per_reply_micro_f1(ds: SeqLabelSentenceIDDataset, logits, labels, tau: float) -> pd.DataFrame:
    probs2d_raw, gold2d = flatten_valid(logits, labels)
    probs2d = sigmoid_stable(probs2d_raw)
    pred2d  = (probs2d >= tau).astype(int)

    valid_counts = build_valid_counts_from_labels(labels)
    rows = []
    cursor = 0
    for ex, vcount in zip(ds.examples, valid_counts):
        vcount = int(vcount)
        sl = slice(cursor, cursor+vcount)
        cursor += vcount
        if ex["is_request"] == 1:
            continue
        y_true = gold2d[sl]
        y_pred = pred2d[sl]
        f1m = f1_score(y_true, y_pred, average="micro", zero_division=0)
        rows.append({
            "email_id": ex["email_id"],
            "pair_idx": ex["pair_idx"],
            "F1_micro": float(f1m),
            "tau_used": float(tau),
        })
    return pd.DataFrame(rows)

per_reply_df = per_reply_micro_f1(test_ds, test_logits, test_labels, tau_primary)
per_reply_csv = os.path.join(CFG["output_dir"], f"per_reply_f1_{CFG['MODEL_TAG']}.csv")
per_reply_df.to_csv(per_reply_csv, index=False)
print(f"[TEST] Wrote per-reply micro-F1 → {per_reply_csv}")

# (C) Δ on replies + (D) Correlation with polarity gap
def compute_delta_and_correlation(this_csv: str, other_csv: str,
                                  doc_csv: str = "", doc_score_col: str = "overall_politeness"):
    if not other_csv or not os.path.exists(other_csv):
        print("[INFO] Skipping Δ & correlation (no COMPARE_WITH_CSV provided).")
        return None

    A = pd.read_csv(this_csv)
    B = pd.read_csv(other_csv)
    need = {"email_id","pair_idx","F1_micro"}
    if not need.issubset(A.columns) or not need.issubset(B.columns):
        print("[WARN] Missing expected columns in per-reply CSV; skipping Δ & correlation.")
        return None

    merged = A.merge(B, on=["email_id","pair_idx"], suffixes=("_this","_other"))
    merged["ΔF1"] = merged["F1_micro_this"] - merged["F1_micro_other"]

    print("\n[REPLY Δ] this - other (sign depends on which CSV is 'this')")
    print(merged["ΔF1"].describe())

    delta_tbl = merged[["email_id","pair_idx","F1_micro_this","F1_micro_other","ΔF1"]].copy()
    delta_tbl_path = os.path.join(CFG["output_dir"], f"reply_delta_table_vs_{os.path.basename(other_csv)}.csv")
    delta_tbl.to_csv(delta_tbl_path, index=False)
    print(f"[REPLY Δ] Wrote per-reply Δ table → {delta_tbl_path}")

    corr_summary = None
    if doc_csv and os.path.exists(doc_csv):
        docs = pd.read_csv(doc_csv)
        if not {"email_id", doc_score_col}.issubset(docs.columns):
            print("[WARN] DOC_SCORES_CSV missing required columns; skipping correlation.")
        else:
            s = docs[["email_id", doc_score_col]].rename(columns={doc_score_col: "score"})
            test_emails = df_test[[CFG["EMAIL_COL"], CFG["PAIR_COL"]]].drop_duplicates()
            test_emails = test_emails.merge(s, left_on=CFG["EMAIL_COL"], right_on="email_id", how="left") \
                                     .drop(columns=["email_id"])
            gaps = test_emails.groupby(CFG["PAIR_COL"])["score"].agg(list).reset_index()
            gaps = gaps[gaps["score"].apply(lambda x: len([v for v in x if pd.notna(v)])==2)].copy()
            gaps["polarity_gap"] = gaps["score"].apply(lambda x: abs(x[0]-x[1]))
            gaps = gaps[[CFG["PAIR_COL"], "polarity_gap"]].rename(columns={CFG["PAIR_COL"]:"pair_idx"})

            merged2 = merged.merge(gaps, on="pair_idx", how="left")
            DX = merged2.dropna(subset=["polarity_gap","ΔF1"]).copy()
            if DX.empty:
                print("[INFO] No pairs with complete scores for correlation.")
            else:
                from scipy.stats import spearmanr, pearsonr
                sp = spearmanr(DX["polarity_gap"], DX["ΔF1"])
                pe = pearsonr(DX["polarity_gap"], DX["ΔF1"])
                print("\n[CORRELATION] ΔF1(reply) vs |politeness(request)-politeness(reply)|")
                print(f"Spearman ρ = {sp.statistic:.3f} (p={sp.pvalue:.3g})")
                print(f"Pearson  r = {pe.statistic:.3f} (p={pe.pvalue:.3g})")
                corr_summary = {"spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
                                "pearson_r": float(pe.statistic), "pearson_p": float(pe.pvalue),
                                "n_pairs_correlated": int(len(DX))}
    else:
        print("[INFO] Skipping correlation (no DOC_SCORES_CSV).")

    return corr_summary

corr_summary = compute_delta_and_correlation(
    this_csv=per_reply_csv,
    other_csv=CFG["COMPARE_WITH_CSV"],
    doc_csv=CFG["DOC_SCORES_CSV"],
    doc_score_col=CFG["DOC_SCORE_COL"]
)

# Per-label IoU under primary τ
_, test_bin_for_IoU, gold2d_for_IoU, _ = eval_with_tau(test_logits, test_labels, tau_primary)
per_label_j = jaccard_score(gold2d_for_IoU, test_bin_for_IoU, average=None, zero_division=0)
print(f"\n[TEST] per-label Jaccard (IoU) at primary τ={tau_primary:.2f}")
for i, lbl in enumerate(LABELS):
    print(f"  {lbl:6s}: {per_label_j[i]:.4f}")

# Save TEST predictions per sentence (with per-row Jaccard, at primary τ)
if CFG["save_test_predictions"]:
    rows = []
    probs2d_raw, _gold2d = flatten_valid(test_logits, test_labels)
    probs2d_all_sent = sigmoid_stable(probs2d_raw)

    valid_counts = build_valid_counts_from_labels(test_labels)
    cursor = 0
    for ex, vcount in zip(test_ds.examples, valid_counts):
        vcount = int(vcount)
        sl = slice(cursor, cursor+vcount)
        cursor += vcount

        probs_S = probs2d_all_sent[sl]
        gold_S  = (np.stack(ex["labels"]) > 0.5).astype(int)
        pred_S  = (probs_S >= tau_primary).astype(int)

        S_ok = min(gold_S.shape[0], probs_S.shape[0])
        probs_S = probs_S[:S_ok]
        gold_S  = gold_S[:S_ok]
        pred_S  = pred_S[:S_ok]

        for j in range(S_ok):
            gold = gold_S[j]
            pred = pred_S[j]
            inter = int(np.logical_and(gold, pred).sum())
            union = int(np.logical_or (gold, pred).sum())
            # NB: jacc=1 when both gold and pred are all-zero (sklearn zero_division=1 convention).
            jacc = float(inter/union) if union>0 else 1.0
            row = {
                "model_tag": CFG["MODEL_TAG"],
                "uses_request_context": int(CFG["USES_REQUEST_CONTEXT"]),
                "email_id": ex["email_id"],
                "is_request": ex["is_request"],
                "seed": ex["seed"],
                "pair_idx": ex["pair_idx"],
                "sentence_idx": ex["sentence_idx_list"][j] if j < len(ex["sentence_idx_list"]) else j,
                "gold_labels": ",".join([l for k,l in enumerate(LABELS) if gold[k] == 1]),
                f"pred_labels_tau_{tau_primary:.2f}": ",".join([l for k,l in enumerate(LABELS) if probs_S[j][k] >= tau_primary]),
                f"jaccard_row_tau_{tau_primary:.2f}": jacc,
                "primary_tau": float(tau_primary),
                "primary_tau_strategy": tau_primary_name,
            }
            for k,lbl in enumerate(LABELS):
                row[f"pred_prob_{lbl}"] = float(probs_S[j][k])
            rows.append(row)

    out_csv = os.path.join(CFG["output_dir"], f"test_predictions_per_sentence_{CFG['MODEL_TAG']}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("[DONE] Wrote:", out_csv)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# seqlabel_req_plus_rep.py

print("SeqLabel + ASL Request+Reply")

import os
import re
import ast
import gc
import json
import math
import random
import inspect
import textwrap
from typing import List, Dict, Any, Tuple, Union

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
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

CFG = {
    "train_csv": "data/splits/train_seed42.csv",
    "val_csv":   "data/splits/val_seed42.csv",
    "test_csv":  "data/splits/test_seed42.csv",
    "TEXT_COL":     "covered_text",
    "LABEL_COL":    "GoldFaceAct",
    "EMAIL_COL":    "email_id",
    "SENTIDX_COL":  "sentence_idx",
    "ISREQ_COL":    "is_request",   # 1=Request, 0=Reply
    "SEED_COL":     "seed",
    "PAIR_COL":     "pair_idx",
    "LABELS": [
        "HNeg+","HNeg-","HPos+","HPos-","Neutral",
        "SNeg+","SNeg-","SPos+","SPos-",
    ],
    "model_name": "bert-base-uncased",
    "max_length": 512,
    "output_dir": os.environ.get(
    "OUTDIR",
    "bert_seqlabel_req_plus_rep_sentid_asl"
),
    "epochs": 5,
    "batch_size": 4,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "early_stopping_patience": 2,
    "use_pos_weight": False,
    "asl_gamma_pos": 0.0,
    "asl_gamma_neg": 4.0,
    "asl_clip": 0.05,
    "debug_sequences": 3,
    "save_test_predictions": True,
    "seed": 42,
    # Force τ=0.60 for Global τ@F1 in BOTH VAL print and TEST eval
    "FORCE_TAU_F1": 0.60,
}

os.makedirs(CFG["output_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
if torch.cuda.is_available(): torch.cuda.manual_seed_all(CFG["seed"])
print(f"[INFO] Single-seed run, seed = {CFG['seed']}, FORCE_TAU_F1 = {CFG['FORCE_TAU_F1']}")

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

# Labels + parsing

LABELS = CFG["LABELS"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

def parse_labels(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return [str(x).strip() for x in ast.literal_eval(s)]
        except Exception:
            pass
    return [t.strip() for t in re.split(r"[;,]", s) if t.strip()]

def to_multi_hot(names):
    vec = np.zeros(len(LABELS), dtype=np.float32)
    for n in names:
        if n in label2id:
            vec[label2id[n]] = 1.0
    return vec

# Load splits
def load_split(csv_path: str, CFG):
    df = pd.read_csv(csv_path)
    need = [CFG["TEXT_COL"], CFG["LABEL_COL"], CFG["EMAIL_COL"], CFG["SENTIDX_COL"],
            CFG["ISREQ_COL"], CFG["SEED_COL"], CFG["PAIR_COL"]]
    for c in need:
        assert c in df.columns, f"Missing '{c}' in {csv_path}"
    for c in (CFG["EMAIL_COL"], CFG["SENTIDX_COL"], CFG["ISREQ_COL"], CFG["SEED_COL"], CFG["PAIR_COL"]):
        df[c] = df[c].astype(int)
    df["gold_list"] = df[CFG["LABEL_COL"]].map(parse_labels)
    df["y_vec"] = df["gold_list"].map(to_multi_hot)
    return df.reset_index(drop=True)

df_train = load_split(CFG["train_csv"], CFG)
df_val   = load_split(CFG["val_csv"],   CFG)
df_test  = load_split(CFG["test_csv"],  CFG)
print(f"[INFO] Loaded rows: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

def check_split_integrity(df_train, df_val, df_test, seed_col):
    s_tr, s_va, s_te = set(df_train[seed_col]), set(df_val[seed_col]), set(df_test[seed_col])
    print("\n[SEED GROUPS]")
    print(f"Train ({len(s_tr)}): {sorted(list(s_tr))}")
    print(f"Val   ({len(s_va)}): {sorted(list(s_va))}")
    print(f"Test  ({len(s_te)}): {sorted(list(s_te))}")
    print("\n[Overlap checks]")
    print(f"Train ∩ Val  ({len(s_tr & s_va)}): {sorted(list(s_tr & s_va))}")
    print(f"Train ∩ Test ({len(s_tr & s_te)}): {sorted(list(s_tr & s_te))}")
    print(f"Val   ∩ Test ({len(s_va & s_te)}): {sorted(list(s_va & s_te))}")

check_split_integrity(df_train, df_val, df_test, CFG["SEED_COL"])

# Tokenizer
tok = AutoTokenizer.from_pretrained(CFG["model_name"], use_fast=True)
assert tok.is_fast, "Use a *fast* tokenizer to get offset_mapping."

# Build Request Reply pairs

def build_pairs(df: pd.DataFrame, CFG) -> List[Dict[str, Any]]:
    exs = []
    sort_cols = [CFG["SEED_COL"], CFG["PAIR_COL"], CFG["ISREQ_COL"], CFG["EMAIL_COL"], CFG["SENTIDX_COL"]]
    for (sd, pr), grp in df.sort_values(sort_cols).groupby(
        [CFG["SEED_COL"], CFG["PAIR_COL"]], sort=False
    ):
        gr_req = grp[grp[CFG["ISREQ_COL"]] == 1].sort_values([CFG["EMAIL_COL"], CFG["SENTIDX_COL"]])
        gr_rep = grp[grp[CFG["ISREQ_COL"]] == 0].sort_values([CFG["EMAIL_COL"], CFG["SENTIDX_COL"]])

        texts_req = gr_req[CFG["TEXT_COL"]].astype(str).tolist()
        texts_rep = gr_rep[CFG["TEXT_COL"]].astype(str).tolist()
        labels_req = [np.array(v, dtype=np.float32) for v in gr_req["y_vec"].tolist()]
        labels_rep = [np.array(v, dtype=np.float32) for v in gr_rep["y_vec"].tolist()]
        sidx_req = gr_req[CFG["SENTIDX_COL"]].astype(int).tolist()
        sidx_rep = gr_rep[CFG["SENTIDX_COL"]].astype(int).tolist()
        email_req = gr_req[CFG["EMAIL_COL"]].astype(int).tolist()
        email_rep = gr_rep[CFG["EMAIL_COL"]].astype(int).tolist()

        texts = texts_req + texts_rep
        labels = labels_req + labels_rep
        sides = [0] * len(texts_req) + [1] * len(texts_rep)  # 0=Request, 1=Reply
        sidxs = sidx_req + sidx_rep
        emails = email_req + email_rep

        exs.append({
            "seed": int(sd),
            "pair_idx": int(pr),
            "texts": texts,
            "labels": labels,
            "sides": sides,
            "sentence_idx_list": sidxs,
            "email_ids": emails,
        })
    return exs

train_pairs = build_pairs(df_train, CFG)
val_pairs   = build_pairs(df_val,   CFG)
test_pairs  = build_pairs(df_test,  CFG)
print(f"[INFO] Pair counts: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

def concat_with_ranges(texts: List[str]):
    parts, ranges, pos = [], [], 0
    for i, s in enumerate(texts):
        if i > 0:
            parts.append(" "); pos += 1
        start = pos
        parts.append(s); pos += len(s)
        ranges.append((start, pos))
    return "".join(parts), ranges

def encode_with_sentence_ids_and_types(concat_text, char_ranges, side_flags, tokenizer, max_length):
    enc = tokenizer(
        concat_text,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        padding=False,
    )
    ids = enc["input_ids"]
    attn = enc["attention_mask"]
    offsets = enc["offset_mapping"]
    sp = enc["special_tokens_mask"]
    L = len(ids)
    token_type_ids = [0] * L
    S = len(char_ranges)
    sentence_ids = [-1] * L
    sent_token_counts = [0] * S

    for t_idx, ((a, b), is_sp, m) in enumerate(zip(offsets, sp, attn)):
        if m == 0 or is_sp or a == b:
            continue
        for s_idx, (sa, se) in enumerate(char_ranges):
            if not (b <= sa or a >= se):
                sentence_ids[t_idx] = s_idx
                token_type_ids[t_idx] = int(side_flags[s_idx])  # 0=Request,1=Reply
                sent_token_counts[s_idx] += 1
                break

    kept = [i for i, c in enumerate(sent_token_counts) if c > 0]
    if not kept:
        return ids, attn, token_type_ids, sentence_ids, kept, []

    remap = {old: i for i, old in enumerate(kept)}
    sentence_ids = [(remap[s] if (s != -1 and s in remap) else -1) for s in sentence_ids]
    side_kept = [side_flags[i] for i in kept]
    return ids, attn, token_type_ids, sentence_ids, kept, side_kept

class SeqLabelSentenceIDDataset(Dataset):
    def __init__(self, pairs, tokenizer, cfg):
        self.examples = []
        self._skipped = 0
        for ex in pairs:
            concat, ranges = concat_with_ranges(ex["texts"])
            ids, attn, tt, sids, kept, side_kept = encode_with_sentence_ids_and_types(
                concat, ranges, ex["sides"], tokenizer, cfg["max_length"]
            )
            if len(kept) == 0:
                self._skipped += 1
                continue
            labels_kept = [ex["labels"][i] for i in kept]
            sidxs_kept = [ex["sentence_idx_list"][i] for i in kept]
            emails_kept = [ex["email_ids"][i] for i in kept]
            self.examples.append({
                "input_ids": ids,
                "attention_mask": attn,
                "token_type_ids": tt,
                "sentence_ids": sids,
                "labels": labels_kept,
                "side_flags": side_kept,
                "email_ids": emails_kept,
                "seed": ex["seed"],
                "pair_idx": ex["pair_idx"],
                "sentidx_list": sidxs_kept,
                "concat_text": concat,
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
            "side_flags": torch.tensor(e["side_flags"], dtype=torch.long),
        }

train_ds = SeqLabelSentenceIDDataset(train_pairs, tok, CFG)
val_ds   = SeqLabelSentenceIDDataset(val_pairs,   tok, CFG)
test_ds  = SeqLabelSentenceIDDataset(test_pairs,  tok, CFG)

class DataCollatorSeqLabelSentenceID:
    def __init__(self, pad_token_id, label_dim):
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
        label_grid, side_grid = [], []
        for x in batch:
            S = x["labels"].shape[0]
            if S < max_S:
                pad_rows = torch.full((max_S - S, self.label_dim), -1.0, dtype=torch.float32)
                label_grid.append(torch.cat([x["labels"], pad_rows], dim=0))
                side_grid.append(F.pad(x["side_flags"], (0, max_S - S), value=-1))
            else:
                label_grid.append(x["labels"])
                side_grid.append(x["side_flags"])
        labels = torch.stack(label_grid, dim=0)
        side_flags = torch.stack(side_grid, dim=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "sentence_ids": sentence_ids,
            "labels": labels,
            "side_flags": side_flags,
        }

collator = DataCollatorSeqLabelSentenceID(
    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else 0,
    label_dim=len(LABELS),
)

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

        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss.mean()

class BertSeqLabelSentenceID(nn.Module):
    def __init__(self, model_name, num_labels, pos_weight=None, gp=0.0, gn=4.0, clip=0.05):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.pos_weight = pos_weight
        self.asl = AsymmetricLoss(gamma_pos=gp, gamma_neg=gn, clip=clip)

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

    def forward(self, input_ids, attention_mask, token_type_ids=None, sentence_ids=None, labels=None, **_):
        H = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        B, L, Hdim = H.shape
        if labels is not None:
            S_max = labels.shape[1]
        else:
            S_max = int(sentence_ids.max(dim=1).values.clamp_min(-1).max().item() + 1)

        sent_embs = []
        for b in range(B):
            if labels is not None:
                S_b = int((labels[b, :, 0] != -1).sum().item())
            else:
                S_b = int(sentence_ids[b].max().item() + 1)
            means_b = self._mean_pool_by_sentence_id(H[b], attention_mask[b], sentence_ids[b], S_b, Hdim)
            if S_b < S_max:
                pad = torch.zeros(S_max - S_b, Hdim, device=H.device, dtype=H.dtype)
                means_b = torch.cat([means_b, pad], dim=0)
            sent_embs.append(means_b)

        logits = self.classifier(self.dropout(torch.stack(sent_embs, dim=0)))
        loss = None
        if labels is not None:
            mask = labels[..., 0] != -1
            if mask.any():
                y = labels[mask, :]
                z = logits[mask, :]
                pw = self.pos_weight.to(z.device) if self.pos_weight is not None else None
                loss = self.asl(z, y, pos_weight=pw)
            else:
                loss = torch.zeros([], device=logits.device, dtype=torch.float32)
        return {"loss": loss, "logits": logits}

pos_weight_tensor = None
if CFG["use_pos_weight"]:
    y = np.stack(df_train["y_vec"].values)
    pos = y.sum(axis=0); neg = y.shape[0] - pos
    pos_weight_tensor = torch.tensor((neg / (pos + 1e-6)).astype(np.float32))

model = BertSeqLabelSentenceID(
    CFG["model_name"],
    num_labels=len(LABELS),
    pos_weight=pos_weight_tensor,
    gp=CFG["asl_gamma_pos"],
    gn=CFG["asl_gamma_neg"],
    clip=CFG["asl_clip"],
)

# Metric helpers

def flatten_valid(pred_logits: np.ndarray, labels: np.ndarray):
    pred_logits = np.asarray(pred_logits)
    labels = np.asarray(labels)
    padded_mask = (labels == -1).all(axis=-1)
    keep_mask = ~padded_mask
    probs2d = pred_logits.reshape(-1, pred_logits.shape[-1])[keep_mask.reshape(-1)]
    gold2d = labels.reshape(-1, labels.shape[-1])[keep_mask.reshape(-1)]
    gold2d = (gold2d > 0).astype(int)
    return probs2d, gold2d

def metrics_from_bin(gold: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    hamming = np.not_equal(pred, gold).mean()
    return {
        "micro/f1": f1_score(gold, pred, average="micro", zero_division=0),
        "macro/f1": f1_score(gold, pred, average="macro", zero_division=0),
        "micro/precision": precision_score(gold, pred, average="micro", zero_division=0),
        "micro/recall": recall_score(gold, pred, average="micro", zero_division=0),
        "jaccard/micro": jaccard_score(gold, pred, average="micro", zero_division=0),
        "jaccard/macro": jaccard_score(gold, pred, average="macro", zero_division=0),
        "jaccard/samples": jaccard_score(gold, pred, average="samples", zero_division=0),
        "hamming_loss": float(hamming),
        "avg_true_k": float(gold.sum(axis=1).mean()),
        "avg_pred_k": float(pred.sum(axis=1).mean()),
    }

def training_metrics_fixed_tau(eval_pred):
    logits, labels = eval_pred
    probs2d, gold2d = flatten_valid(logits, labels)
    preds = (sigmoid_stable(probs2d) >= 0.50).astype(int)
    return metrics_from_bin(gold2d, preds)

class TableLoggerCallback(TrainerCallback):
    def __init__(self):
        self.rows = []
        self.last_train_loss = None
        self._printed = False
        self._seen = set()
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = float(logs["loss"])
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        m = metrics or {}
        g = lambda k: m.get(f"eval_{k}", m.get(k))
        epoch = int(round(state.epoch or 0)) if state.epoch is not None else (len(self.rows) + 1)
        if epoch in self._seen:
            return
        self._seen.add(epoch)
        row = {
            "epoch": epoch,
            "train_loss": self.last_train_loss,
            "val_loss": g("loss"),
            "micro/f1": g("micro/f1"),
            "macro/f1": g("macro/f1"),
            "micro/precision": g("micro/precision"),
            "micro/recall": g("micro/recall"),
            "jaccard/micro": g("jaccard/micro"),
            "jaccard/macro": g("jaccard/macro"),
            "jaccard/samples": g("jaccard/samples"),
        }
        self.rows.append(row)
        def fmt(x):
            return "—" if x is None else f"{x:.6f}"
        if not self._printed:
            print("\nEpoch\tTraining Loss\tValidation Loss\tMicro/f1\tMacro/f1\tMicro/precision\tMicro/recall\tJaccard/micro\tJaccard/macro\tJaccard/samples")
            self._printed = True
        print(f"{row['epoch']}\t{fmt(row['train_loss'])}\t{fmt(row['val_loss'])}\t{fmt(row['micro/f1'])}\t{fmt(row['macro/f1'])}\t{fmt(row['micro/precision'])}\t{fmt(row['micro/recall'])}\t{fmt(row['jaccard/micro'])}\t{fmt(row['jaccard/macro'])}\t{fmt(row['jaccard/samples'])}")
    def on_train_end(self, args, state, control, **kwargs):
        if self.rows:
            out_csv = os.path.join(args.output_dir, "epoch_log_summary.csv")
            pd.DataFrame(self.rows).to_csv(out_csv, index=False)
            print(f"[LOG] Wrote epoch summary → {out_csv}")

# Trainer kwargs helper 
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
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=CFG["early_stopping_patience"]),
        TableLoggerCallback(),
    ],
)))

print("[INFO] Starting training…")
trainer.train()

# Threshold helpers (VAL)
GRID = np.linspace(0.05, 0.95, 19)

def pick_global_tau_by(metric_name: str, logits, labels, grid=GRID):
    probs2d, gold2d = flatten_valid(logits, labels)
    probs = sigmoid_stable(probs2d)
    best_tau, best_score = 0.5, -1.0
    for tau in grid:
        pred = (probs >= tau).astype(int)
        score = (
            f1_score(gold2d, pred, average="micro", zero_division=0)
            if metric_name == "micro/f1"
            else jaccard_score(gold2d, pred, average="micro", zero_division=0)
        )
        if score > best_score:
            best_tau, best_score = float(tau), float(score)
    return best_tau, best_score

def pick_pcut_tau(logits, labels, grid=GRID):
    probs2d, gold2d = flatten_valid(logits, labels)
    probs = sigmoid_stable(probs2d)
    target_k = gold2d.sum(axis=1).mean()
    best_tau, best_diff = 0.5, float("inf")
    for tau in grid:
        pred = (probs >= tau).astype(int)
        diff = abs(pred.sum(axis=1).mean() - target_k)
        if diff < best_diff:
            best_diff, best_tau = diff, float(tau)
    return best_tau, float(target_k)

def pick_scut_f1_tau_vec(logits, labels, grid=GRID):
    probs2d, gold2d = flatten_valid(logits, labels)
    probs = sigmoid_stable(probs2d)
    C = probs.shape[1]
    tau_vec = np.zeros(C, dtype=np.float32)
    for c in range(C):
        best_tau_c, best_f1_c = 0.5, -1.0
        p = probs[:, c]
        g = gold2d[:, c]
        for tau in grid:
            f1_c = f1_score(g, (p >= tau).astype(int), average="binary", zero_division=0)
            if f1_c > best_f1_c:
                best_f1_c, best_tau_c = f1_c, float(tau)
        tau_vec[c] = best_tau_c
    return tau_vec

def eval_with_thresholds(logits, labels, scalar_tau=None, tau_vec=None):
    probs2d, gold2d = flatten_valid(logits, labels)
    probs = sigmoid_stable(probs2d)
    if tau_vec is not None:
        pred = (probs >= np.asarray(tau_vec).reshape(1, -1)).astype(int)
    else:
        pred = (probs >= float(scalar_tau)).astype(int)
    return metrics_from_bin(gold2d, pred), pred, gold2d, probs

# VAL (with τ forced to 0.60)
print("[INFO] Searching thresholds on VAL…")
val_out = trainer.predict(val_ds)
val_logits, val_labels = val_out.predictions, val_out.label_ids

tau_f1_auto, _ = pick_global_tau_by("micro/f1", val_logits, val_labels)  # computed but overridden
tau_jac, _ = pick_global_tau_by("jaccard/micro", val_logits, val_labels)
tau_vec = pick_scut_f1_tau_vec(val_logits, val_labels)
tau_pcut, _ = pick_pcut_tau(val_logits, val_labels)

# FORCE τ to 0.60 for Global τ@F1 
tau_f1 = float(CFG["FORCE_TAU_F1"])

print(f"[VAL][Global τ@F1] τ={tau_f1:.2f}  (auto-best was {tau_f1_auto:.2f}, forced for comparability)")
print(json.dumps(eval_with_thresholds(val_logits, val_labels, scalar_tau=tau_f1)[0], indent=2))
print(f"[VAL][Global τ@Jaccard] τ={tau_jac:.2f}  metrics=")
print(json.dumps(eval_with_thresholds(val_logits, val_labels, scalar_tau=tau_jac)[0], indent=2))
print(f"[VAL][Per-label SCut-F1] τ_vec={[float(round(x,2)) for x in tau_vec.tolist()]}  metrics=")
print(json.dumps(eval_with_thresholds(val_logits, val_labels, tau_vec=tau_vec)[0], indent=2))
print(f"[VAL][PCut] τ={tau_pcut:.2f}  metrics=")
print(json.dumps(eval_with_thresholds(val_logits, val_labels, scalar_tau=tau_pcut)[0], indent=2))

# TEST under VAL-selected thresholds (Global τ@F1 forced to 0.60)
test_out = trainer.predict(test_ds)
test_logits, test_labels = test_out.predictions, test_out.label_ids

print(f"[INFO] Evaluating on TEST using Global τ@F1 (τ={tau_f1:.2f})…")
test_main, test_pred_bin, test_gold2d, test_probs = eval_with_thresholds(
    test_logits, test_labels, scalar_tau=tau_f1
)
print("[TEST] Global τ@F1", json.dumps(test_main, indent=2))

print("[TEST] Per-label SCut-F1")
print(json.dumps(eval_with_thresholds(test_logits, test_labels, tau_vec=tau_vec)[0], indent=2))

print("[TEST] PCut")
print(json.dumps(eval_with_thresholds(test_logits, test_labels, scalar_tau=tau_pcut)[0], indent=2))

print("[TEST] Global τ@Jaccard")
print(json.dumps(eval_with_thresholds(test_logits, test_labels, scalar_tau=tau_jac)[0], indent=2))

# REQUEST vs REPLY split metrics on TEST
def flatten_pred_gold_side_per_example(ds: SeqLabelSentenceIDDataset, logits, tau: float):
    logits = np.asarray(logits)
    preds_list, golds_list, sides_list = [], [], []
    N = len(ds.examples)
    for i in range(N):
        ex = ds.examples[i]
        S_b = len(ex["labels"])
        if S_b == 0:
            continue
        probs_b = sigmoid_stable(logits[i][:S_b])
        pred_b = (probs_b >= tau).astype(int)
        gold_b = (np.stack(ex["labels"][:S_b]) > 0.5).astype(int)
        side_b = np.array(ex["side_flags"][:S_b], dtype=int)
        preds_list.append(pred_b)
        golds_list.append(gold_b)
        sides_list.append(side_b)
    pred2d = np.vstack(preds_list) if preds_list else np.zeros((0, len(LABELS)), dtype=int)
    gold2d = np.vstack(golds_list) if golds_list else np.zeros((0, len(LABELS)), dtype=int)
    side_flags = np.concatenate(sides_list) if sides_list else np.zeros((0,), dtype=int)
    assert pred2d.shape == gold2d.shape
    assert pred2d.shape[0] == side_flags.shape[0]
    return pred2d, gold2d, side_flags

pred_all, gold_all, side_flags = flatten_pred_gold_side_per_example(test_ds, test_logits, tau_f1)
mask_req = side_flags == 0
mask_rep = side_flags == 1

def metrics_subset(gold2d, pred2d, mask, title):
    m = metrics_from_bin(gold2d[mask], pred2d[mask])
    print(f"[TEST][{title}] τ={tau_f1:.2f} metrics=")
    print(json.dumps(m, indent=2))
    return m

m_req = metrics_subset(gold_all, pred_all, mask_req, "REQUEST")
m_rep = metrics_subset(gold_all, pred_all, mask_rep, "REPLY")

# Per-label IoU on TEST under τ=0.60
per_label_j = jaccard_score(gold_all, pred_all, average=None, zero_division=0)
print("[TEST] per-label Jaccard (IoU) under Global τ@F1")
for i, lbl in enumerate(LABELS):
    print(f"  {lbl:6s}: {per_label_j[i]:.4f}")

# Per-reply micro-F1 CSV
def per_reply_micro_f1(ds: SeqLabelSentenceIDDataset, logits, tau: float) -> pd.DataFrame:
    logits = np.asarray(logits)
    rows = []
    N = len(ds.examples)
    for i in range(N):
        ex = ds.examples[i]
        S_b = len(ex["labels"])
        if S_b == 0:
            continue
        probs_b = sigmoid_stable(logits[i][:S_b])
        pred_b = (probs_b >= tau).astype(int)
        gold_b = (np.stack(ex["labels"][:S_b]) > 0.5).astype(int)
        side_b = np.array(ex["side_flags"][:S_b], dtype=int)
        emails_b = np.array(ex["email_ids"][:S_b], dtype=int)
        reply_mask = side_b == 1
        if not reply_mask.any():
            continue
        y_true = gold_b[reply_mask]
        y_pred = pred_b[reply_mask]
        emails = emails_b[reply_mask]
        for em in np.unique(emails):
            em_mask = emails == em
            f1m = f1_score(y_true[em_mask], y_pred[em_mask], average="micro", zero_division=0)
            rows.append({
                "seed": ex["seed"],
                "pair_idx": ex["pair_idx"],
                "reply_email_id": int(em),
                "F1_micro": float(f1m),
            })
    return pd.DataFrame(rows)

per_reply_df = per_reply_micro_f1(test_ds, test_logits, tau_f1)
per_reply_csv = os.path.join(CFG["output_dir"], "per_reply_f1_RequestPlusReply.csv")
per_reply_df.to_csv(per_reply_csv, index=False)
print(f"[TEST] Wrote per-reply micro-F1 → {per_reply_csv}")

# Save TEST predictions under τ=0.60
if CFG["save_test_predictions"]:
    rows = []
    logits_np = np.asarray(test_logits)
    for i, ex in enumerate(test_ds.examples):
        S_b = len(ex["labels"])
        if S_b == 0:
            continue
        probs_S = sigmoid_stable(logits_np[i][:S_b])
        pred_S = (probs_S >= tau_f1).astype(int)
        gold_S = (np.stack(ex["labels"][:S_b]) > 0.5).astype(int)
        S_ok = min(gold_S.shape[0], probs_S.shape[0])
        probs_S, pred_S, gold_S = probs_S[:S_ok], pred_S[:S_ok], gold_S[:S_ok]
        for j in range(S_ok):
            gold = gold_S[j]
            pred = pred_S[j]
            inter = int(np.logical_and(gold, pred).sum())
            union = int(np.logical_or(gold, pred).sum())
            jacc = float(inter / union) if union > 0 else 1.0
            row = {
                "seed": ex["seed"],
                "pair_idx": ex["pair_idx"],
                "side": "REQUEST" if ex["side_flags"][j] == 0 else "REPLY",
                "email_id": int(ex["email_ids"][j]),
                "sentence_idx": ex["sentidx_list"][j],
                "gold_labels": ",".join([l for k, l in enumerate(LABELS) if gold[k] == 1]),
                f"pred_labels_tau_{tau_f1:.2f}": ",".join([l for k, l in enumerate(LABELS) if probs_S[j][k] >= tau_f1]),
                f"jaccard_row_tau_{tau_f1:.2f}": jacc,
            }
            for k, lbl in enumerate(LABELS):
                row[f"pred_prob_{lbl}"] = float(probs_S[j][k])
            rows.append(row)
    out_csv = os.path.join(CFG["output_dir"], "test_predictions_per_sentence_req_plus_rep_sentid_asl.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("[DONE] Wrote:", out_csv)

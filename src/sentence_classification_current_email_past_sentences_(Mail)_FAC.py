#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# past_same_email_Mail.py

print("TargetFirst: PastSameEmail_Mail (req or rep)")

import os, re, ast, json, math, random, textwrap, inspect
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
)
from sklearn.metrics import (
    f1_score, precision_score, recall_score, jaccard_score
)

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

    "filter_is_request": None,

    "LABELS": ["HNeg+","HNeg-","HPos+","HPos-","Neutral","SNeg+","SNeg-","SPos+","SPos-"],

    "model_name": "bert-base-uncased",
    "max_length": 256,

    "output_dir": "bert_targetfirst_past_same_email_asl",
    "epochs": 5,
    "batch_size": 16,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "early_stopping_patience": 2,

    "use_pos_weight": False,
    "asl_gamma_pos": 0.0,
    "asl_gamma_neg": 4.0,
    "asl_clip": 0.05,

    "seed": 42,
    "debug_samples": 3,
    "save_test_predictions": True,

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
    need = [CFG["TEXT_COL"], CFG["LABEL_COL"], CFG["EMAIL_COL"], CFG["SENTIDX_COL"], CFG["ISREQ_COL"], CFG["SEED_COL"]]
    for c in need:
        assert c in df.columns, f"Missing column '{c}' in {csv_path}"
    if CFG["filter_is_request"] in (0,1):
        df = df[df[CFG["ISREQ_COL"]] == CFG["filter_is_request"]].copy()

    for c in (CFG["EMAIL_COL"], CFG["SENTIDX_COL"], CFG["ISREQ_COL"], CFG["SEED_COL"]):
        df[c] = df[c].astype(int)

    df["gold_list"] = df[CFG["LABEL_COL"]].map(parse_labels)
    df["y_vec"]     = df["gold_list"].map(to_multi_hot)
    return df.reset_index(drop=True)

df_train = load_split(CFG["train_csv"], CFG)
df_val   = load_split(CFG["val_csv"],   CFG)
df_test  = load_split(CFG["test_csv"],  CFG)

print(f"[INFO] Loaded rows: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

# Integrity & distribution checks
def check_split_integrity(df_train, df_val, df_test, seed_col):
    s_tr = set(df_train[seed_col].unique()); s_va = set(df_val[seed_col].unique()); s_te = set(df_test[seed_col].unique())
    print("\n[SEED GROUPS]")
    print(f"Train ({len(s_tr)}): {sorted(list(s_tr))}")
    print(f"Val   ({len(s_va)}): {sorted(list(s_va))}")
    print(f"Test  ({len(s_te)}): {sorted(list(s_te))}")
    ov_tr_va = sorted(list(s_tr & s_va)); ov_tr_te = sorted(list(s_tr & s_te)); ov_va_te = sorted(list(s_va & s_te))
    print("\n[Overlap checks]")
    print(f"Train ∩ Val  ({len(ov_tr_va)}): {ov_tr_va}")
    print(f"Train ∩ Test ({len(ov_tr_te)}): {ov_tr_te}")
    print(f"Val   ∩ Test ({len(ov_va_te)}): {ov_va_te}")
    if ov_tr_va or ov_tr_te or ov_va_te:
        print("[LEAKAGE WARNING] Some seeds appear in multiple splits!")
    else:
        print("[OK] No seeds in common across splits.")

def print_is_request_counts(df_train, df_val, df_test, col):
    print("\n[TRAIN is_request counts]\n", df_train[col].value_counts(dropna=False))
    print("[VAL   is_request counts]\n", df_val[col].value_counts(dropna=False))
    print("[TEST  is_request counts]\n", df_test[col].value_counts(dropna=False))

check_split_integrity(df_train, df_val, df_test, CFG["SEED_COL"])
print_is_request_counts(df_train, df_val, df_test, CFG["ISREQ_COL"])

# Build Target (s_i) + Context (past sentences, same email, joined by spaces)
def add_target_and_context(df: pd.DataFrame, CFG) -> pd.DataFrame:
    email_col, sent_col, text_col = CFG["EMAIL_COL"], CFG["SENTIDX_COL"], CFG["TEXT_COL"]
    df2 = df.sort_values([email_col, sent_col]).copy()
    df2["target"]  = df2[text_col].astype(str)
    df2["context"] = ""
    for eid, grp in df2.groupby(email_col, sort=False):
        texts = grp[text_col].astype(str).tolist()
        ctx_vals, pasts = [], []
        for s in texts:
            ctx_vals.append(" ".join(pasts) if pasts else "")
            pasts.append(s)
        df2.loc[grp.index, "context"] = ctx_vals
    return df2.reset_index(drop=True)

df_train_ctx = add_target_and_context(df_train, CFG)
df_val_ctx   = add_target_and_context(df_val,   CFG)
df_test_ctx  = add_target_and_context(df_test,  CFG)

# Tokenizer & Dataset (pair encoding; "only_second")
tok = AutoTokenizer.from_pretrained(CFG["model_name"])

def tokenize_pairs(target_list: List[str], context_list: List[str]):
    return tok(
        target_list,
        context_list,
        padding=True,
        truncation="only_second",
        max_length=CFG["max_length"],
        return_token_type_ids=True,
        return_tensors=None
    )

class PastReqRepDataset(Dataset):
    def __init__(self, df_ctx: pd.DataFrame):
        self.df = df_ctx.reset_index(drop=True)
        self.target  = self.df["target"].astype(str).tolist()
        self.context = self.df["context"].astype(str).tolist()
        self.labels  = np.stack(self.df["y_vec"].values)
        self.enc = tokenize_pairs(self.target, self.context)
    def __len__(self): return len(self.target)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k,v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

train_ds = PastReqRepDataset(df_train_ctx)
val_ds   = PastReqRepDataset(df_val_ctx)
test_ds  = PastReqRepDataset(df_test_ctx)

# (Optional) pos_weight for ASL
pos_weight_tensor = None
if CFG["use_pos_weight"]:
    y = np.stack(df_train_ctx["y_vec"].values)
    pos = y.sum(axis=0); neg = y.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).astype(np.float32)
    pos_weight_tensor = torch.tensor(pos_weight)
    print("[INFO] pos_weight:", {lbl: float(round(w,2)) for lbl,w in zip(LABELS, pos_weight)})

# Model + ASL
model = AutoModelForSequenceClassification.from_pretrained(
    CFG["model_name"],
    num_labels=len(LABELS),
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id
)

class AsymmetricLoss(nn.Module):
    
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, reduction="mean"):
        super().__init__()
        self.gp, self.gn, self.clip, self.eps, self.reduction = gamma_pos, gamma_neg, clip, eps, reduction

    def forward(self, logits, targets, pos_weight=None):
        # [FIX-LOSSFP32][FIX-CLAMP]
        logits_f = logits.float().clamp(-30.0, 30.0)
        targets_f = targets.float()

        # [FIX-LOGSIG]
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

class ASLTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, asl_gamma_pos=0.0, asl_gamma_neg=4.0, asl_clip=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.asl = AsymmetricLoss(gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.asl(logits, labels, pos_weight=self.pos_weight)
        return (loss, outputs) if return_outputs else loss

# Training-time metrics (epoch table @ τ=0.50)
def compute_metrics_training(eval_pred):
    logits, labels = eval_pred
    probs = sigmoid_stable(logits)
    pred  = (probs >= 0.50).astype(int)
    return {
        "micro/f1":        f1_score(labels, pred, average="micro",  zero_division=0),
        "macro/f1":        f1_score(labels, pred, average="macro",  zero_division=0),
        "micro/precision": precision_score(labels, pred, average="micro", zero_division=0),
        "micro/recall":    recall_score(labels, pred, average="micro",  zero_division=0),
        "jaccard/micro":   jaccard_score(labels, pred, average="micro",  zero_division=0),
        "jaccard/macro":   jaccard_score(labels, pred, average="macro",  zero_division=0),
        "jaccard/samples": jaccard_score(labels, pred, average="samples", zero_division=0),
    }

class TableLoggerCallback(TrainerCallback):
    def __init__(self):
        self.rows=[]; self.last_train_loss=None; self._printed=False; self._seen=set()
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
def _to_pylist(x):
    if isinstance(x, list): return x
    if hasattr(x, "tolist"): return x.tolist()
    return list(x)

def debug_print_samples(df_ctx: pd.DataFrame, ds: PastReqRepDataset, CFG, n=3):
    if n <= 0: return
    print("\n" + "="*72)
    print(f"[DEBUG] Showing {n} examples. LABEL ORDER: {LABELS}")
    print("="*72)

    rq0 = df_ctx[(df_ctx[CFG["ISREQ_COL"]]==1) & (df_ctx["context"].str.len()==0)].index.tolist()
    rp0 = df_ctx[(df_ctx[CFG["ISREQ_COL"]]==0) & (df_ctx["context"].str.len()==0)].index.tolist()
    rqa = df_ctx[(df_ctx[CFG["ISREQ_COL"]]==1) & (df_ctx["context"].str.len()>0)].index.tolist()
    rpa = df_ctx[(df_ctx[CFG["ISREQ_COL"]]==0) & (df_ctx["context"].str.len()>0)].index.tolist()

    chosen=[]
    if rq0: chosen.append(random.choice(rq0))
    if rp0 and len(chosen)<2: chosen.append(random.choice(rp0))
    pool=rqa+rpa; random.shuffle(pool)
    for idx in pool:
        if len(chosen)>=n: break
        if idx not in chosen: chosen.append(idx)
    if len(chosen)<n:
        all_idx=list(range(len(df_ctx))); random.shuffle(all_idx)
        for idx in all_idx:
            if len(chosen)>=n: break
            if idx not in chosen: chosen.append(idx)

    for i in chosen[:n]:
        row = df_ctx.iloc[i]
        enc_i = {k: v[i] for k, v in ds.enc.items()}
        input_ids = _to_pylist(enc_i["input_ids"])
        tokens    = tok.convert_ids_to_tokens(input_ids)
        typeids   = _to_pylist(enc_i.get("token_type_ids", [0]*len(input_ids)))
        gold_vec  = row["y_vec"].astype(int).tolist()
        print("-"*72)
        print(f"[Row {i}] email_id={row[CFG['EMAIL_COL']]} | sentence_idx={row[CFG['SENTIDX_COL']]} | is_request={row[CFG['ISREQ_COL']]}")
        print("TARGET :", textwrap.shorten(row['target'],  width=160, placeholder="…"))
        print("CONTEXT:", textwrap.shorten(row['context'], width=160, placeholder="…"))
        print("Gold vec :", gold_vec, "=>", [lbl for k,lbl in enumerate(LABELS) if gold_vec[k]==1])
        show = min(120, len(tokens))
        print("TOKENS    :", " ".join(tokens[:show]))
        print("TYPE_IDS  :", " ".join(map(str, typeids[:show])))
    print("="*72 + "\n")

# Trainer kwargs helper
def make_trainer_kwargs(extra: Dict[str, Any]) -> Dict[str, Any]:
    if TOK_KW is not None:
        extra[TOK_KW] = tok
    return extra

# Trainer & Train

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
    save_total_limit=2,       
    save_safetensors=False,
)

trainer_kwargs = make_trainer_kwargs(dict(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics_training,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG["early_stopping_patience"]),
               TableLoggerCallback()],
))
trainer = ASLTrainer(
    **trainer_kwargs,
    pos_weight=(pos_weight_tensor if CFG["use_pos_weight"] else None),
    asl_gamma_pos=CFG["asl_gamma_pos"],
    asl_gamma_neg=CFG["asl_gamma_neg"],
    asl_clip=CFG["asl_clip"],
)

debug_print_samples(df_train_ctx, train_ds, CFG, n=CFG["debug_samples"])
print("[INFO] Starting training…")
trainer.train()

# Multi-τ evaluation helpers
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

def eval_with_tau(logits: np.ndarray, labels: np.ndarray, tau: Union[float, np.ndarray]):
    probs = sigmoid_stable(logits)
    if isinstance(tau, np.ndarray):
        pred = (probs >= tau.reshape(1, -1)).astype(int)
    else:
        pred = (probs >= float(tau)).astype(int)
    return metrics_from_bin(labels, pred), pred, probs

def search_global_tau_for(metric_name: str, logits, labels, grid=None):
    if grid is None: grid = np.linspace(0.05, 0.95, 19)
    probs = sigmoid_stable(logits)
    best_tau, best_m, best_metrics = 0.5, -1.0, {}
    for tau in grid:
        pred = (probs >= tau).astype(int)
        m = metrics_from_bin(labels, pred)
        if m[metric_name] > best_m:
            best_m, best_tau, best_metrics = m[metric_name], float(tau), m
    return best_tau, best_metrics

def per_label_scut_f1(logits, labels):
    probs = sigmoid_stable(logits)
    C = probs.shape[1]; grid = np.linspace(0.05, 0.95, 19)
    tau_vec = np.zeros(C, np.float32)
    for c in range(C):
        y, p = labels[:,c], probs[:,c]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(y, (p>=t).astype(int), zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, float(t)
        tau_vec[c] = best_t
    pred = (probs >= tau_vec.reshape(1,-1)).astype(int)
    return tau_vec, metrics_from_bin(labels, pred)

def pcut_match_cardinality(logits, labels):
    probs = sigmoid_stable(logits)
    target_k = labels.sum(axis=1).mean()
    grid = np.linspace(0.05, 0.95, 19)
    best_tau, best_gap, best_metrics = 0.5, 1e9, {}
    for tau in grid:
        pred = (probs >= tau).astype(int)
        gap = abs(pred.sum(axis=1).mean() - target_k)
        if gap < best_gap: best_tau, best_gap, best_metrics = float(tau), gap, metrics_from_bin(labels, pred)
    return best_tau, best_metrics

# VAL: choose thresholds (forced τ@F1 to 0.60 for reporting)
print("[INFO] Evaluating on VAL (multi τ strategies)…")
val_out = trainer.predict(val_ds)
val_logits = val_out.predictions
val_labels = val_out.label_ids

tau_f1_auto, _  = search_global_tau_for("micro/f1",        val_logits, val_labels)
tau_jacc, _     = search_global_tau_for("jaccard/samples", val_logits, val_labels)
tau_vec,  val_s = per_label_scut_f1(val_logits, val_labels)
tau_pcut, _     = pcut_match_cardinality(val_logits, val_labels)

tau_f1 = float(CFG["FORCE_TAU_F1"])

print(f"[VAL][Global τ@F1] τ={tau_f1:.2f}  (auto-best was {tau_f1_auto:.2f}, forced for comparability)")
print(json.dumps(eval_with_tau(val_logits, val_labels, tau_f1)[0], indent=2))
print(f"[VAL][Global τ@Jaccard] τ={tau_jacc:.2f}  metrics=")
print(json.dumps(eval_with_tau(val_logits, val_labels, tau_jacc)[0], indent=2))
print(f"[VAL][Per-label SCut-F1] τ_vec={[float(round(x,2)) for x in tau_vec.tolist()]}  metrics=")
print(json.dumps(val_s, indent=2))
print(f"[VAL][PCut] τ={tau_pcut:.2f}  metrics=")
print(json.dumps(eval_with_tau(val_logits, val_labels, tau_pcut)[0], indent=2))

# TEST under forced τ=0.60 + alternatives
print(f"[INFO] Evaluating on TEST using Global τ@F1 (τ={tau_f1:.2f})…")
test_out = trainer.predict(test_ds)
test_logits = test_out.predictions
test_labels = test_out.label_ids

test_main, test_bin, test_probs = eval_with_tau(test_logits, test_labels, tau_f1)
test_scut, _, _  = eval_with_tau(test_logits, test_labels, tau_vec)
test_pcut, _, _  = eval_with_tau(test_logits, test_labels, tau_pcut)
test_jacc, _, _  = eval_with_tau(test_logits, test_labels, tau_jacc)

print("[TEST] Global τ@F1", json.dumps(test_main, indent=2))
print("[TEST] Per-label SCut-F1", json.dumps(test_scut, indent=2))
print("[TEST] PCut", json.dumps(test_pcut, indent=2))
print("[TEST] Global τ@Jaccard", json.dumps(test_jacc, indent=2))

# Per-label IoU under main τ
per_label_j = jaccard_score(test_labels, test_bin, average=None, zero_division=0)
print("[TEST] per-label Jaccard (IoU) under Global τ@F1")
for i, lbl in enumerate(LABELS):
    print(f"  {lbl:6s}: {per_label_j[i]:.4f}")

# Split TEST by REQUEST vs REPLY
def metrics_from_mask(labels_all: np.ndarray, preds_all: np.ndarray, mask_bool: np.ndarray, title: str):
    g = labels_all[mask_bool]
    p = preds_all[mask_bool]
    m = {
        "micro/f1":        f1_score(g, p, average="micro",  zero_division=0),
        "macro/f1":        f1_score(g, p, average="macro",  zero_division=0),
        "micro/precision": precision_score(g, p, average="micro", zero_division=0),
        "micro/recall":    recall_score(g, p, average="micro",  zero_division=0),
        "jaccard/micro":   jaccard_score(g, p, average="micro",  zero_division=0),
        "jaccard/macro":   jaccard_score(g, p, average="macro",  zero_division=0),
        "jaccard/samples": jaccard_score(g, p, average="samples", zero_division=0),
        "hamming_loss":    float(np.not_equal(g, p).mean()),
        "avg_true_k":      float(g.sum(axis=1).mean()),
        "avg_pred_k":      float(p.sum(axis=1).mean()),
    }
    print(f"[TEST][{title}] τ={tau_f1:.2f} metrics=")
    print(json.dumps(m, indent=2))
    return m

is_req_flags = df_test_ctx[CFG["ISREQ_COL"]].values.astype(int)
mask_req = (is_req_flags == 1)
mask_rep = (is_req_flags == 0)

m_req = metrics_from_mask(test_labels, test_bin, mask_req, "REQUEST")
m_rep = metrics_from_mask(test_labels, test_bin, mask_rep, "REPLY")

# Per-reply micro-F1 CSV
def per_reply_micro_f1(df_ctx: pd.DataFrame,
                       labels_all: np.ndarray,
                       preds_all: np.ndarray,
                       tau: float,
                       out_path: str):
    rows = []
    eid = df_ctx[CFG["EMAIL_COL"]].values
    is_rep = (df_ctx[CFG["ISREQ_COL"]].values == 0)
    seeds = df_ctx[CFG["SEED_COL"]].values
    unique_reply_eids = np.unique(eid[is_rep])

    for e in unique_reply_eids:
        m = (eid == e)
        g = labels_all[m]
        p = preds_all[m]
        f1_micro = f1_score(g, p, average="micro", zero_division=0)
        rows.append({
            "email_id": int(e),
            "seed": int(seeds[m][0]),
            "F1_micro": float(f1_micro),
            "num_sentences": int(m.sum()),
            "tau": float(tau)
        })
    out_df = pd.DataFrame(rows).sort_values(["seed","email_id"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    print(f"[TEST] Wrote per-reply micro-F1 → {out_path}")

per_reply_out = os.path.join(CFG["output_dir"], "per_reply_f1_PastSameEmail.csv")
per_reply_micro_f1(df_test_ctx, test_labels, test_bin, tau_f1, per_reply_out)

# Save predictions (+ per-row Jaccard) to CSV
if CFG["save_test_predictions"]:
    pred_cols = [f"pred_prob_{l}" for l in LABELS]
    probs_save = test_probs
    preds_save = test_bin

    save_df = df_test_ctx.copy()
    save_df[pred_cols] = pd.DataFrame(probs_save, index=save_df.index)
    save_df[f"pred_labels_tau_{tau_f1:.2f}"] = [
        ",".join([LABELS[k] for k,v in enumerate(probs_save[i]) if v >= tau_f1])
        for i in range(len(save_df))
    ]
    labels_bool = test_labels.astype(bool)
    preds_bool  = preds_save.astype(bool)
    inter = np.logical_and(labels_bool, preds_bool).sum(axis=1).astype(np.float32)
    union = np.logical_or (labels_bool, preds_bool).sum(axis=1).astype(np.float32)
    row_j = np.divide(inter, union, out=np.ones_like(inter, dtype=np.float32), where=(union!=0))
    save_df[f"jaccard_row_tau_{tau_f1:.2f}"] = row_j

    out_csv = os.path.join(CFG["output_dir"], "test_predictions_targetfirst_pastsameemail_asl.csv")
    save_df.to_csv(out_csv, index=False)
    print("[DONE] Wrote:", out_csv)
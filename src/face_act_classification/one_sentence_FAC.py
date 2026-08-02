print("Sentence only")

import os, re, ast, json, math, textwrap, random
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    "ISREQ_COL":    "is_request",   # 1=request, 0=reply
    "SEED_COL":     "seed",
    "PAIR_COL":     "pair_idx",

    # Fixed head order
    "LABELS": ["HNeg+","HNeg-","HPos+","HPos-","Neutral","SNeg+","SNeg-","SPos+","SPos-"],

    "model_name": "bert-base-uncased",
    "max_length": 128,

    "output_dir": "bert_multi_label_faceacts_sentence_only_asl",
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
    "print_examples": 0,
    "save_test_predictions": True,
    "FORCE_TAU_F1": 0.60,
}

os.makedirs(CFG["output_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
if torch.cuda.is_available(): torch.cuda.manual_seed_all(CFG["seed"])

def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    np.clip(x, -50.0, 50.0, out=x)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

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

def load_split(csv_path: str, CFG):
    df = pd.read_csv(csv_path)
    need = [CFG["TEXT_COL"], CFG["LABEL_COL"], CFG["EMAIL_COL"], CFG["SENTIDX_COL"],
            CFG["ISREQ_COL"], CFG["SEED_COL"], CFG["PAIR_COL"]]
    for c in need:
        assert c in df.columns, f"Column '{c}' not found in {csv_path}"
    for c in [CFG["EMAIL_COL"], CFG["SENTIDX_COL"], CFG["ISREQ_COL"], CFG["SEED_COL"], CFG["PAIR_COL"]]:
        df[c] = df[c].astype(int)
    df["gold_list"] = df[CFG["LABEL_COL"]].map(parse_labels)
    df["y_vec"]     = df["gold_list"].map(to_multi_hot)
    return df.reset_index(drop=True)

df_train = load_split(CFG["train_csv"], CFG)
df_val   = load_split(CFG["val_csv"],   CFG)
df_test  = load_split(CFG["test_csv"],  CFG)
print(f"[INFO] Loaded rows: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

tok = AutoTokenizer.from_pretrained(CFG["model_name"])

class SentDataset(Dataset):
    """
    Keeps df for later (to align per-row flags/id when splitting metrics).
    """
    def __init__(self, df: pd.DataFrame, cfg: Dict[str, Any]):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.texts = self.df[cfg["TEXT_COL"]].astype(str).tolist()
        self.labels = np.stack(self.df["y_vec"].values)  # [N, C]
        self.enc = tok(
            self.texts,
            padding=True,
            truncation="only_first",
            max_length=cfg["max_length"],
            return_token_type_ids=True,
            return_tensors=None
        )
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k,v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

train_ds = SentDataset(df_train, CFG)
val_ds   = SentDataset(df_val,   CFG)
test_ds  = SentDataset(df_test,  CFG)


pos_weight_tensor = None
if CFG["use_pos_weight"]:
    y = np.stack(df_train["y_vec"].values)
    pos = y.sum(axis=0); neg = y.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).astype(np.float32)
    pos_weight_tensor = torch.tensor(pos_weight)
    print("[INFO] pos_weight:", {lbl: float(round(w,2)) for lbl,w in zip(LABELS, pos_weight)})


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, reduction="mean"):
        super().__init__()
        self.gp, self.gn, self.clip, self.eps, self.reduction = gamma_pos, gamma_neg, clip, eps, reduction
    def forward(self, logits, targets, pos_weight=None):
        x = torch.sigmoid(logits)
        x_pos = x; x_neg = 1.0 - x
        if self.clip and self.clip>0: x_neg = (x_neg + self.clip).clamp(max=1.0)
        log_pos = torch.log(x_pos.clamp(min=self.eps))
        log_neg = torch.log(x_neg.clamp(min=self.eps))
        if self.gp > 0: log_pos = log_pos * (1.0 - x_pos) ** self.gp
        if self.gn > 0: log_neg = log_neg * (x_pos) ** self.gn
        loss = -(targets * log_pos + (1.0 - targets) * log_neg)
        if pos_weight is not None:
            loss = loss * (1.0 + targets * (pos_weight.to(loss.device) - 1.0))
        return loss.mean()

model = AutoModelForSequenceClassification.from_pretrained(
    CFG["model_name"],
    num_labels=len(LABELS),
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id
)

class ASLTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.asl = AsymmetricLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")           
        outputs = model(**inputs)
        logits = outputs.logits                
        loss = self.asl(logits, labels, pos_weight=self.pos_weight)
        return (loss, outputs) if return_outputs else loss

def compute_metrics_full(eval_pred):
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
        self.rows = []; self.last_train_loss=None; self._printed=False; self._seen=set()
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs: self.last_train_loss = float(logs["loss"])
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        g = lambda k: metrics.get(f"eval_{k}", metrics.get(k))
        epoch = int(round(state.epoch or 0)) if state.epoch is not None else (len(self.rows)+1)
        key = (epoch, g("loss"))
        if key in self._seen: return
        self._seen.add(key)
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
        def fmt(x): return "—" if x is None else f"{x:.6f}"
        if not self._printed:
            print("\nEpoch\tTraining Loss\tValidation Loss\tMicro/f1\tMacro/f1\tMicro/precision\tMicro/recall\tJaccard/micro\tJaccard/macro\tJaccard/samples")
            self._printed=True
        print(f"{row['epoch']}\t{fmt(row['train_loss'])}\t{fmt(row['val_loss'])}\t{fmt(row['micro/f1'])}\t{fmt(row['macro/f1'])}\t{fmt(row['micro/precision'])}\t{fmt(row['micro/recall'])}\t{fmt(row['jaccard/micro'])}\t{fmt(row['jaccard/macro'])}\t{fmt(row['jaccard/samples'])}")
    def on_train_end(self, args, state, control, **kwargs):
        if self.rows:
            out_csv=os.path.join(args.output_dir, "epoch_log_summary.csv")
            pd.DataFrame(self.rows).to_csv(out_csv, index=False)
            print(f"[LOG] Wrote epoch summary → {out_csv}")

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
    logging_steps=50
)

trainer = ASLTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tok,
    compute_metrics=compute_metrics_full,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG["early_stopping_patience"]),
               TableLoggerCallback()],
    pos_weight=(pos_weight_tensor if CFG["use_pos_weight"] else None),
    gamma_pos=CFG["asl_gamma_pos"],
    gamma_neg=CFG["asl_gamma_neg"],
    clip=CFG["asl_clip"],
)

print("[INFO] Starting training…")
trainer.train()

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
    pred  = (probs >= tau).astype(int) if isinstance(tau, np.ndarray) else (probs >= float(tau)).astype(int)
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
        best_f1, best_t = -1.0, 0.5
        y, p = labels[:,c], probs[:,c]
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

print("[INFO] Evaluating on VAL (multi τ strategies)…")
val_out = trainer.predict(val_ds)
val_logits = val_out.predictions
val_labels = val_out.label_ids

tau_f1_auto, _ = search_global_tau_for("micro/f1",        val_logits, val_labels) 
tau_jacc,     _ = search_global_tau_for("jaccard/samples", val_logits, val_labels)
tau_vec,  val_scut = per_label_scut_f1(val_logits, val_labels)
tau_pcut, _     = pcut_match_cardinality(val_logits, val_labels)

tau_f1 = float(CFG["FORCE_TAU_F1"])

print(f"[VAL][Global τ@F1] τ={tau_f1:.2f}  metrics={json.dumps(eval_with_tau(val_logits, val_labels, tau_f1)[0], indent=2)}")
print(f"[VAL][Global τ@Jaccard] τ={tau_jacc:.2f}  metrics={json.dumps(eval_with_tau(val_logits, val_labels, tau_jacc)[0], indent=2)}")
print(f"[VAL][Per-label SCut-F1] τ_vec={np.round(tau_vec,2).tolist()}  metrics={json.dumps(per_label_scut_f1(val_logits, val_labels)[1], indent=2)}")
print(f"[VAL][PCut] τ={tau_pcut:.2f}  metrics={json.dumps(eval_with_tau(val_logits, val_labels, tau_pcut)[0], indent=2)}")
print(f"[INFO] Evaluating on TEST using Global τ@F1 (τ={tau_f1:.2f})…")
test_out = trainer.predict(test_ds)
test_logits = test_out.predictions
test_labels = test_out.label_ids

test_main, test_pred_bin, test_probs = eval_with_tau(test_logits, test_labels, tau_f1)
print("[TEST] Global τ@F1", json.dumps(test_main, indent=2))
print("[TEST] Per-label SCut-F1", json.dumps(eval_with_tau(test_logits, test_labels, tau_vec)[0], indent=2))
print("[TEST] PCut", json.dumps(eval_with_tau(test_logits, test_labels, tau_pcut)[0], indent=2))
print("[TEST] Global τ@Jaccard", json.dumps(eval_with_tau(test_logits, test_labels, tau_jacc)[0], indent=2))

per_label_j = jaccard_score(test_labels, test_pred_bin, average=None, zero_division=0)
print("[TEST] per-label Jaccard (IoU) under Global τ@F1")
for i, lbl in enumerate(LABELS):
    print(f"  {lbl:6s}: {per_label_j[i]:.4f}")


# Split TEST metrics: REQUEST vs REPLY

is_req = df_test[CFG["ISREQ_COL"]].astype(int).to_numpy()
mask_req = (is_req == 1)
mask_rep = (is_req == 0)

def metrics_subset(labels_all, pred_all, mask, title):
    m = metrics_from_bin(labels_all[mask], pred_all[mask])
    print(f"[TEST][{title}] τ={tau_f1:.2f} metrics=")
    print(json.dumps(m, indent=2))
    return m

_ = metrics_subset(test_labels, test_pred_bin, mask_req, "REQUEST")
_ = metrics_subset(test_labels, test_pred_bin, mask_rep, "REPLY")


# Per-reply micro-F1 CSV
def per_reply_micro_f1(df_split: pd.DataFrame, labels_all: np.ndarray, pred_all: np.ndarray) -> pd.DataFrame:
    # Only replies
    rep = df_split[df_split[CFG["ISREQ_COL"]] == 0].copy()
    idx = rep.index.to_numpy()
    emails = rep[CFG["EMAIL_COL"]].to_numpy()
    seeds  = rep[CFG["SEED_COL"]].to_numpy()
    pairs  = rep[CFG["PAIR_COL"]].to_numpy()

    rows = []
    for em in np.unique(emails):
        em_mask_glob = (df_split[CFG["EMAIL_COL"]].to_numpy() == em) & (df_split[CFG["ISREQ_COL"]].to_numpy() == 0)
        if not em_mask_glob.any():
            continue
        y_true = labels_all[em_mask_glob]
        y_pred = pred_all[em_mask_glob]
        f1m = f1_score(y_true, y_pred, average="micro", zero_division=0)

        j0 = np.argmax(em_mask_glob.astype(int))
        rows.append({
            "seed": int(df_split.iloc[j0][CFG["SEED_COL"]]),
            "pair_idx": int(df_split.iloc[j0][CFG["PAIR_COL"]]),
            "reply_email_id": int(em),
            "F1_micro": float(f1m),
        })
    return pd.DataFrame(rows)

per_reply_df = per_reply_micro_f1(df_test, test_labels, test_pred_bin)
per_reply_csv = os.path.join(CFG["output_dir"], "per_reply_f1_SentenceOnly.csv")
per_reply_df.to_csv(per_reply_csv, index=False)
print(f"[TEST] Wrote per-reply micro-F1 → {per_reply_csv}")


if CFG["save_test_predictions"]:
    probs_save = test_probs
    preds_save = test_pred_bin

    save_df = df_test.copy()
    for k, lbl in enumerate(LABELS):
        save_df[f"pred_prob_{lbl}"] = probs_save[:, k]

    tau = float(CFG["FORCE_TAU_F1"])
    save_df[f"pred_labels_tau_{tau:.2f}"] = [
        ",".join([LABELS[k] for k,v in enumerate(probs_save[i]) if v >= tau])
        for i in range(len(save_df))
    ]
    labels_bool = test_labels.astype(bool)
    preds_bool  = preds_save.astype(bool)
    inter = np.logical_and(labels_bool, preds_bool).sum(axis=1).astype(np.float32)
    union = np.logical_or (labels_bool, preds_bool).sum(axis=1).astype(np.float32)
    row_j = np.divide(inter, union, out=np.ones_like(inter, dtype=np.float32), where=(union!=0))
    save_df[f"jaccard_row_tau_{tau:.2f}"] = row_j

    out_csv = os.path.join(CFG["output_dir"], "test_predictions_sentence_only_asl.csv")
    save_df.to_csv(out_csv, index=False)
    print("[DONE] Wrote:", out_csv)
# the great version that we used for the paper
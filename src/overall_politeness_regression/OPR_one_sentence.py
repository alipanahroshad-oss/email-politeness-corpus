#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ======== REPOSITORY PATHS ========
# Run this script from the repository root.
DOC_CSV = "data/corpus/email_text_gold_three_dimensions_politeness_score_with_seed_correct.csv"
SENT_CSV = "data/corpus/sentences_with_golden_face_act.csv"

# Output directory produced by src/face_act_classification/one_sentence_FAC.py.
# The resolver below accepts either this parent directory or a direct checkpoint directory.
ASL_MODEL_DIR = "bert_multi_label_faceacts_sentence_only_asl"
ASL_TOK_NAME = "bert-base-uncased"

# -------- Standard imports --------
import os, ast, glob, json, math, random, copy
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pandas.api.types import CategoricalDtype
from sklearn.model_selection import GroupShuffleSplit

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, get_linear_schedule_with_warmup

# -------- Global config --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42
OUT_DIR = "./end2end_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Columns / targets
DOC_ID_COL   = "email_id"
DOC_TEXT_COL = "text_email"
SEED_COL     = "seed"
# Dataset column names are kept unchanged for reproducibility.
# In the paper, Structural_Politeness_and_Politeness_Markers is referred to as
# Positive Face Saving, and Tone_and_Overall_Consideration as Negative Face Saving.
TARGETS = [
    "Directness_vs_Indirectness__GOLD",
    "Structural_Politeness_and_Politeness_Markers__GOLD",
    "Tone_and_Overall_Consideration__GOLD"
]

SENT_TEXT_COL = "covered_text"
SENT_IDX_COL  = "sentence_idx"
FA_COL        = "GoldFaceAct"

# IMPORTANT: label list used to read the model's output columns "prob_<label>"
FA_LABS = ["HNeg+","HNeg-","HPos+","HPos-","Neutral","SNeg+","SNeg-","SPos+","SPos-"]

# ---- LOCKED FEATURE ORDERS
SUM9_ORDER  = [f"sumprob_{l}" for l in FA_LABS]
SUM13_EXTRA = ["sum_H", "sum_S", "sum_praise", "sum_threat"]
SUM13_ORDER = SUM9_ORDER + SUM13_EXTRA

# Doc-level BERT encoder (text models)
DOC_BERT_NAME = "bert-base-uncased"

# Training knobs (can tune later)
BERT_LR = 2e-5
BERT_EPOCHS = 5
BERT_BS = 8
BERT_DROPOUT = 0.1
BERT_MAXLEN = 512

MLP_HIDDEN = 256
MLP_DROPOUT = 0.2
MLP_LR = 1e-3
MLP_EPOCHS = 60
MLP_BS = 32

# Which families/variants to run (both MT & ST will be produced)
RUN_BERT_TEXT = True
RUN_PredFA_MLP = True
RUN_BERT_PLUS_PredFA = True

DO_MULTI_TASK = True
DO_SINGLE_TASK = True

# ============================================================
# Utils
# ============================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def _decategorize_series(s: pd.Series) -> pd.Series:
    if isinstance(s.dtype, CategoricalDtype):
        s = s.astype(object)
    return s

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, CategoricalDtype):
            df[col] = df[col].astype(object)
    if DOC_ID_COL in df.columns:    df[DOC_ID_COL] = df[DOC_ID_COL].astype(str)
    if SENT_TEXT_COL in df.columns: df[SENT_TEXT_COL] = df[SENT_TEXT_COL].astype(str)
    if FA_COL in df.columns:        df[FA_COL] = df[FA_COL].astype(str)
    if SEED_COL in df.columns:      df[SEED_COL] = pd.to_numeric(df[SEED_COL], errors="coerce")
    if SENT_IDX_COL in df.columns:  df[SENT_IDX_COL] = pd.to_numeric(df[SENT_IDX_COL], errors="coerce")
    return df.reset_index(drop=True)

def grouped_split_70_15_15_index(groups: np.ndarray, random_state=42) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    idx = np.arange(len(groups))
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state)
    tr, tmp = next(gss1.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_state)
    va_rel, te_rel = next(gss2.split(tmp, groups=groups[tmp]))
    va, te = tmp[va_rel], tmp[te_rel]
    return tr, va, te

# ============================================================
# FA sentence tagger inference (BERT+ASL checkpoint)
# ============================================================
def _has_model_files(d):
    return any(os.path.isfile(os.path.join(d, fn)) for fn in [
        "pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt.index", "flax_model.msgpack"
    ])

def resolve_hf_model_dir(parent_dir: str) -> str:
    """
    Resolve a Hugging Face model directory.

    Preference order:
      1) model files directly in ``parent_dir``;
      2) the checkpoint recorded as ``best_model_checkpoint`` by Trainer;
      3) the most recent available checkpoint as a fallback.

    This keeps OPR inference aligned with the best FAC checkpoint selected on
    validation data by the FAC training script.
    """
    if _has_model_files(parent_dir):
        return parent_dir

    # Trainer normally stores trainer_state.json inside checkpoint-* folders.
    state_paths = []
    root_state = os.path.join(parent_dir, "trainer_state.json")
    if os.path.isfile(root_state):
        state_paths.append(root_state)

    state_paths.extend(
        sorted(
            glob.glob(os.path.join(parent_dir, "checkpoint-*", "trainer_state.json")),
            key=os.path.getmtime,
            reverse=True,
        )
    )

    for state_path in state_paths:
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            best_ckpt = state.get("best_model_checkpoint")
            if not best_ckpt:
                continue

            candidates = [best_ckpt]
            # If the stored path came from another machine or working directory,
            # the checkpoint basename still identifies the corresponding folder.
            candidates.append(os.path.join(parent_dir, os.path.basename(best_ckpt)))
            if not os.path.isabs(best_ckpt):
                candidates.append(os.path.join(parent_dir, best_ckpt))

            for candidate in candidates:
                if _has_model_files(candidate):
                    return candidate
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    # Fallback only if no usable best-checkpoint metadata is available.
    ckpts = sorted(
        glob.glob(os.path.join(parent_dir, "checkpoint-*")),
        key=os.path.getmtime,
        reverse=True,
    )
    for ck in ckpts:
        if _has_model_files(ck):
            return ck

    raise OSError(f"No model files found in or under '{parent_dir}'")

class SentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, tokenizer, max_length=128):
        df = df.copy()
        df[DOC_ID_COL] = df[DOC_ID_COL].astype(str)
        df[SENT_IDX_COL] = pd.to_numeric(df[SENT_IDX_COL], errors="coerce").fillna(0).astype(int)
        enc = tokenizer(df[text_col].astype(str).tolist(), padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        self.input_ids = enc["input_ids"]; self.attn = enc["attention_mask"]
        self.doc_ids = df[DOC_ID_COL].tolist()
        self.sent_idx = df[SENT_IDX_COL].tolist()
    def __len__(self): return self.input_ids.size(0)
    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attn[i],
            "doc_id": self.doc_ids[i],
            "sent_idx": self.sent_idx[i],
        }

def predict_fa_probs(df_split: pd.DataFrame, model_dir: str, tok_name: str,
                     batch=64, max_len=128) -> pd.DataFrame:
    if len(df_split)==0:
        return pd.DataFrame(columns=[DOC_ID_COL, SENT_IDX_COL] + [f"prob_{l}" for l in FA_LABS])
    model_dir = resolve_hf_model_dir(model_dir)
    tok = AutoTokenizer.from_pretrained(tok_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    mdl.eval()

    ds = SentDataset(df_split, SENT_TEXT_COL, tok, max_length=max_len)
    dl = DataLoader(ds, batch_size=batch, shuffle=False)

    doc_ids, sids, allP = [], [], []
    with torch.no_grad():
        for b in dl:
            logits = mdl(input_ids=b["input_ids"].to(DEVICE),
                         attention_mask=b["attention_mask"].to(DEVICE)).logits
            P = torch.sigmoid(logits).cpu().numpy()
            allP.append(P)
            doc_ids.extend(b["doc_id"]); sids.extend(b["sent_idx"])
    P = np.vstack(allP)
    out = {DOC_ID_COL: list(map(str, doc_ids)), SENT_IDX_COL: list(map(int, sids))}
    for j, l in enumerate(FA_LABS): out[f"prob_{l}"] = P[:, j]
    dfp = pd.DataFrame(out)
    dfp[DOC_ID_COL] = dfp[DOC_ID_COL].astype(str)
    dfp[SENT_IDX_COL] = pd.to_numeric(dfp[SENT_IDX_COL], errors="coerce").fillna(0).astype(int)
    return dfp

# ============================================================
# PredFA aggregation — HSPT-13 (9 per-label + H/S + praise/threat) + z-score
# ============================================================
def aggregate_predfa_hspt13(dfp: pd.DataFrame, doc_id_col: str, sent_idx_col: str) -> pd.DataFrame:
    """
    Build **HSPT-13** PredFA features per email.

    Returns columns in LOCKED order: [doc_id] + SUM13_ORDER
    """
    need = [doc_id_col, sent_idx_col] + [f"prob_{l}" for l in FA_LABS]
    for c in need:
        if c not in dfp.columns:
            raise ValueError(f"Missing PredFA column: {c}")

    dfp = dfp.copy()
    dfp[doc_id_col]   = _decategorize_series(dfp[doc_id_col]).astype(str)
    dfp[sent_idx_col] = pd.to_numeric(_decategorize_series(dfp[sent_idx_col]), errors="coerce").fillna(0).astype(int)
    for l in FA_LABS:
        col = f"prob_{l}"
        dfp[col] = pd.to_numeric(_decategorize_series(dfp[col]), errors="coerce").fillna(0.0).astype(float)

    dfp = dfp.sort_values([doc_id_col, sent_idx_col], kind="mergesort", ignore_index=True)

    rows = []
    for doc_id, g in dfp.groupby(doc_id_col, sort=False):
        row = {doc_id_col: doc_id}
        # 9 per-label sums
        for l in FA_LABS:
            row[f"sumprob_{l}"] = float(g[f"prob_{l}"].sum())

        # 2 H/S totals (neutral excluded)
        sum_H = row["sumprob_HNeg+"] + row["sumprob_HNeg-"] + row["sumprob_HPos+"] + row["sumprob_HPos-"]
        sum_S = row["sumprob_SNeg+"] + row["sumprob_SNeg-"] + row["sumprob_SPos+"] + row["sumprob_SPos-"]

        # 2 praise/threat (neutral excluded) — “+” = saving/praise, “−” = threat
        sum_praise = row["sumprob_HPos+"] + row["sumprob_HNeg+"] + row["sumprob_SPos+"] + row["sumprob_SNeg+"]
        sum_threat = row["sumprob_HPos-"] + row["sumprob_HNeg-"] + row["sumprob_SPos-"] + row["sumprob_SNeg-"]

        row["sum_H"] = float(sum_H)
        row["sum_S"] = float(sum_S)
        row["sum_praise"] = float(sum_praise)
        row["sum_threat"] = float(sum_threat)

        rows.append(row)

    df_out = pd.DataFrame(rows)
    # Enforce locked column order
    keep_cols = [doc_id_col] + SUM13_ORDER
    for c in keep_cols:
        if c not in df_out.columns:
            df_out[c] = 0.0
    return df_out[keep_cols]

def align_predfa_to_split(F_pred: pd.DataFrame, df_doc: pd.DataFrame, idx: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    F = df_doc.loc[idx, [DOC_ID_COL]].astype({DOC_ID_COL: str}).merge(
        F_pred.astype({DOC_ID_COL: str}), on=DOC_ID_COL, how="left"
    ).fillna(0.0)
    keep_cols = [DOC_ID_COL] + SUM13_ORDER
    for c in keep_cols:
        if c not in F.columns:
            F[c] = 0.0
    F = F[keep_cols]
    return F[SUM13_ORDER].to_numpy(dtype=np.float32), SUM13_ORDER

def zscore_fit_transform(Xtr: np.ndarray, Xva: np.ndarray, Xte: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xtr_z = (Xtr - mu) / sd
    Xva_z = (Xva - mu) / sd
    Xte_z = (Xte - mu) / sd
    return Xtr_z, Xva_z, Xte_z, mu, sd

# ============================================================
# Doc-level datasets and models
# ============================================================
class DocBertDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, targets: List[str], tokenizer):
        self.df = df.reset_index(drop=True)
        enc = tokenizer(df[text_col].astype(str).tolist(), padding=True, truncation=True,
                        max_length=BERT_MAXLEN, return_tensors="pt")
        self.ids = enc["input_ids"]; self.attn = enc["attention_mask"]
        self.y = torch.tensor(df[targets].to_numpy(np.float32))
    def __len__(self): return self.ids.size(0)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.attn[i], "targets": self.y[i]}

class DocBertFusionDataset(Dataset):
    """Same as DocBertDataset but carries aligned PredFA vectors for fusion."""
    def __init__(self, df: pd.DataFrame, text_col: str, targets: List[str], tokenizer, predfa: np.ndarray):
        assert len(df) == len(predfa), "PredFA feature rows must align with df rows."
        self.df = df.reset_index(drop=True)
        enc = tokenizer(df[text_col].astype(str).tolist(), padding=True, truncation=True,
                        max_length=BERT_MAXLEN, return_tensors="pt")
        self.ids = enc["input_ids"]; self.attn = enc["attention_mask"]
        self.y = torch.tensor(df[targets].to_numpy(np.float32))
        self.predfa = torch.tensor(predfa, dtype=torch.float32)
    def __len__(self): return self.ids.size(0)
    def __getitem__(self, i):
        return {
            "input_ids": self.ids[i],
            "attention_mask": self.attn[i],
            "targets": self.y[i],
            "predfa": self.predfa[i],
        }

class DocPredFADataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return self.X.size(0)
    def __getitem__(self, i): return {"x": self.X[i], "targets": self.Y[i]}

# ================== FUSION MODEL (concat → 128 ReLU → 1 per target) ==================
class BertDocRegressor(nn.Module):
    def __init__(self, encoder_name=DOC_BERT_NAME, n_targets=3, fusion_dim=0, dropout=BERT_DROPOUT):
        super().__init__()
        self.enc = AutoModel.from_pretrained(encoder_name)
        hdim = self.enc.config.hidden_size
        self.fusion_dim = int(fusion_dim)
        in_dim = hdim + (self.fusion_dim if self.fusion_dim > 0 else 0)

        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(n_targets)])

    def forward(self, input_ids, attention_mask, predfa: Optional[torch.Tensor]=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = (out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None)
               else out.last_hidden_state[:, 0, :])
        if (predfa is None) or (self.fusion_dim == 0):
            z = cls
        else:
            if predfa.size(-1) != self.fusion_dim:
                D = predfa.size(-1)
                if D > self.fusion_dim:
                    predfa = predfa[..., :self.fusion_dim]
                else:
                    pad = torch.zeros(predfa.size(0), self.fusion_dim - D, device=predfa.device, dtype=predfa.dtype)
                    predfa = torch.cat([predfa, pad], dim=-1)
            z = torch.cat([cls, predfa], dim=-1)
        z = self.dropout(z)
        h = self.hidden(z)
        outs = [head(h) for head in self.heads]
        return torch.cat(outs, dim=-1)

# ============================================================
# Train / Eval helpers
# ============================================================
def mae_spearman(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> Dict:
    out = {"per_target": {}}
    rhos=[]
    for i, name in enumerate(target_names):
        yt, yp = y_true[:,i], y_pred[:,i]
        mae = float(np.mean(np.abs(yt-yp)))
        rho, _ = spearmanr(yt, yp)
        rho = float(rho) if (rho is not None and not np.isnan(rho)) else 0.0
        rhos.append(rho)
        out["per_target"][name] = {"MAE": mae, "Spearman": rho}
    out["rho_macro"] = float(np.mean(rhos))
    return out

def save_metrics(outdir: str, best_cfg: Dict, Yte: np.ndarray, Pte: np.ndarray,
                 Yva: Optional[np.ndarray], Pva: Optional[np.ndarray], target_names: List[str]):
    ensure_dir(outdir)
    m_test = mae_spearman(Yte, Pte, target_names)
    m_val  = None
    if Yva is not None and Pva is not None:
        m_val = mae_spearman(Yva, Pva, target_names)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({"best_cfg": best_cfg, "val": m_val, "test": m_test, "targets": target_names}, f, indent=2)
    np.save(os.path.join(outdir, "test_y.npy"), Yte)
    np.save(os.path.join(outdir, "test_pred.npy"), Pte)

# ============================================================
# Family 1: BERT (text-only) and BERT+PredFA (fusion)
# ============================================================
def run_bert_family(df_doc, tr_idx, va_idx, te_idx, predfa_feats=None, tag="bert_text", multi_task=True):
    sub = f"{tag}_{'mt' if multi_task else 'st'}"
    outdir = os.path.join(OUT_DIR, sub); ensure_dir(outdir)
    tk = AutoTokenizer.from_pretrained(DOC_BERT_NAME, use_fast=True)

    df_tr = df_doc.loc[tr_idx].reset_index(drop=True)
    df_va = df_doc.loc[va_idx].reset_index(drop=True)
    df_te = df_doc.loc[te_idx].reset_index(drop=True)

    if predfa_feats is None:
        ds_tr = DocBertDataset(df_tr, DOC_TEXT_COL, TARGETS, tk)
        ds_va = DocBertDataset(df_va, DOC_TEXT_COL, TARGETS, tk)
        ds_te = DocBertDataset(df_te, DOC_TEXT_COL, TARGETS, tk)
        fusion_dim = 0
    else:
        Xtr, Xva, Xte = predfa_feats
        ds_tr = DocBertFusionDataset(df_tr, DOC_TEXT_COL, TARGETS, tk, Xtr)
        ds_va = DocBertFusionDataset(df_va, DOC_TEXT_COL, TARGETS, tk, Xva)
        ds_te = DocBertFusionDataset(df_te, DOC_TEXT_COL, TARGETS, tk, Xte)
        fusion_dim = Xtr.shape[1]

    dl_tr = DataLoader(ds_tr, batch_size=BERT_BS, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BERT_BS, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=BERT_BS, shuffle=False)

    if multi_task:
        model = BertDocRegressor(encoder_name=DOC_BERT_NAME, n_targets=3, fusion_dim=fusion_dim).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)
        steps_total = len(dl_tr)*BERT_EPOCHS; warmup = int(0.06*steps_total)
        sch = get_linear_schedule_with_warmup(opt, warmup, steps_total)
        loss_fn = nn.SmoothL1Loss()

        best_rho = -1e9; best_state = None
        for _ in range(1, BERT_EPOCHS+1):
            model.train()
            for b in dl_tr:
                opt.zero_grad()
                ids=b["input_ids"].to(DEVICE); att=b["attention_mask"].to(DEVICE); y=b["targets"].to(DEVICE)
                pf = b.get("predfa", None); pf = pf.to(DEVICE) if pf is not None else None
                pr = model(ids, att, pf)
                loss = loss_fn(pr, y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step()
            # dev
            model.eval(); Yv=[]; Pv=[]
            with torch.no_grad():
                for b in dl_va:
                    ids=b["input_ids"].to(DEVICE); att=b["attention_mask"].to(DEVICE); y=b["targets"].to(DEVICE)
                    pf = b.get("predfa", None); pf = pf.to(DEVICE) if pf is not None else None
                    pr = model(ids, att, pf); Yv.append(y.cpu().numpy()); Pv.append(pr.cpu().numpy())
            Yv=np.concatenate(Yv,0); Pv=np.concatenate(Pv,0)
            rho = mae_spearman(Yv, Pv, TARGETS)["rho_macro"]
            if rho > best_rho:
                best_rho = rho; best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        # test
        model.eval(); Ys=[]; Ps=[]
        with torch.no_grad():
            for b in dl_te:
                ids=b["input_ids"].to(DEVICE); att=b["attention_mask"].to(DEVICE); y=b["targets"].to(DEVICE)
                pf = b.get("predfa", None); pf = pf.to(DEVICE) if pf is not None else None
                pr = model(ids, att, pf); Ys.append(y.cpu().numpy()); Ps.append(pr.cpu().numpy())
        Yt=np.concatenate(Ys,0); Pt=np.concatenate(Ps,0)
        # dev dump
        model.eval(); Ys=[]; Ps=[]
        with torch.no_grad():
            for b in dl_va:
                ids=b["input_ids"].to(DEVICE); att=b["attention_mask"].to(DEVICE); y=b["targets"].to(DEVICE)
                pf = b.get("predfa", None); pf = pf.to(DEVICE) if pf is not None else None
                pr = model(ids, att, pf); Ys.append(y.cpu().numpy()); Ps.append(pr.cpu().numpy())
        Yv=np.concatenate(Ys,0); Pv=np.concatenate(Ps,0)
        save_metrics(outdir, {"variant":"multi_task", "fusion":"concat_128_relu_1"}, Yt, Pt, Yv, Pv, TARGETS)
        return {"Y":Yt, "P":Pt}

    # single-task: 3 independent runs
    preds_te = np.zeros((len(te_idx), 3), dtype=np.float32); ys_te = df_doc.loc[te_idx, TARGETS].to_numpy(np.float32)
    preds_va = np.zeros((len(va_idx), 3), dtype=np.float32); ys_va = df_doc.loc[va_idx, TARGETS].to_numpy(np.float32)
    for t_i, _ in enumerate(TARGETS):
        model = BertDocRegressor(encoder_name=DOC_BERT_NAME, n_targets=1, fusion_dim=fusion_dim).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)
        steps_total = len(dl_tr)*BERT_EPOCHS; warmup = int(0.06*steps_total)
        sch = get_linear_schedule_with_warmup(opt, warmup, steps_total)
        loss_fn = nn.SmoothL1Loss()
        best_rho=-1e9; best_state=None
        for _ in range(1, BERT_EPOCHS+1):
            model.train()
            for b in dl_tr:
                opt.zero_grad()
                ids=b["input_ids"].to(DEVICE); att=b["attention_mask"].to(DEVICE)
                y=b["targets"][:,t_i].unsqueeze(1).to(DEVICE)
                pf = b.get("predfa", None); pf = pf.to(DEVICE) if pf is not None else None
                pr = model(ids, att, pf)
                loss = loss_fn(pr, y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); sch.step()
            # dev
            model.eval(); Yv=[]; Pv=[]
            with torch.no_grad():
                for b in dl_va:
                    ids=b["input_ids"].to(DEVICE); att=b["attention_mask"].to(DEVICE)
                    y=b["targets"][:,t_i].unsqueeze(1).to(DEVICE)
                    pf = b.get("predfa", None); pf = pf.to(DEVICE) if pf is not None else None
                    pr = model(ids, att, pf); Yv.append(y.cpu().numpy()); Pv.append(pr.cpu().numpy())
            Yv=np.concatenate(Yv,0).squeeze(1); Pv=np.concatenate(Pv,0).squeeze(1)
            rho,_=spearmanr(Yv, Pv); rho = 0.0 if (rho is None or np.isnan(rho)) else float(rho)
            if rho > best_rho: best_rho = rho; best_state = copy.deepcopy(model.state_dict())
        # test with best
        model.load_state_dict(best_state); model.eval(); Ys=[]; Ps=[]
        with torch.no_grad():
            for b in dl_te:
                ids=b["input_ids"].to(DEVICE); att=b["attention_mask"].to(DEVICE)
                pf = b.get("predfa", None); pf = pf.to(DEVICE) if pf is not None else None
                pr = model(ids, att, pf); Ys.append(b["targets"][:,t_i].unsqueeze(1).numpy()); Ps.append(pr.cpu().numpy())
        preds_te[:,t_i] = np.concatenate(Ps,0).squeeze(1)
        # val dump
        model.eval(); Ps=[]
        with torch.no_grad():
            for b in dl_va:
                pf = b.get("predfa", None); pf = pf.to(DEVICE) if pf is not None else None
                pr = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), pf); Ps.append(pr.cpu().numpy())
        preds_va[:,t_i] = np.concatenate(Ps,0).squeeze(1)
    save_metrics(outdir, {"variant":"single_task", "fusion":"concat_128_relu_1"}, ys_te, preds_te, ys_va, preds_va, TARGETS)
    return {"Y":ys_te, "P":preds_te}

# ============================================================
# Family 2: PredFA-only (MLP)
# ============================================================
class PredFAMLP(nn.Module):
    def __init__(self, in_dim, out_dim=3, hidden=MLP_HIDDEN, dropout=MLP_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

def run_predfa_mlp_family(Xtr, Xva, Xte, Y_tr, Y_va, Y_te, tag="predfa_mlp", multi_task=True):
    sub = f"{tag}_{'mt' if multi_task else 'st'}"
    outdir = os.path.join(OUT_DIR, sub); ensure_dir(outdir)

    tr_ds = DocPredFADataset(Xtr, Y_tr); va_ds = DocPredFADataset(Xva, Y_va); te_ds = DocPredFADataset(Xte, Y_te)
    dl_tr = DataLoader(tr_ds, batch_size=MLP_BS, shuffle=True)
    dl_va = DataLoader(va_ds, batch_size=MLP_BS, shuffle=False)
    dl_te = DataLoader(te_ds, batch_size=MLP_BS, shuffle=False)

    if multi_task:
        model = PredFAMLP(in_dim=Xtr.shape[1], out_dim=3).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()
        best_rho=-1e9; best_state=None
        for _ in range(1, MLP_EPOCHS+1):
            model.train()
            for b in dl_tr:
                opt.zero_grad()
                pr = model(b["x"].to(DEVICE)); loss = loss_fn(pr, b["targets"].to(DEVICE))
                loss.backward(); opt.step()
            # dev
            model.eval(); Yv=[]; Pv=[]
            with torch.no_grad():
                for b in dl_va:
                    pr = model(b["x"].to(DEVICE)); Yv.append(b["targets"].numpy()); Pv.append(pr.cpu().numpy())
            Yv=np.concatenate(Yv,0); Pv=np.concatenate(Pv,0)
            rho = mae_spearman(Yv, Pv, TARGETS)["rho_macro"]
            if rho>best_rho: best_rho=rho; best_state=copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        # test
        model.eval(); Ys=[]; Ps=[]
        with torch.no_grad():
            for b in dl_te:
                pr = model(b["x"].to(DEVICE)); Ys.append(b["targets"].numpy()); Ps.append(pr.cpu().numpy())
        Yt=np.concatenate(Ys,0); Pt=np.concatenate(Ps,0)
        # val dump
        model.eval(); Ys=[]; Ps=[]
        with torch.no_grad():
            for b in dl_va:
                pr = model(b["x"].to(DEVICE)); Ys.append(b["targets"].numpy()); Ps.append(pr.cpu().numpy())
        Yv=np.concatenate(Ys,0); Pv=np.concatenate(Ps,0)
        save_metrics(outdir, {"variant":"multi_task"}, Yt, Pt, Yv, Pv, TARGETS)
        return {"Y":Yt, "P":Pt}

    # single-task: 3 runs
    preds_te = np.zeros_like(Y_te); preds_va = np.zeros_like(Y_va)
    for t_i in range(3):
        model = PredFAMLP(in_dim=Xtr.shape[1], out_dim=1).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()
        best_rho=-1e9; best_state=None
        for _ in range(1, MLP_EPOCHS+1):
            model.train()
            for b in dl_tr:
                opt.zero_grad()
                y = b["targets"][:,t_i].unsqueeze(1).to(DEVICE)
                pr = model(b["x"].to(DEVICE)); loss = loss_fn(pr, y)
                loss.backward(); opt.step()
            # dev
            model.eval(); Yv=[]; Pv=[]
            with torch.no_grad():
                for b in dl_va:
                    y = b["targets"][:,t_i].numpy()
                    pr = model(b["x"].to(DEVICE)).cpu().numpy()
                    Yv.append(y); Pv.append(pr.squeeze(1))
            Yv=np.concatenate(Yv,0); Pv=np.concatenate(Pv,0)
            rho,_=spearmanr(Yv, Pv); rho = 0.0 if (rho is None or np.isnan(rho)) else float(rho)
            if rho>best_rho: best_rho=rho; best_state=copy.deepcopy(model.state_dict())
        # test
        model.load_state_dict(best_state); model.eval(); Ps=[]
        with torch.no_grad():
            for b in dl_te:
                pr = model(b["x"].to(DEVICE)).cpu().numpy(); Ps.append(pr.squeeze(1))
        preds_te[:,t_i] = np.concatenate(Ps,0)
        # val
        model.eval(); Ps=[]
        with torch.no_grad():
            for b in dl_va:
                pr = model(b["x"].to(DEVICE)).cpu().numpy(); Ps.append(pr.squeeze(1))
        preds_va[:,t_i] = np.concatenate(Ps,0)
    save_metrics(outdir, {"variant":"single_task"}, Y_te, preds_te, Y_va, preds_va, TARGETS)
    return {"Y":Y_te, "P":preds_te}

# ============================================================
# Main
# ============================================================
def main():
    set_seed(RANDOM_STATE)

    # ----- Load data
    df_doc = sanitize_dataframe(pd.read_csv(DOC_CSV))
    df_sent = sanitize_dataframe(pd.read_csv(SENT_CSV))

    # If SEED missing on doc file → project from sentence file (mode)
    if SEED_COL not in df_doc.columns:
        seed_map = (df_sent[[DOC_ID_COL, SEED_COL]].dropna()
                    .drop_duplicates(subset=[DOC_ID_COL, SEED_COL])
                    .groupby(DOC_ID_COL)[SEED_COL]
                    .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]).reset_index())
        df_doc = sanitize_dataframe(df_doc.merge(seed_map, on=DOC_ID_COL, how="left"))

    # Drop docs without all targets
    df_doc = df_doc.dropna(subset=TARGETS).reset_index(drop=True)

    # ----- Split 70/15/15 by seed
    groups = df_doc[SEED_COL].to_numpy().astype(int)
    tr_idx, va_idx, te_idx = grouped_split_70_15_15_index(groups, random_state=RANDOM_STATE)
    seeds_tr, seeds_va, seeds_te = set(groups[tr_idx]), set(groups[va_idx]), set(groups[te_idx])
    assert not (seeds_tr & seeds_va) and not (seeds_tr & seeds_te) and not (seeds_va & seeds_te)

    # ----- Slice sentence rows per split
    need_cols = [DOC_ID_COL, SENT_IDX_COL, SENT_TEXT_COL, SEED_COL, FA_COL]
    assert all(c in df_sent.columns for c in need_cols), f"Sentence CSV missing {need_cols}"
    df_sent[DOC_ID_COL] = df_sent[DOC_ID_COL].astype(str)

    ids_tr = set(df_doc.loc[tr_idx, DOC_ID_COL].astype(str))
    ids_va = set(df_doc.loc[va_idx, DOC_ID_COL].astype(str))
    ids_te = set(df_doc.loc[te_idx, DOC_ID_COL].astype(str))

    df_tr_sent = df_sent[df_sent[DOC_ID_COL].isin(ids_tr)].reset_index(drop=True)
    df_va_sent = df_sent[df_sent[DOC_ID_COL].isin(ids_va)].reset_index(drop=True)
    df_te_sent = df_sent[df_sent[DOC_ID_COL].isin(ids_te)].reset_index(drop=True)

    # ----- Predict FA (BERT+ASL)
    print("[FA] Predicting sentence-level FA probabilities (BERT+ASL)…")
    tr_pred = predict_fa_probs(df_tr_sent, ASL_MODEL_DIR, ASL_TOK_NAME, batch=64, max_len=128)
    va_pred = predict_fa_probs(df_va_sent, ASL_MODEL_DIR, ASL_TOK_NAME, batch=64, max_len=128)
    te_pred = predict_fa_probs(df_te_sent, ASL_MODEL_DIR, ASL_TOK_NAME, batch=64, max_len=128)

    tr_pred.to_csv(os.path.join(OUT_DIR,"predfa_train.csv"), index=False)
    va_pred.to_csv(os.path.join(OUT_DIR,"predfa_val.csv"), index=False)
    te_pred.to_csv(os.path.join(OUT_DIR,"predfa_test.csv"), index=False)

    # ----- Aggregate PredFA = HSPT-13 (locked order)
    print("[FA] Aggregating PredFA per doc (HSPT-13)…")
    F_tr = aggregate_predfa_hspt13(tr_pred, DOC_ID_COL, SENT_IDX_COL)
    F_va = aggregate_predfa_hspt13(va_pred, DOC_ID_COL, SENT_IDX_COL)
    F_te = aggregate_predfa_hspt13(te_pred, DOC_ID_COL, SENT_IDX_COL)

    Xtr_raw, feat_cols = align_predfa_to_split(F_tr, df_doc, tr_idx)
    Xva_raw, _         = align_predfa_to_split(F_va, df_doc, va_idx)
    Xte_raw, _         = align_predfa_to_split(F_te, df_doc, te_idx)

    # ----- Z-score the 13 features using TRAIN stats + persist μ/σ and order
    Xtr, Xva, Xte, mu, sd = zscore_fit_transform(Xtr_raw, Xva_raw, Xte_raw)

    params_csv = os.path.join(OUT_DIR, "hspt13_zscore_params.csv")
    pd.DataFrame({"feature": SUM13_ORDER, "mu": mu.tolist(), "sd": sd.tolist()}).to_csv(params_csv, index=False)
    print(f"[PredFA] Saved z-score params → {params_csv}")

    order_json = os.path.join(OUT_DIR, "hspt13_feature_order.json")
    with open(order_json, "w") as f:
        json.dump(SUM13_ORDER, f, indent=2)
    print(f"[PredFA] Saved feature order → {order_json}")

    # ----- Targets arrays
    Y = df_doc[TARGETS].to_numpy(np.float32)
    Y_tr, Y_va, Y_te = Y[tr_idx], Y[va_idx], Y[te_idx]

    results = {}
    # ====================================================
    # Family 1: BERT (text-only)
    # ====================================================
    if RUN_BERT_TEXT:
        print("\n[DOC] BERT (text-only)…")
        if DO_MULTI_TASK:
            results["BERT_text_mt"] = run_bert_family(df_doc, tr_idx, va_idx, te_idx,
                                                      predfa_feats=None, tag="bert_text", multi_task=True)
        if DO_SINGLE_TASK:
            results["BERT_text_st"] = run_bert_family(df_doc, tr_idx, va_idx, te_idx,
                                                      predfa_feats=None, tag="bert_text", multi_task=False)

    # ====================================================
    # Family 2: PredFA-only (MLP) — uses 13-dim HSPT-13
    # ====================================================
    if RUN_PredFA_MLP:
        print("\n[DOC] PredFA-only (MLP, HSPT-13)…")
        if DO_MULTI_TASK:
            results["PredFA_mlp_mt"] = run_predfa_mlp_family(Xtr, Xva, Xte, Y_tr, Y_va, Y_te,
                                                             tag="predfa_mlp", multi_task=True)
        if DO_SINGLE_TASK:
            results["PredFA_mlp_st"] = run_predfa_mlp_family(Xtr, Xva, Xte, Y_tr, Y_va, Y_te,
                                                             tag="predfa_mlp", multi_task=False)

    # ====================================================
    # Family 3: BERT + PredFA (fusion via concat)
    # ====================================================
    if RUN_BERT_PLUS_PredFA:
        print("\n[DOC] BERT + PredFA (concat fusion, HSPT-13)…")
        predfa_feats = (Xtr, Xva, Xte)
        if DO_MULTI_TASK:
            results["BERT_plus_PredFA_mt"] = run_bert_family(df_doc, tr_idx, va_idx, te_idx,
                                                             predfa_feats=predfa_feats, tag="bert_text_predfa", multi_task=True)
        if DO_SINGLE_TASK:
            results["BERT_plus_PredFA_st"] = run_bert_family(df_doc, tr_idx, va_idx, te_idx,
                                                             predfa_feats=predfa_feats, tag="bert_text_predfa", multi_task=False)

    # ----------------------------------------------------
    # Consolidated comparison table + pretty print
    # ----------------------------------------------------
    compare = {}
    def add_row(key, folder):
        met_path = os.path.join(OUT_DIR, folder, "metrics.json")
        if not os.path.isfile(met_path): return False
        with open(met_path,"r") as f: m = json.load(f)
        test = m["test"]; per = test["per_target"]
        compare[f"{key}_rho_macro"] = test["rho_macro"]
        compare[f"{key}_Directness_MAE"] = per[TARGETS[0]]["MAE"]
        compare[f"{key}_Markers_MAE"]    = per[TARGETS[1]]["MAE"]
        compare[f"{key}_Overall_MAE"]    = per[TARGETS[2]]["MAE"]
        compare[f"{key}_rho_D"] = per[TARGETS[0]]["Spearman"]
        compare[f"{key}_rho_M"] = per[TARGETS[1]]["Spearman"]
        compare[f"{key}_rho_O"] = per[TARGETS[2]]["Spearman"]
        return True

    have = []
    if add_row("BERT_text_mt", "bert_text_mt"): have.append("BERT_text_mt")
    if add_row("BERT_text_st", "bert_text_st"): have.append("BERT_text_st")
    if add_row("PredFA_only_mt", "predfa_mlp_mt"): have.append("PredFA_only_mt")
    if add_row("PredFA_only_st", "predfa_mlp_st"): have.append("PredFA_only_st")
    if add_row("BERT_plus_PredFA_mt", "bert_text_predfa_mt"): have.append("BERT_plus_PredFA_mt")
    if add_row("BERT_plus_PredFA_st", "bert_text_predfa_st"): have.append("BERT_plus_PredFA_st")

    with open(os.path.join(OUT_DIR, "compare_table.json"), "w") as f:
        json.dump(compare, f, indent=2)

    print("\n=== TEST Comparison (lower MAE better, higher ρ better) ===")
    def pr_line(key, label):
        D = compare.get(f"{key}_Directness_MAE", None)
        M = compare.get(f"{key}_Markers_MAE", None)
        O = compare.get(f"{key}_Overall_MAE", None)
        rD = compare.get(f"{key}_rho_D", None)
        rM = compare.get(f"{key}_rho_M", None)
        rO = compare.get(f"{key}_rho_O", None)
        if None in (D,M,O,rD,rM,rO): return
        print(f"{label:<22} | MAE ↓  D:{D:.4f} M:{M:.4f} O:{O:.4f} | ρ_D:{rD:.4f} ρ_M:{rM:.4f} ρ_O:{rO:.4f}")

    pr_line("BERT_plus_PredFA_mt", "BERT_plus_PredFA (mt)")
    pr_line("BERT_plus_PredFA_st", "BERT_plus_PredFA (st)")
    pr_line("BERT_text_mt",        "BERT_text (mt)")
    pr_line("BERT_text_st",        "BERT_text (st)")
    pr_line("PredFA_only_mt",      "PredFA_only (mt)")
    pr_line("PredFA_only_st",      "PredFA_only (st)")

    print("\n[DONE] Saved:", os.path.join(OUT_DIR, "compare_table.json"))

if __name__ == "__main__":
    main()

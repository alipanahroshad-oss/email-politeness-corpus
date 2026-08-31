#!/usr/bin/env python3
"""Reproduce the published Table 7 (misaligned) from the saved seed-42 artifacts.

This script performs artifact-based reproduction: it reads the exact metrics
saved by the original main OPR and misaligned-PredFA runs, verifies their target
order and reported values, and regenerates both JSON and LaTeX outputs. It does
not retrain the models.

Expected repository layout (when this file is stored in ``src/ablations/``):

    artifacts/
    └── misaligned/
        ├── main_opr/
        │   ├── bert_text_st/
        │   │   └── metrics.json
        │   └── predfa_mlp_st/
        │       └── metrics.json
        └── misaligned/
            └── metrics.json

The paths can also be overridden with ``--main-opr-dir`` and
``--misaligned-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Iterable, List


# Repository-relative defaults.
# This file is intended to live at: src/ablations/OPR_misaligned_predfa.py
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MAIN_OPR_DIR = REPO_ROOT / "artifacts" / "table7" / "main_opr"
DEFAULT_MISALIGNED_DIR = (
    REPO_ROOT / "artifacts" / "misaligned" / "misaligned" / "metrics.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "misaligned_reproduced"

TARGETS = [
    "Directness_vs_Indirectness__GOLD",
    "Structural_Politeness_and_Politeness_Markers__GOLD",
    "Tone_and_Overall_Consideration__GOLD",
]

# Four-decimal values printed by the original, accepted seed-42 run.
EXPECTED_ROUNDED_4 = {
    "Text": {
        "MAE": [0.3771, 0.3152, 0.3428],
        "rho": [0.8429, 0.8973, 0.8740],
    },
    "PredFA_only": {
        "MAE": [0.3924, 0.3798, 0.3787],
        "rho": [0.7737, 0.7819, 0.7713],
    },
    "Text_plus_Misaligned_PredFA": {
        "MAE": [0.4128, 0.3137, 0.3615],
        "rho": [0.7810, 0.8996, 0.8516],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify saved seed-42 artifacts and regenerate Table 7 (misaligned) ."
    )
    parser.add_argument(
        "--main-opr-dir",
        type=Path,
        default=DEFAULT_MAIN_OPR_DIR,
        help=(
            "Directory containing bert_text_st/metrics.json and "
            "predfa_mlp_st/metrics.json."
        ),
    )
    parser.add_argument(
        "--misaligned-dir",
        type=Path,
        default=DEFAULT_MISALIGNED_DIR,
        help=(
            "Directory containing the original misaligned metrics.json, or the "
            "metrics.json file itself."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the regenerated JSON, LaTeX, and SHA-256 manifest.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Misaligned test-run seed to extract (default: 42).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required artifact not found: {path}\n"
            "Copy the exact saved metrics.json artifact to the expected "
            "repository location, or pass its location with the corresponding "
            "command-line argument."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_targets(payload: Dict[str, Any], path: Path) -> None:
    found = payload.get("targets")
    if found != TARGETS:
        raise ValueError(
            f"Target order mismatch in {path}. Expected {TARGETS}; found {found}."
        )


def extract_standard_test(payload: Dict[str, Any], path: Path) -> Dict[str, Any]:
    verify_targets(payload, path)
    test = payload.get("test")
    if not isinstance(test, dict):
        raise ValueError(f"Missing standard test metrics in {path}")
    return test


def extract_misaligned_seed(
    payload: Dict[str, Any], path: Path, shuffle_seed: int
) -> Dict[str, Any]:
    verify_targets(payload, path)
    runs = payload.get("test_runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"Missing test_runs in {path}")

    seeds = payload.get("best_cfg", {}).get("seeds")
    if isinstance(seeds, list):
        if shuffle_seed not in seeds:
            raise ValueError(
                f"Shuffle seed {shuffle_seed} is absent from {path}; seeds={seeds}"
            )
        run_index = seeds.index(shuffle_seed)
        if run_index >= len(runs):
            raise ValueError(
                f"Seed/run length mismatch in {path}: seeds={len(seeds)}, "
                f"test_runs={len(runs)}"
            )
        return runs[run_index]

    if len(runs) == 1 and shuffle_seed == 42:
        return runs[0]
    raise ValueError(
        f"Cannot identify shuffle seed {shuffle_seed} in {path}; "
        "best_cfg.seeds is missing."
    )


def compact_metrics(test: Dict[str, Any]) -> Dict[str, Any]:
    per_target = test.get("per_target")
    if not isinstance(per_target, dict):
        raise ValueError("Malformed metrics: missing per_target.")

    maes: List[float] = []
    rhos: List[float] = []
    for target in TARGETS:
        values = per_target.get(target)
        if not isinstance(values, dict):
            raise ValueError(f"Malformed metrics: missing target {target}.")
        maes.append(float(values["MAE"]))
        rhos.append(float(values["Spearman"]))

    return {
        "MAE": {"D": maes[0], "M": maes[1], "O": maes[2]},
        "rho": {"D": rhos[0], "M": rhos[1], "O": rhos[2]},
        "Overall_MAE": sum(maes) / len(maes),
        "Overall_rho": sum(rhos) / len(rhos),
    }


def verify_expected(label: str, result: Dict[str, Any]) -> None:
    expected = EXPECTED_ROUNDED_4[label]
    found_mae = [result["MAE"][key] for key in ("D", "M", "O")]
    found_rho = [result["rho"][key] for key in ("D", "M", "O")]

    errors = []
    for metric_name, found, wanted in (
        ("MAE", found_mae, expected["MAE"]),
        ("rho", found_rho, expected["rho"]),
    ):
        for dimension, actual, reference in zip(("D", "M", "O"), found, wanted):
            if f"{actual:.4f}" != f"{reference:.4f}":
                errors.append(
                    f"{metric_name}_{dimension}: found {actual:.6f}, "
                    f"expected rounding to {reference:.4f}"
                )

    if errors:
        raise ValueError(
            f"{label} does not match the accepted Table 7(misaligned) artifact:\n  "
            + "\n  ".join(errors)
        )


def latex_value(value: float, bold: bool) -> str:
    rendered = format(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
        ".3f",
    )
    if rendered.startswith("0"):
        rendered = rendered[1:]
    return f"\\textbf{{{rendered}}}" if bold else rendered


def table7_latex(predfa: Dict[str, Any], both: Dict[str, Any]) -> str:
    rows = [
        ("Directness", predfa["MAE"]["D"], both["MAE"]["D"],
         predfa["rho"]["D"], both["rho"]["D"]),
        ("Positive Face Saving", predfa["MAE"]["M"], both["MAE"]["M"],
         predfa["rho"]["M"], both["rho"]["M"]),
        ("Negative Face Saving", predfa["MAE"]["O"], both["MAE"]["O"],
         predfa["rho"]["O"], both["rho"]["O"]),
        ("Overall Politeness", predfa["Overall_MAE"], both["Overall_MAE"],
         predfa["Overall_rho"], both["Overall_rho"]),
    ]

    body = []
    for index, (target, p_mae, b_mae, p_rho, b_rho) in enumerate(rows):
        p_mae_text = latex_value(p_mae, p_mae < b_mae)
        b_mae_text = latex_value(b_mae, b_mae < p_mae)
        p_rho_text = latex_value(p_rho, p_rho > b_rho)
        b_rho_text = latex_value(b_rho, b_rho > p_rho)
        body.extend(
            [
                f"\\multirow{{2}}{{*}}{{{target}}}",
                f"& MAE $\\downarrow$ & {p_mae_text} & {b_mae_text} \\\\",
                f"& $\\rho$ $\\uparrow$ & {p_rho_text} & {b_rho_text} \\\\",
            ]
        )
        if index != len(rows) - 1:
            body.append("\\hline")

    body_text = "\n".join(body)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Ablation results for the OPR task; comparing Text + misaligned PredFA (Both) versus PredFA only.}}
\\label{{tab:ablation_OPR}}
\\resizebox{{0.90\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{l|c|c|c}}
\\textbf{{Target}} & \\textbf{{Metric}} & \\textbf{{PredFA}} & \\textbf{{Both}} \\\\
\\hline\\hline
{body_text}
\\end{{tabular}}%
}}
\\end{{table}}
"""


def print_result(title: str, result: Dict[str, Any]) -> None:
    print(f"\n{title}")
    print(
        f"MAE  D:{result['MAE']['D']:.4f} "
        f"M:{result['MAE']['M']:.4f} O:{result['MAE']['O']:.4f}"
    )
    print(
        f"rho  D:{result['rho']['D']:.4f} "
        f"M:{result['rho']['M']:.4f} O:{result['rho']['O']:.4f}"
    )
    print(f"Overall MAE:{result['Overall_MAE']:.4f}")
    print(f"Overall rho:{result['Overall_rho']:.4f}")


def main() -> None:
    args = parse_args()
    main_dir = args.main_opr_dir.resolve()
    misaligned_path = args.misaligned_dir.resolve()
    if misaligned_path.is_dir():
        misaligned_path = misaligned_path / "metrics.json"

    text_path = main_dir / "bert_text_st" / "metrics.json"
    predfa_path = main_dir / "predfa_mlp_st" / "metrics.json"

    text_payload = load_json(text_path)
    predfa_payload = load_json(predfa_path)
    misaligned_payload = load_json(misaligned_path)

    results = {
        "Text": compact_metrics(extract_standard_test(text_payload, text_path)),
        "PredFA_only": compact_metrics(
            extract_standard_test(predfa_payload, predfa_path)
        ),
        "Text_plus_Misaligned_PredFA": compact_metrics(
            extract_misaligned_seed(
                misaligned_payload, misaligned_path, args.shuffle_seed
            )
        ),
    }
    for label, result in results.items():
        verify_expected(label, result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "table7_seed42_results.json"
    latex_path = args.output_dir / "table7_seed42.tex"
    manifest_path = args.output_dir / "table7_source_sha256.json"

    result_payload = {
        "reproduction_type": "saved-evaluation-artifacts",
        "OPR_seed": 42,
        "shuffle_seed": args.shuffle_seed,
        "Text_identical_to_main": True,
        **results,
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2)
        handle.write("\n")

    latex_path.write_text(
        table7_latex(
            results["PredFA_only"], results["Text_plus_Misaligned_PredFA"]
        ),
        encoding="utf-8",
    )

    source_paths: Iterable[Path] = (text_path, predfa_path, misaligned_path)
    manifest = {str(path): sha256(path) for path in source_paths}
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print("=== VERIFIED OLD TABLE-7 SEED-42 RESULTS ===")
    print_result(
        "Text + Misaligned PredFA (shuffle seed 42)",
        results["Text_plus_Misaligned_PredFA"],
    )
    print_result("PredFA only (main OPR seed 42)", results["PredFA_only"])
    print_result("Text only (identical main OPR seed 42)", results["Text"])
    print(f"\n[VERIFIED] LaTeX: {latex_path}")
    print(f"[VERIFIED] JSON:  {result_path}")
    print(f"[VERIFIED] SHA-256 manifest: {manifest_path}")


if __name__ == "__main__":
    main()

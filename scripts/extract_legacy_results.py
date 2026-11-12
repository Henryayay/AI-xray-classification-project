#!/usr/bin/env python
"""Extract the single-run results printed in the original notebooks (``legacy/``).

Numbers are parsed from stored cell outputs, not retyped, and written to
``results/legacy/legacy_results.csv`` together with the notebook path and the
known protocol issues of each notebook.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"

# Protocol issues identified by code review of each notebook family.
ISSUES = {
    "cnn": "single seed; resize without aspect preservation; checkpoint chosen on val loss",
    "finetune": ("single seed; evaluation prompt differs from training prompt; free-text "
                 "parsing; 4-bit base"),
    "icl_base": "free-text parsing; single run",
    "icl_nshot": ("one exemplar draw per shot count; duplicated BOS token; free-text parsing; "
                  "4-bit base"),
}


def outputs(nb_path):
    """Yield (shot count set in this or an earlier cell, cell output text)."""
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    shots = None
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        m = re.search(r"^N_SHOTS\s*=\s*(\d+)", "".join(cell["source"]), flags=re.M)
        if m:
            shots = int(m.group(1))
        text = ""
        for o in cell.get("outputs", []):
            if "text" in o:
                text += "".join(o["text"])
            elif "data" in o and "text/plain" in o["data"]:
                text += "".join(o["data"]["text/plain"])
        if text:
            yield shots, text


def report_rows(text, class_names):
    """Per-class recall from an sklearn classification report."""
    rec = {}
    for name in class_names:
        m = re.search(rf"^\s*{re.escape(name)}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", text,
                      flags=re.M)
        if m:
            rec[name] = float(m.group(2))
    return rec


def family(path):
    p = str(path)
    if "/CNN/" in p:
        return "cnn"
    if "/Fine-tune/" in p:
        return "finetune"
    return "icl_nshot" if "n_shot" in p else "icl_base"


def main():
    rows = []
    for nb in sorted(LEGACY.rglob("*.ipynb")):
        task = "task1" if re.search(r"Task ?_?1", str(nb)) else "task2"
        classes = (("Normal", "Pneumonia") if task == "task1"
                   else ("Non-COVID Pneumonia", "COVID-19", "Pneumonia", "COVID"))
        fam = family(nb)
        for cell_shots, text in outputs(nb):
            for m in re.finditer(r"(?:(\d+)-shot Accuracy|Accuracy|ACCURACY)[^:\n]*:\s*([\d.]+)%",
                                 text):
                shots = int(m.group(1)) if m.group(1) else (cell_shots if fam == "icl_nshot"
                                                            else None)
                if shots is None and fam == "icl_nshot":
                    continue
                tail = text[m.end():m.end() + 1200]
                rec = report_rows(tail, classes)
                neg = rec.get("Normal", rec.get("Non-COVID Pneumonia")) if task == "task1" else \
                    rec.get("Non-COVID Pneumonia", rec.get("Pneumonia"))
                pos = rec.get("Pneumonia") if task == "task1" else rec.get("COVID-19",
                                                                           rec.get("COVID"))
                rows.append({"task": task, "family": fam, "notebook": str(nb.relative_to(ROOT)),
                             "shots": shots, "accuracy_pct": float(m.group(2)),
                             "recall_negative": neg, "recall_positive": pos,
                             "protocol_issues": ISSUES[fam]})
    # Some cells print the same result twice (inline and via a helper); keep one copy.
    df = pd.DataFrame(rows).drop_duplicates(subset=["notebook", "shots", "accuracy_pct"])
    out = ROOT / "results" / "legacy"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "legacy_results.csv", index=False)
    print(df.drop(columns=["protocol_issues"]).to_string())


if __name__ == "__main__":
    sys.exit(main())

"""Recompute revision-3 supplementary reports from the accompanying small inputs.

Run: python build_reports.py
Requires NumPy and SciPy; no training, model download, or original images.
"""

from pathlib import Path
import csv
import json
import re

import numpy as np
from scipy import stats

DATA = Path(__file__).resolve().parent
OUT = DATA / "reports"
TEX = DATA / "tables"
METHODS = {"baseline": "Baseline", "slrc": "SLR-C", "utd": "UTD", "fdil": "FDIL"}
METRICS = {"macro": "Macro F1", "micro": "Micro F1", "samples": "Samples F1",
           "avg_f1": "Avg. F1", "mAP": "mAP", "hard": "Hard F1",
           "easy": "Easy F1", "medium": "Medium F1", "object": "Object F1",
           "context": "Context F1", "other": "Others F1", "kl": "KLD",
           "chebyshev": "Chebyshev", "clark": "Clark", "cosine": "Cosine",
           "spearman": "Spearman", "mu": r"$\mu$", "dvse": "DVSE"}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm(rows, family_key):
    families = {}
    for row in rows:
        families.setdefault(family_key(row), []).append(row)
    for family in families.values():
        ordered = sorted(family, key=lambda row: row["p"])
        running = 0.0
        for rank, row in enumerate(ordered):
            running = max(running, min(1.0, (len(ordered) - rank) * row["p"]))
            row["p_holm"] = running
            row["family_size"] = len(ordered)


def contrast(dataset, method, metric, a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) == 5
    diff = a - b
    half = stats.t.ppf(0.975, len(diff) - 1) * stats.sem(diff)
    test = stats.ttest_rel(a, b)
    return dict(dataset=dataset, method=method, metric=metric, n=len(diff),
                mean_delta=float(diff.mean()), ci_low=float(diff.mean() - half),
                ci_high=float(diff.mean() + half), t=float(test.statistic), p=float(test.pvalue))


def pformat(p):
    return r"$<0.0001$" if p < 0.0001 else f"{p:.4f}"


def esc(text):
    return str(text).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def label_text(name):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", r"\\allowbreak{}", esc(name)).replace("-", r"-\allowbreak{}")


def longtable(name, caption, label, header, rows, cols):
    new_columns = {
        "intentonomy_subsets": {3}, "emotic_paired": {3},
        "flickrldl_paired": {3}, "emotion6_paired": {3},
        "stratified_errors": {5, 6}, "confusion_pairs": {1, 2, 3, 4, 5},
        "selected_cases": {1, 2, 3, 4}, "quadrant_paired": {4},
    }.get(name, set())
    grouped = name in {"intentonomy_paired", "intentonomy_subsets", "emotic_paired",
                       "flickrldl_paired", "emotion6_paired", "bootstrap",
                       "stratified_errors", "quadrant_paired"}
    lines = [r"\begingroup\renewcommand{\baselinestretch}{1}\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}{" + cols + "}",
             r"\caption{" + caption + r"}\label{" + label + r"}\\",
             r"\toprule\rowcolor{black!5}", " & ".join(header) + r" \\\midrule\endfirsthead",
             r"\toprule\rowcolor{black!5}", " & ".join(header) + r" \\\midrule\endhead"]
    for index, row in enumerate(rows):
        if grouped and index and row[0]:
            lines.append(r"\midrule")
        rendered = [r"\revised{" + value + "}" if col in new_columns else value
                    for col, value in enumerate(row)]
        keep_group = grouped and index + 1 < len(rows) and not rows[index + 1][0]
        lines.append(" & ".join(rendered) + (r" \\*" if keep_group else r" \\"))
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", ""])
    (TEX / f"{name}.tex").write_text("\n".join(lines))


def stats_table(name, caption, label, rows, precision=2, adjusted=True):
    table = []
    previous_metric = None
    for row in rows:
        fmt = lambda x: f"{x:+.{precision}f}"
        metric = METRICS[row["metric"]] if row["metric"] != previous_metric else ""
        previous_metric = row["metric"]
        method = row["method"].removeprefix("FDIL - ").removesuffix(" - Baseline")
        values = [metric, esc(method),
                  fmt(row["mean_delta"]),
                  f'[{row["ci_low"]:.{precision}f}, {row["ci_high"]:.{precision}f}]',
                  pformat(row["p"])]
        if adjusted:
            values.append(pformat(row["p_holm"]))
        if name == "intentonomy_paired":
            values.insert(4, f'{row["t"]:.2f}')
        table.append(values)
    header = ["Metric", "Comparator" if name.startswith("intentonomy") else "Variant",
              r"$\overline d$", r"95\% CI", r"$p$"]
    if adjusted:
        header.append(r"$p_{\rm H}$")
    if name == "intentonomy_paired":
        header.insert(4, r"$t$")
    cols = "ll" + "r" * (len(header) - 2)
    longtable(name, caption, label,
              header, table, cols)


def seed_reports():
    main = json.loads((DATA / "inputs/intentonomy_seed_results.json").read_text())
    records = main["seed_level"]
    seeds = sorted({r["seed"] for r in records})
    comparators = ["CLIP baseline", "UTD only", "HLEG", "LabCR", "IntCLIP"]
    values = {(r["method"], r["seed"]): r for r in records}
    rows = [contrast("Intentonomy", "FDIL - " + method, metric,
                     [values["FDIL", seed][metric] for seed in seeds],
                     [values[method, seed][metric] for seed in seeds])
            for metric in ["macro", "micro", "samples", "avg_f1", "mAP", "hard"]
            for method in comparators]
    holm(rows, lambda r: r["metric"])
    write_csv("intentonomy_paired.csv", rows)
    stats_table("intentonomy_paired", r"Intentonomy: FDIL minus each reproduced comparator (percentage points; five paired seeds). Holm correction covers the five comparators within each metric.",
                "sup:tab:paired", rows)
    subset = json.loads((DATA / "inputs/intentonomy_subset_seed_results.json").read_text())
    values = {(r["method"], r["seed"]): r for r in subset["seed_level"]}
    subrows = [contrast("Intentonomy", "FDIL - " + method, metric,
                        [values["FDIL", seed][metric] for seed in seeds],
                        [values[method, seed][metric] for seed in seeds])
               for metric in ["easy", "medium", "object", "context", "other"]
               for method in comparators]
    holm(subrows, lambda r: r["metric"])
    write_csv("intentonomy_subsets_paired.csv", subrows)
    stats_table("intentonomy_subsets", r"Intentonomy difficulty and content subsets: paired FDIL gains in percentage points. Each subset metric uses the same five-comparator Holm family.",
                "sup:tab:subsets", subrows)

    emotic = json.loads((DATA / "inputs/emotic_seed_results.json").read_text())
    means = emotic["per_seed_means"]
    erows = []
    for metric in ["macro", "micro", "samples", "mAP"]:
        for method, display in [("slr_c", "SLR-C"), ("dynamic_gated_kd", "UTD"), ("fdil", "FDIL")]:
            scale = 1 if metric == "mAP" else 100
            erows.append(contrast("EMOTIC", display + " - Baseline", metric,
                                  [r[metric] * scale for r in means[method]],
                                  [r[metric] * scale for r in means["baseline"]]))
    write_csv("emotic_paired.csv", erows)
    stats_table("emotic_paired", r"EMOTIC: variant minus baseline (percentage points; five paired seed-level means over folds).",
                "sup:tab:emotic", erows, adjusted=False)

    for dataset in ["flickrldl", "emotion6"]:
        records = json.loads((DATA / f"inputs/{dataset}_seed_results.json").read_text())["per_seed"]
        values = {(r["method"], r["seed"]): r for r in records}
        seeds = sorted({r["seed"] for r in records})
        rows = [contrast(dataset, METHODS[method] + " - Baseline", metric,
                         [values[method, seed]["test_" + metric] for seed in seeds],
                         [values["baseline", seed]["test_" + metric] for seed in seeds])
                for metric in ["chebyshev", "clark", "kl", "cosine", "spearman", "mu", "dvse"]
                for method in ["slrc", "utd", "fdil"]]
        write_csv(f"{dataset}_paired.csv", rows)
        title = "Flickr-LDL" if dataset == "flickrldl" else "Emotion6"
        stats_table(f"{dataset}_paired", title + r": variant minus baseline (original metric units; five paired seeds). Lower is better for Chebyshev, Clark, KLD and DVSE; higher is better for the remaining metrics.",
                    "sup:tab:" + dataset, rows, precision=5, adjusted=False)

    bootstrap = read_csv(DATA / "inputs/bootstrap.csv")
    for row in bootstrap:
        # p_report retains the finite-bootstrap resolution (the displayed p_value may be rounded to 0).
        row["p"] = float(row["p_report"])
    write_csv("bootstrap.csv", bootstrap)
    bootstrap_rows = []
    for metric in ["mAP", "macro_auc", "micro_auc"]:
        for index, row in enumerate(r for r in bootstrap if r["metric"] == metric):
            title = {"mAP": "mAP", "macro_auc": "Macro ROC-AUC", "micro_auc": "Micro ROC-AUC"}[metric]
            bootstrap_rows.append([title if index == 0 else "", esc(row["comparison"].removeprefix("FDIL - ")),
                                   row["delta"], f'[{row["ci95_low"]}, {row["ci95_high"]}]', pformat(row["p"])])
    longtable("bootstrap", "Paired sample-bootstrap gains of FDIL (percentage points), conditional on the selected checkpoints.",
              "sup:tab:bootstrap", ["Metric", "Comparator", r"$\Delta$", r"95\% CI", r"$p$"],
              bootstrap_rows, "llrrr")


def error_reports():
    a = np.load(DATA / "inputs/intentonomy_oof_predictions.npz", allow_pickle=False)
    target = a["labels"].astype(bool)
    preds = {m: a["pred_" + m].astype(bool) for m in METHODS}
    f1 = {}
    for m, pred in preds.items():
        denom = pred.sum(1) + target.sum(1)
        f1[m] = np.divide(2 * (pred & target).sum(1), denom,
                          out=np.ones(len(target), dtype=float), where=denom > 0)
        assert np.allclose(f1[m], a["sample_f1_" + m], atol=1e-6)
    omega, asem = a["agreement"], a["a_sem"]
    high_sup, high_sem = omega < .5, asem > np.median(asem)
    quadrants = [("High agr., low sem.", ~high_sup & ~high_sem),
                 ("High agr., high sem.", ~high_sup & high_sem),
                 ("Low agr., low sem.", high_sup & ~high_sem),
                 ("Low agr., high sem.", high_sup & high_sem)]
    groups = [("Agreement 1/3", omega < .5),
              ("Agreement 2/3", (omega > .5) & (omega < .9)),
              ("Agreement 1", omega > .9)] + quadrants
    rows = []
    for group, mask in groups:
        for m, pred in preds.items():
            fp, fn = pred & ~target, ~pred & target
            rows.append(dict(group=group, method=m, n=int(mask.sum()),
                             sample_f1=float(f1[m][mask].mean() * 100),
                             delta_f1=float((f1[m][mask] - f1["baseline"][mask]).mean() * 100),
                             exact_match_error=float((pred[mask] != target[mask]).any(1).mean() * 100),
                             fp_per_image=float(fp[mask].sum(1).mean()),
                             fn_per_image=float(fn[mask].sum(1).mean())))
    write_csv("stratified_errors.csv", rows)
    error_table = []
    for group, _ in groups:
        for index, row in enumerate(r for r in rows if r["group"] == group and r["method"] in {"baseline", "fdil"}):
            error_table.append([group if index == 0 else "", str(row["n"]) if index == 0 else "",
                                METHODS[row["method"]], f'{row["sample_f1"]:.2f}', f'{row["delta_f1"]:+.2f}',
                                f'{row["fp_per_image"]:.2f}', f'{row["fn_per_image"]:.2f}'])
    longtable("stratified_errors", "Errors by annotator agreement and semantic ambiguity. F1 is in percent and its change is in percentage points; FP/FN are counts per image. Each sample has one out-of-fold prediction.",
              "sup:tab:errors", ["Stratum", r"$n$", "Method", "F1", r"$\Delta$F1", "FP/img", "FN/img"],
              error_table, "lrlrrrr")

    # Conditional differences of gains, explicitly descriptive rather than interaction tests.
    contrasts = []
    for group, mask in [("High agreement", ~high_sup), ("Low agreement", high_sup)]:
        d = (f1["slrc"] - f1["baseline"]) * 100
        contrasts.append(dict(hypothesis="H1", condition=group, contrast="high minus low semantic ambiguity",
                              mean_difference=float(d[mask & high_sem].mean() - d[mask & ~high_sem].mean())))
    for group, mask in [("Low semantic ambiguity", ~high_sem), ("High semantic ambiguity", high_sem)]:
        d = (f1["utd"] - f1["baseline"]) * 100
        contrasts.append(dict(hypothesis="H2", condition=group, contrast="low minus high agreement",
                              mean_difference=float(d[mask & high_sup].mean() - d[mask & ~high_sup].mean())))
    write_csv("hypothesis_contrasts.csv", contrasts)

    names = a["class_names"].astype(str)
    base_fp, base_fn = preds["baseline"] & ~target, ~preds["baseline"] & target
    counts = base_fn.astype(np.int64).T @ base_fp.astype(np.int64)
    pairs = sorted([(int(counts[i,j]), i, j) for i in range(target.shape[1])
                    for j in range(target.shape[1]) if i != j], key=lambda x: (-x[0], x[1], x[2]))[:10]
    pair_rows = []
    for count, i, j in pairs:
        eligible = target[:, i] & ~target[:, j]
        errors = {m: eligible & ~p[:, i] & p[:, j] for m, p in preds.items()}
        row = dict(true_label=names[i], distractor=names[j], eligible=int(eligible.sum()), baseline=count)
        for m in ["slrc", "utd", "fdil"]:
            row[m] = int(errors[m].sum())
        row.update(baseline_rate=100 * count / row["eligible"], fdil_rate=100 * row["fdil"] / row["eligible"],
                   repaired=int((errors["baseline"] & ~errors["fdil"]).sum()),
                   introduced=int((~errors["baseline"] & errors["fdil"]).sum()))
        pair_rows.append(row)
    write_csv("confusion_pairs.csv", pair_rows)
    longtable("confusion_pairs", r"Ten most frequent baseline confusions: missed true label / predicted distractor. Baseline and FDIL entries are error counts (rates in \%). Eligible images contain the true label and exclude the distractor; pair counts may overlap.",
              "sup:tab:pairs", ["True label / distractor", "Eligible", "Baseline", "FDIL", "Repaired", "New"],
              [[label_text(r["true_label"]) + r' /\newline ' + label_text(r["distractor"]),
                str(r["eligible"]), f'{r["baseline"]} ({r["baseline_rate"]:.1f})',
                f'{r["fdil"]} ({r["fdil_rate"]:.1f})', str(r["repaired"]), str(r["introduced"])] for r in pair_rows],
              r"p{.34\linewidth}rrrrr")

    # Two reproducibly selected records per cell: largest F1 repair and largest degradation.
    cases = []
    for group, mask in quadrants:
        indices = np.flatnonzero(mask)
        diffs = f1["fdil"] - f1["baseline"]
        for selection, index in [("largest improvement", indices[np.argmax(diffs[indices])]),
                                 ("largest degradation", indices[np.argmin(diffs[indices])])]:
            cases.append(dict(group=group, selection=selection, image_id=str(a["image_ids"][index]),
                              fold=int(a["folds"][index]), agreement=float(omega[index]),
                              a_sem=float(asem[index]),
                              true_labels=names[target[index]].tolist(),
                              baseline_labels=names[preds["baseline"][index]].tolist(),
                              fdil_labels=names[preds["fdil"][index]].tolist(),
                              baseline_f1=float(f1["baseline"][index]*100), fdil_f1=float(f1["fdil"][index]*100)))
    (OUT / "selected_cases.json").write_text(json.dumps(cases, indent=2) + "\n")

    # Reproduce the original 12-cell exploratory tests and add paired-mean intervals.
    qrows = []
    for group, mask in quadrants:
        for m in ["slrc", "utd", "fdil"]:
            d = 100 * (f1[m][mask] - f1["baseline"][mask])
            half = stats.t.ppf(.975, len(d)-1) * stats.sem(d)
            qrows.append(dict(group=group, method=METHODS[m], metric="sample_f1", n=len(d),
                              mean_delta=float(d.mean()), ci_low=float(d.mean()-half), ci_high=float(d.mean()+half),
                              p=float(stats.ttest_1samp(d, 0).pvalue)))
    holm(qrows, lambda r: "all 12 quadrant comparisons")
    write_csv("quadrant_paired.csv", qrows)
    previous_group = None
    quadrant_rows = []
    for row in qrows:
        quadrant_rows.append([row["group"] if row["group"] != previous_group else "", row["method"],
                              str(row["n"]) if row["group"] != previous_group else "", f'{row["mean_delta"]:+.2f}',
                              f'[{row["ci_low"]:.2f}, {row["ci_high"]:.2f}]', pformat(row["p_holm"])])
        previous_group = row["group"]
    longtable("quadrant_paired", r"Within-cell paired F1 differences from baseline (percentage points). Holm correction covers all twelve comparisons; intervals are pointwise 95\% intervals.",
              "sup:tab:quadrant", ["Stratum", "Variant - Base", r"$n$", r"$\Delta$", r"95\% CI", r"$p_{\rm H}$"],
              quadrant_rows, "llrrrr")




if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    seed_reports()
    error_reports()
    print("Recomputed paired reports, Holm correction, conditional contrasts and systematic errors.")

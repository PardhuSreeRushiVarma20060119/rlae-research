#!/usr/bin/env python3

import json
import os
import argparse


# ----------------------------------------------------------
# LOAD JSON-LINES LOG
# ----------------------------------------------------------

def load_results(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Log file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    # Case 1: Proper JSON array
    if content.startswith("["):
        return json.loads(content)

    # Case 2: Concatenated JSON objects
    records = []
    decoder = json.JSONDecoder()
    idx = 0

    while idx < len(content):
        content = content.lstrip()
        try:
            obj, offset = decoder.raw_decode(content)
            records.append(obj)
            content = content[offset:]
        except json.JSONDecodeError:
            break

    return records


# ----------------------------------------------------------
# KEEP LATEST ENTRY PER SCALE (by timestamp)
# ----------------------------------------------------------

def get_latest_per_scale(records):
    latest = {}
    for r in records:
        scale = r["scale"]
        if scale not in latest:
            latest[scale] = r
        else:
            if r.get("timestamp", 0) > latest[scale].get("timestamp", 0):
                latest[scale] = r
    return latest


# ----------------------------------------------------------
# FORMATTING HELPERS
# ----------------------------------------------------------

def format_float(x):
    return f"{x:.4f}"


def scale_label(scale):
    mapping = {
        "small": "1.5B",
        "medium": "3B",
        "large": "7B"
    }
    return mapping.get(scale, scale)


# ----------------------------------------------------------
# LATEX TABLE PRINTER
# ----------------------------------------------------------

def generate_latex_table(title, header, rows):
    if not rows:
        return

    print("\n% ====================================================")
    print(f"% {title}")
    print("% ====================================================")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print(header)
    print("\\midrule")

    for row in rows:
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{{title}}}")
    print("\\end{table}")
    print()


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from exp5 JSON logs."
    )
    parser.add_argument(
        "--logfile",
        type=str,
        required=True,
        help="Path to exp5_comparison_results.json"
    )
    args = parser.parse_args()

    records = load_results(args.logfile)
    latest = get_latest_per_scale(records)

    scales_order = ["small", "medium", "large"]

    # --------------------------------------------------
    # SEC1
    # --------------------------------------------------
    rows = []
    for scale in scales_order:
        if scale in latest and "sec1" in latest[scale]["sections"]:
            sec = latest[scale]["sections"]["sec1"]
            rows.append(
                f"{scale_label(scale)} & "
                f"{format_float(sec['peak_kl'])} & "
                f"{format_float(sec['peak_js'])} \\\\"
            )

    generate_latex_table(
        "Unstructured Weight Mutation (SEC1)",
        "Model & Peak KL & Peak JS \\\\",
        rows
    )

    # --------------------------------------------------
    # SEC2
    # --------------------------------------------------
    rows = []
    for scale in scales_order:
        if scale in latest and "sec2" in latest[scale]["sections"]:
            sec = latest[scale]["sections"]["sec2"]
            rows.append(
                f"{scale_label(scale)} & "
                f"{format_float(sec['peak_kl'])} & "
                f"{format_float(sec['peak_js'])} \\\\"
            )

    generate_latex_table(
        "Structured Gradient Mutation (SEC2)",
        "Model & Peak KL & Peak JS \\\\",
        rows
    )

    # --------------------------------------------------
    # SEC3
    # --------------------------------------------------
    rows = []
    for scale in scales_order:
        if scale in latest and "sec3" in latest[scale]["sections"]:
            sec = latest[scale]["sections"]["sec3"]
            rows.append(
                f"{scale_label(scale)} & "
                f"{format_float(sec['peak_kl'])} & "
                f"{format_float(sec['peak_js'])} \\\\"
            )

    generate_latex_table(
        "RLAE Behavioral Divergence (SEC3)",
        "Model & Peak KL & Peak JS \\\\",
        rows
    )

    # --------------------------------------------------
    # POST RESET
    # --------------------------------------------------
    rows = []
    for scale in scales_order:
        if scale in latest and "post_reset" in latest[scale]["sections"]:
            sec = latest[scale]["sections"]["post_reset"]
            rows.append(
                f"{scale_label(scale)} & "
                f"{format_float(sec['kl'])} & "
                f"{format_float(sec['js'])} \\\\"
            )

    generate_latex_table(
        "Post-Reset Divergence",
        "Model & KL & JS \\\\",
        rows
    )


if __name__ == "__main__":
    main()
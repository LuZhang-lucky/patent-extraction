import os
import json
import csv
from statistics import mean

INPUT_DIR = "results_v5_qwen3_8b_postfix"
OUTPUT_CSV = "evaluation_summary_v5_postfix.csv"

CONF_FIELDS = [
    "publiceringsnummer",
    "dokumentkod",
    "publiceringsdatum",
    "ansokningsnummer",
    "ansokningsdatum",
    "patenttid_fr",
    "prioritetsdatum",
    "prioritetsnummer",
    "prioritetsland",
    "sokande",
    "uppfinnare",
    "benamning",
    "klass",
    "IPC",
]

def has_value(v):
    if v is None:
        return False
    if isinstance(v, list) and len(v) == 0:
        return False
    if isinstance(v, str) and not v.strip():
        return False
    return True

def main():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".json")])

    rows = []
    field_presence = {f: 0 for f in CONF_FIELDS}
    field_scores = {f: [] for f in CONF_FIELDS}
    doc_scores = []

    for filename in files:
        path = os.path.join(INPUT_DIR, filename)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conf = data.get("_confidence", {})
        fields = conf.get("fields", {})
        flags = conf.get("flags", [])
        doc_conf = conf.get("document_confidence")

        if doc_conf is not None:
            doc_scores.append(doc_conf)

        row = {
            "file": filename,
            "document_confidence": doc_conf,
            "flag_count": len(flags),
            "flags": " | ".join(flags),
        }

        present_count = 0
        for field in CONF_FIELDS:
            value = data.get(field)
            score = fields.get(field)

            present = has_value(value)
            if present:
                present_count += 1
                field_presence[field] += 1

            if score is not None:
                field_scores[field].append(score)

            row[f"{field}_present"] = 1 if present else 0
            row[f"{field}_score"] = score

        row["present_field_count"] = present_count
        rows.append(row)

    fieldnames = [
        "file",
        "document_confidence",
        "present_field_count",
        "flag_count",
        "flags",
    ]

    for field in CONF_FIELDS:
        fieldnames.append(f"{field}_present")
        fieldnames.append(f"{field}_score")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved file-level summary -> {OUTPUT_CSV}")

    print("\n=== Dataset Summary ===")
    print(f"Total files: {len(files)}")
    print(f"Average document confidence: {round(mean(doc_scores), 4) if doc_scores else 'N/A'}")

    print("\nField presence:")
    for field in CONF_FIELDS:
        print(f"  {field}: {field_presence[field]}/{len(files)}")

    print("\nAverage field score:")
    for field in CONF_FIELDS:
        avg_score = round(mean(field_scores[field]), 4) if field_scores[field] else "N/A"
        print(f"  {field}: {avg_score}")

if __name__ == "__main__":
    main()
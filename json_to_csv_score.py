import os
import json
import csv

INPUT_DIR = "results_v5_qwen3_8b_postfix"
OUTPUT_CSV = "results_v5_postfix.csv"

FIELDS = [
    "publiceringsnummer",
    "dokumentkod",
    "publiceringsdatum",
    "ansokningsnummer",
    "ansokningsdatum",
    "patenttid_fr",
    "prioritetsdatum",
    "prioritetsnummer",
    "prioritetsland",
    "benamning",
    "klass",
    "IPC",
]

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

def flatten_sokande(value):
    if not isinstance(value, list):
        return ""
    parts = []
    for item in value:
        if isinstance(item, dict):
            namn = item.get("namn") or ""
            landskod = item.get("landskod") or ""
            if landskod:
                parts.append(f"{namn} ({landskod})")
            else:
                parts.append(namn)
        elif isinstance(item, str):
            parts.append(item)
    return " | ".join([p for p in parts if p])

def flatten_uppfinnare(value):
    if not isinstance(value, list):
        return ""
    return " | ".join([str(x) for x in value if x])

def main():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".json")])

    rows = []

    for filename in files:
        path = os.path.join(INPUT_DIR, filename)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        row = {
            "file": filename,
            "publiceringsnummer": data.get("publiceringsnummer"),
            "dokumentkod": data.get("dokumentkod"),
            "publiceringsdatum": data.get("publiceringsdatum"),
            "ansokningsnummer": data.get("ansokningsnummer"),
            "ansokningsdatum": data.get("ansokningsdatum"),
            "patenttid_fr": data.get("patenttid_fr"),
            "prioritetsdatum": data.get("prioritetsdatum"),
            "prioritetsnummer": data.get("prioritetsnummer"),
            "prioritetsland": data.get("prioritetsland"),
            "sokande": flatten_sokande(data.get("sokande")),
            "uppfinnare": flatten_uppfinnare(data.get("uppfinnare")),
            "benamning": data.get("benamning"),
            "klass": data.get("klass"),
            "IPC": data.get("IPC"),
        }

        conf = data.get("_confidence", {})
        conf_fields = conf.get("fields", {})

        row["document_confidence"] = conf.get("document_confidence")
        row["confidence_flags"] = " | ".join(conf.get("flags", []))

        for field in CONF_FIELDS:
            row[f"{field}_score"] = conf_fields.get(field)

        rows.append(row)

    fieldnames = [
        "file",
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
        "document_confidence",
        "confidence_flags",
    ] + [f"{field}_score" for field in CONF_FIELDS]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
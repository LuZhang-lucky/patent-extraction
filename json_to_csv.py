# -*- coding: utf-8 -*-
import os
import json
import csv

RESULT_DIR = "results_20_rename_2"   # folder containing JSON files
OUTPUT_CSV = "results_20_rename_2.csv"


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def safe_join_list(value):
    if isinstance(value, list):
        return " | ".join(str(x) for x in value)
    if value is None:
        return ""
    return str(value)


def safe_get(d, *keys, default=""):
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def safe_str(value):
    if value is None:
        return ""
    return str(value)


def join_people(people_list):
    """
    Convert:
    [
      {"namn": "GENERAL ELECTRIC Co.", "landskod": "US"},
      {"namn": "Another Name", "landskod": None}
    ]
    into:
    'GENERAL ELECTRIC Co. (US) | Another Name'
    """
    if not isinstance(people_list, list):
        return ""

    parts = []
    for person in people_list:
        if not isinstance(person, dict):
            parts.append(str(person))
            continue

        namn = person.get("namn")
        landskod = person.get("landskod")

        if namn and landskod:
            parts.append(f"{namn} ({landskod})")
        elif namn:
            parts.append(str(namn))
        elif landskod:
            parts.append(f"({landskod})")

    return " | ".join(parts)


def main():
    json_files = sorted(
        [f for f in os.listdir(RESULT_DIR) if f.lower().endswith(".json")]
    )

    rows = []

    for filename in json_files:
        path = os.path.join(RESULT_DIR, filename)
        data = load_json_file(path)

        row = {
            "filename": filename.replace(".json", ".pdf"),
            "layout_type": "",

            # document_id
            "publiceringsnummer_ai": safe_str(
                safe_get(data,  "publiceringsnummer")
            ),
            "dokumentkod_ai": safe_str(
                safe_get(data, "dokumentkod")
            ),
            "publiceringsdatum_ai": safe_str(
                safe_get(data,  "publiceringsdatum")
            ),

            # ansoknings_id
            "ansokningsnummer_ai": safe_str(
                safe_get(data,  "ansokningsnummer")
            ),
            "ansokningsdatum_ai": safe_str(
                safe_get(data, "ansokningsdatum")
            ),
            "patenttid_fr_ai": safe_str(
                safe_get(data,  "patenttid_fr")
            ),

            # prioritetsuppgifter
            "prioritetsdatum_ai": safe_str(
                safe_get(data, "prioritetsdatum")
            ),
            "prioritetsnummer_ai": safe_str(
                safe_get(data,  "prioritetsnummer")
            ),
            "prioritetsland_ai": safe_str(
                safe_get(data,  "prioritetsland")
            ),

            # sokande / uppfinnare
            "sokande_ai": join_people(data.get("sokande", [])),
        
            "uppfinnare_ai": safe_join_list(data.get("uppfinnare", [])),

            # benamning
            "benamning_ai": safe_str(data.get("benamning")),

            # klassificering
            "klass_ai": safe_str(
                safe_get(data,  "klass")
            ),
            "IPC_ai": safe_str(
                safe_get(data,  "IPC")
            ),

            "comments": ""
        }

        rows.append(row)

    fieldnames = [
        "filename",
        "layout_type",
        "publiceringsnummer_ai",
        "dokumentkod_ai",
        "publiceringsdatum_ai",
        "ansokningsnummer_ai",
        "ansokningsdatum_ai",
        "patenttid_fr_ai",
        "prioritetsdatum_ai",
        "prioritetsnummer_ai",
        "prioritetsland_ai",
        "sokande_ai",
        "uppfinnare_ai",
        "benamning_ai",
        "klass_ai",
        "IPC_ai",
        "comments"
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. CSV saved as {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
import os
import re
import csv
import json
import time
import fitz
import logging
from pathlib import Path

from ollama import Client
from json_repair import repair_json

# ===== 1. SETTINGS =====
OLLAMA_HOST = "http://192.168.10.236:11434"
MODEL_NAME = "qwen3:8b"

client = Client(host=OLLAMA_HOST)

PDF_DIR = "raw_pdfs_C"
RESULT_DIR = "results_v5_qwen3_8b_postfix"
DEBUG_DIR = "debug_v5_score_postfix"

EXPECTED_KEYS = {
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
}

DEFAULT_SCHEMA = {
    "publiceringsnummer": None,
    "dokumentkod": None,
    "publiceringsdatum": None,
    "ansokningsnummer": None,
    "ansokningsdatum": None,
    "patenttid_fr": None,
    "prioritetsdatum": None,
    "prioritetsnummer": None,
    "prioritetsland": None,
    "sokande": [],
    "uppfinnare": [],
    "benamning": None,
    "klass": None,
    "IPC": None,
}

# ===== LOGGING =====
os.makedirs(DEBUG_DIR, exist_ok=True)
log_path = os.path.join(DEBUG_DIR, "run.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8")
    ]
)

def log_step(filename, stage, status, message=""):
    logging.info(f"{filename} | {stage} | {status} | {message}")

def save_debug_file(subfolder, filename, suffix, content):
    folder = os.path.join(DEBUG_DIR, subfolder)
    os.makedirs(folder, exist_ok=True)
    safe_name = Path(filename).stem
    path = os.path.join(folder, f"{safe_name}.{suffix}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# ===== 2. READ PDF TEXT =====
def extract_text_from_pdf(pdf_path, max_lines=60):
    with fitz.open(pdf_path) as doc:
        if len(doc) == 0:
            raise ValueError("PDF has no pages")

        page = doc[0]
        text = page.get_text("text")

        if not text or not text.strip():
            raise ValueError("No text extracted from first page")

        lines = [line.rstrip() for line in text.split("\n")]
        lines = [line for line in lines if line.strip()]
        return "\n".join(lines[:max_lines]).strip()

# ===== 3. BUILD PROMPT =====
def build_prompt(filename, patent_text):
    prompt = f"""
You extract bibliographic metadata from historical Swedish patent documents (PRV).

Return EXACTLY one valid JSON object.
No explanation.
No markdown.
No comments.
No code fences.
No extra text before or after the JSON.

Use exactly this structure:
{{
  "publiceringsnummer": null,
  "dokumentkod": null,
  "publiceringsdatum": null,
  "ansokningsnummer": null,
  "ansokningsdatum": null,
  "patenttid_fr": null,
  "prioritetsdatum": null,
  "prioritetsnummer": null,
  "prioritetsland": null,
  "sokande": [],
  "uppfinnare": [],
  "benamning": null,
  "klass": null,
  "IPC": null
}}

General rules:
- Extract values only when explicitly present in the document.
- Do not guess, infer, or hallucinate.
- If a scalar value is missing, return null.
- If sokande or uppfinnare is missing, return [].
- Never return empty strings ("").
- Every key in the schema must appear exactly once.

Field rules:
- publiceringsnummer: publication number near "PATENT N". Extract only the number.
- dokumentkod: document code from filename if present, e.g. ".B.pdf" -> "B", ".C1.pdf" -> "C1".
- publiceringsdatum: publication date after "PUBLICERAT DEN".
- ansokningsnummer: application number near "Ans.", "Ans. nr", "P.ans.nr", or before "inkom den". Remove label text.
- ansokningsdatum: filing date after "inkom den" or "Ans. den".
- patenttid_fr: date after "PATENTTID FRÅN DEN".
- benamning: patent title as a single string.
- sokande: applicant/patent holder only. Return:
  [{{"namn": "...", "landskod": "..."}}]
  If country code is missing, use null.
- uppfinnare: extract only if the label "Uppfinnare" appears explicitly. Return a list of names only. Do not include country codes. Do not infer from other names.
  If uncertain, return null rather than guessing.

Date rules:
- Dates are day-month-year. Normalize to YYYYMMDD.
- Convert Roman numeral months I-XII to 01-12.
- Resolve OCR confusion between I and 1 only when the date context is clear.
- If a date is unclear or incomplete, return null.

Classification rules:

Extract classification values only from the section labeled "KLASS".

Fields:
- "IPC": international classification
- "klass": Swedish national classification

Layout rules:
- Earlier layout:
  - If only "KLASS" appears and a code is on the right side of the label, that right-side code is Swedish classification.
- Later layout:
  - "INTERNATIONELL" → IPC
  - "SVENSK" → Swedish classification
  - IPC is usually below to "INTERNATIONELL"
  - Swedish classification is usually below to "SVENSK"
- Later layout with only "KLASS":
  - below "KLASS" = IPC

Format rules:
- IPC begins with a letter A–H
- Swedish classification begins with a number

Restrictions:
- Do not guess missing values
- Do not swap the fields

Null rules:
- If only IPC is present, set "klass" to null
- If only Swedish classification is present, set "IPC" to null


Priority information rules:
- Extract priority data only from an explicit priority statement "Prioritet begärd från den ...".
- Extract: prioritetsdatum, prioritetsland, prioritetsnummer.
- Normalize prioritetsdatum to YYYYMMDD.
- From text like "(US ****)", return prioritetsland = "US" and prioritetsnummer = "****".
- Do not use ansokningsnummer, publiceringsnummer, patentnummer, or classification codes as prioritetsnummer.
- If any priority value is not explicit, return:
  prioritetsdatum = null
  prioritetsnummer = null
  prioritetsland = null

Filename:
{filename}

Patent text:
{patent_text}

Final output rules:
- Return exactly one JSON object.
- Do not wrap the JSON in markdown or code fences.
- Do not include any text before or after the JSON.
- Use null instead of empty string.
- The JSON must be syntactically valid.
""".strip()
    return prompt

def build_retry_prompt(original_prompt, raw_response, error_message):
    return f"""
The previous response was invalid.

Error:
{error_message}

Previous response:
{raw_response}

Try again.

Return EXACTLY one valid JSON object only.
No markdown.
No comments.
No explanation.
No text before or after the JSON.
Use null instead of empty strings.

{original_prompt}
""".strip()

# ===== 4. SEND TO MODEL =====
def ask_model(prompt):
    response = client.generate(
        model=MODEL_NAME,
        prompt=prompt,
        format="json",
        stream=False,
        options={
            "temperature": 0,
            "top_p": 1,
            "num_predict": 700,
            "repeat_penalty": 1.0,
        }
    )

    if isinstance(response, dict):
        return response.get("response", "")
    return response.response

# ===== SAFE JSON LOAD =====
def safe_json_load(text):
    text = text.encode("utf-8", "ignore").decode("utf-8")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_json(text)
        return json.loads(repaired)

# ===== JSON EXTRACTION =====
def find_first_balanced_json_object(text):
    start = None
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            if start is not None:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    return None

def extract_json_from_response(raw_response):
    raw_response = raw_response.strip()

    # 1) direct parse
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    # 2) remove code fences if any
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_response, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3) balanced object extraction
    json_text = find_first_balanced_json_object(cleaned)
    if not json_text:
        raise ValueError("No valid JSON found in model response")

    return safe_json_load(json_text)

# ===== NORMALIZATION / REPAIR =====
def deep_clean_empty_strings(obj):
    if isinstance(obj, dict):
        return {k: deep_clean_empty_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_clean_empty_strings(x) for x in obj]
    if isinstance(obj, str):
        s = obj.strip()
        return None if s == "" else s
    return obj

def normalize_schema(data):
    merged = DEFAULT_SCHEMA.copy()
    merged.update(data)
    return merged

def repair_common_format_issues(data):
    # sokande
    if isinstance(data.get("sokande"), str):
        name = data["sokande"].strip()
        data["sokande"] = [{"namn": name, "landskod": None}] if name else []
    elif data.get("sokande") is None:
        data["sokande"] = []
    elif isinstance(data.get("sokande"), list):
        repaired = []
        for item in data["sokande"]:
            if isinstance(item, str):
                item = item.strip()
                if item:
                    repaired.append({"namn": item, "landskod": None})
            elif isinstance(item, dict):
                namn = item.get("namn")
                landskod = item.get("landskod")
                if isinstance(namn, str):
                    namn = namn.strip()
                if isinstance(landskod, str):
                    landskod = landskod.strip() or None
                if namn:
                    repaired.append({"namn": namn, "landskod": landskod})
        data["sokande"] = repaired
    else:
        data["sokande"] = []

    # uppfinnare
    if isinstance(data.get("uppfinnare"), str):
        value = data["uppfinnare"].strip()
        data["uppfinnare"] = [value] if value else []
    elif data.get("uppfinnare") is None:
        data["uppfinnare"] = []
    elif isinstance(data.get("uppfinnare"), list):
        repaired = []
        for item in data["uppfinnare"]:
            if isinstance(item, str):
                item = item.strip()
                if item:
                    repaired.append(item)
        data["uppfinnare"] = repaired
    else:
        data["uppfinnare"] = []

    # klass
    if isinstance(data.get("klass"), str):
        data["klass"] = data["klass"].strip() or None

    # IPC
    if isinstance(data.get("IPC"), list):
        data["IPC"] = data["IPC"][0] if data["IPC"] else None
    if isinstance(data.get("IPC"), str):
        data["IPC"] = data["IPC"].strip() or None

    # other scalar fields
    scalar_fields = [
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
    ]
    for field in scalar_fields:
        if isinstance(data.get(field), str):
            data[field] = data[field].strip() or None

    return data

# ===== VALIDATION =====
def validate_result(data):
    if not isinstance(data, dict):
        raise ValueError("Output is not a JSON object")

    core_keys = {k for k in data.keys() if not k.startswith("_")}

    missing_keys = EXPECTED_KEYS - core_keys
    extra_keys = core_keys - EXPECTED_KEYS

    if missing_keys:
        raise ValueError(f"Missing keys: {sorted(missing_keys)}")
    if extra_keys:
        raise ValueError(f"Unexpected keys: {sorted(extra_keys)}")

    if not isinstance(data.get("sokande"), list):
        raise ValueError("sokande must be a list")

    if not isinstance(data.get("uppfinnare"), list):
        raise ValueError("uppfinnare must be a list")

    for item in data["sokande"]:
        if not isinstance(item, dict):
            raise ValueError("Each sokande item must be an object")
        if "namn" not in item or "landskod" not in item:
            raise ValueError("Each sokande item must contain namn and landskod")

        if item["namn"] is not None and not isinstance(item["namn"], str):
            raise ValueError("sokande.namn must be string or null")
        if item["landskod"] is not None and not isinstance(item["landskod"], str):
            raise ValueError("sokande.landskod must be string or null")

    for item in data["uppfinnare"]:
        if not isinstance(item, str):
            raise ValueError("Each uppfinnare item must be a string")

    for k, v in data.items():
        if isinstance(v, str) and v.strip() == "":
            raise ValueError(f"{k} is empty string; use null")

    if isinstance(data.get("klass"), str) and not data["klass"].strip():
        raise ValueError("klass is empty string; use null")


def null_score(data):
    core_items = {k: v for k, v in data.items() if not k.startswith("_")}
    total = len(core_items)
    missing = sum(1 for v in core_items.values() if v is None or v == [])
    return missing, total

def looks_like_ipc(value):
    if not isinstance(value, str):
        return False
    v = re.sub(r"\s+", "", value).upper()
    return re.match(r"^[A-H]\d", v) is not None

def looks_like_klass(value):
    if not isinstance(value, str):
        return False
    v = value.strip()
    return re.match(r"^\d", v) is not None

def apply_classification_postfix(data):
    ipc = data.get("IPC")
    klass = data.get("klass")

    # 情况1：klass 看起来像 IPC，而 IPC 为空
    # 直接把 klass 挪到 IPC
    if klass is not None and looks_like_ipc(klass):
        if ipc is None:
            data["IPC"] = klass
            data["klass"] = None
            return data

    # 情况2：klass 看起来像 IPC，而 IPC 已经有值
    # 那 klass 置空，避免错填
    if klass is not None and looks_like_ipc(klass):
        data["klass"] = None
        return data

    # 情况3：IPC 不像 IPC，清空
    if ipc is not None and not looks_like_ipc(ipc):
        data["IPC"] = None

    # 情况4：klass 不像 klass，清空
    if klass is not None and not looks_like_klass(klass):
        data["klass"] = None

    return data

# ===== FULL CONFIDENCE SCORING =====

CONFIDENCE_FIELDS = [
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


def score_scalar_field(value):
    if value is None:
        return None
    if isinstance(value, str):
        return 0.8 if value.strip() else 0.0
    return 0.6

def score_date_field(value):
    if value is None:
        return None
    if isinstance(value, str) and re.fullmatch(r"\d{8}", value):
        return 1.0
    if isinstance(value, str) and value.strip():
        return 0.5
    return 0.0


def score_dokumentkod_field(value, filename):
    if value is None:
        return 0.0

    value = str(value).strip().upper()
    m = re.search(r"\.([A-Z]\d?)\.pdf$", filename, re.IGNORECASE)

    if m:
        expected = m.group(1).upper()
        if value == expected:
            return 1.0
        return 0.3

    return 0.6

def score_ipc_field(value):
    if value is None:
        return None

    v = str(value).replace(" ", "").strip().upper()
    if re.match(r"^[A-H]\d", v):
        return 1.0
    if v:
        return 0.4
    return 0.0

def score_klass_field(value):
    if value is None:
        return None

    v = str(value).strip()
    if re.match(r"^\d", v):
        return 1.0
    if v:
        return 0.4
    return 0.0

def score_sokande_field(value):
    if value is None:
        return None
    if not isinstance(value, list):
        return 0.0
    if len(value) == 0:
        return None

    valid_items = 0
    for item in value:
        if isinstance(item, dict):
            namn = item.get("namn")
            if isinstance(namn, str) and namn.strip():
                valid_items += 1

    if valid_items == len(value):
        return 1.0
    if valid_items > 0:
        return 0.5
    return 0.0

def score_uppfinnare_field(value):
    if value is None:
        return None
    if not isinstance(value, list):
        return 0.0
    if len(value) == 0:
        return None

    valid_items = 0
    for item in value:
        if isinstance(item, str) and item.strip():
            valid_items += 1

    if valid_items == len(value):
        return 1.0
    if valid_items > 0:
        return 0.5
    return 0.0

def score_benamning_field(value):
    if value is None:
        return 0.0
    if isinstance(value, str):
        v = value.strip()
        if len(v) >= 8:
            return 1.0
        if len(v) > 0:
            return 0.5
    return 0.0

def score_field(field, value, filename):
    if field == "dokumentkod":
        return score_dokumentkod_field(value, filename)

    if field in ["publiceringsdatum", "ansokningsdatum", "patenttid_fr", "prioritetsdatum"]:
        return score_date_field(value)

    if field == "IPC":
        return score_ipc_field(value)

    if field == "klass":
        return score_klass_field(value)

    if field == "sokande":
        return score_sokande_field(value)

    if field == "uppfinnare":
        return score_uppfinnare_field(value)

    if field == "benamning":
        return score_benamning_field(value)

    return score_scalar_field(value)


def score_document_full(data, text, filename):
    fields = {}

    for field in CONFIDENCE_FIELDS:
        value = data.get(field)
        score = score_field(field, value, filename)
        fields[field] = None if score is None else round(score, 4)

    valid_scores = [v for v in fields.values() if v is not None]

    if valid_scores:
        document_confidence = round(sum(valid_scores) / len(valid_scores), 4)
    else:
        document_confidence = None

    flags = []

    if document_confidence is not None:
        if document_confidence < 0.5:
            flags.append("very_low_confidence")
        elif document_confidence < 0.7:
            flags.append("needs_review")

    if fields["IPC"] is not None and fields["IPC"] < 0.5:
        flags.append("low_confidence_ipc")

    if fields["klass"] is not None and fields["klass"] < 0.5:
        flags.append("low_confidence_klass")

    if fields["prioritetsnummer"] is not None and fields["prioritetsnummer"] < 0.5:
        flags.append("low_confidence_prioritetsnummer")

    if fields["prioritetsland"] is not None and fields["prioritetsland"] < 0.5:
        flags.append("low_confidence_prioritetsland")

    if fields["prioritetsdatum"] is not None and fields["prioritetsdatum"] < 0.5:
        flags.append("low_confidence_prioritetsdatum")

    return {
        "document_confidence": document_confidence,
        "fields": fields,
        "flags": flags
    }


# ===== 6. SAVE RESULT =====
def save_result(filename, parsed_json):
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_path = os.path.join(RESULT_DIR, filename.replace(".pdf", ".json"))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, ensure_ascii=False, indent=2)

    return output_path

# ===== FAILURE TYPE =====
def classify_error(err_msg):
    msg = err_msg.lower()

    if "no valid json" in msg or "jsondecodeerror" in msg:
        return "parse_error"
    if "missing keys" in msg or "unexpected keys" in msg:
        return "schema_error"
    if "must be a list" in msg or "empty string" in msg:
        return "validation_error"
    return "unknown"

# ===== PROCESS ONE FILE =====
def process_file(pdf_path):
    filename = os.path.basename(pdf_path)
    raw_response = None
    text = None
    prompt = None
    parsed_json = None
    start = time.time()

    try:
        log_step(filename, "extract_text", "start")
        text = extract_text_from_pdf(pdf_path, max_lines=60)
        log_step(filename, "extract_text", "ok", f"text_len={len(text)}")
        save_debug_file("text", filename, "txt", text)

        log_step(filename, "build_prompt", "start")
        prompt = build_prompt(filename, text)
        log_step(filename, "build_prompt", "ok", f"prompt_len={len(prompt)}")
        save_debug_file("prompts", filename, "prompt.txt", prompt)

        log_step(filename, "ask_model", "start")
        raw_response = ask_model(prompt)
        log_step(filename, "ask_model", "ok", f"output_len={len(raw_response)}")
        save_debug_file("raw", filename, "raw.txt", raw_response)

        log_step(filename, "parse_json", "start")
        try:
            parsed_json = extract_json_from_response(raw_response)
        except Exception as e:
            log_step(filename, "retry", "start", repr(e))
            retry_prompt = build_retry_prompt(prompt, raw_response, repr(e))
            save_debug_file("retry_prompts", filename, "retry.prompt.txt", retry_prompt)

            raw_retry = ask_model(retry_prompt)
            save_debug_file("retry_raw", filename, "retry.raw.txt", raw_retry)

            raw_response = raw_retry
            parsed_json = extract_json_from_response(raw_response)

        parsed_json = deep_clean_empty_strings(parsed_json)
        parsed_json = normalize_schema(parsed_json)
        parsed_json = repair_common_format_issues(parsed_json)
       
        log_step(filename, "parse_json", "ok")

        log_step(filename, "validate", "start")
        validate_result(parsed_json)
        log_step(filename, "validate", "ok")

        parsed_json = apply_classification_postfix(parsed_json)

        parsed_json["_confidence"] = score_document_full(parsed_json, text, filename)

        missing, total = null_score(parsed_json)
        if missing >= 10:
            log_step(filename, "quality", "warn", f"high_null_rate={missing}/{total}")

        output_path = save_result(filename, parsed_json)
        elapsed = round(time.time() - start, 2)
        log_step(filename, "save_result", "ok", output_path)
        log_step(filename, "done", "ok", f"{elapsed}s")

        return {
            "file": filename,
            "status": "ok",
            "stage": "done",
            "error": None,
            "error_type": None,
            "output_path": output_path,
        }

    except Exception as e:
        err = repr(e)
        error_type = classify_error(err)
        log_step(filename, "failed", "fail", err)

        if text is not None:
            save_debug_file("text_failed", filename, "txt", text)
        if prompt is not None:
            save_debug_file("prompts_failed", filename, "prompt.txt", prompt)
        if raw_response is not None:
            save_debug_file("raw_failed", filename, "raw.txt", raw_response)
        if parsed_json is not None:
            save_debug_file(
                "json_failed",
                filename,
                "json",
                json.dumps(parsed_json, ensure_ascii=False, indent=2)
            )

        return {
            "file": filename,
            "status": "failed",
            "stage": "failed",
            "error": err,
            "error_type": error_type,
            "output_path": None,
        }

# ===== SUMMARY =====
def write_summary(results, path=os.path.join(DEBUG_DIR, "run_summary.csv")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "status", "stage", "error", "error_type", "output_path"]
        )
        writer.writeheader()
        writer.writerows(results)

# ===== 7. MAIN =====
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    print(f"Found {len(pdf_files)} PDF files.")

    results = []

    for filename in pdf_files:
        pdf_path = os.path.join(PDF_DIR, filename)
        print(f"\nProcessing {filename}")
        result = process_file(pdf_path)
        results.append(result)

        if result["status"] == "ok":
            print(f"Saved result -> {result['output_path']}")
        else:
            print(f"Failed -> {filename}: {result['error']}")

        time.sleep(1)

    write_summary(results)

    failed = [r for r in results if r["status"] == "failed"]
    print(f"\nDone. Total={len(results)} Failed={len(failed)}")

    if failed:
        print("\nFailure breakdown:")
        counts = {}
        for r in failed:
            key = r.get("error_type") or "unknown"
            counts[key] = counts.get(key, 0) + 1
        for k, v in sorted(counts.items()):
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
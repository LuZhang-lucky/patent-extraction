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
MODEL_NAME = "qwen3-vl:latest"

client = Client(host=OLLAMA_HOST)

PDF_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "PDF100")
RESULT_DIR = "results_v6_qwen3vl_imageonly"
DEBUG_DIR = "debug_v6_imageonly"

EXPECTED_KEYS = {
    "benamning",
    "ansokningsnummer",
    "publiceringsnummer",
    "prioritetsnummer",
    "sokande",
    "uppfinnare",
    "ombud",
    "DPK",
    "IPC",
    "ansokningsdatum",
    "beviljatdatum",
    "utlaggningsdatum",
}

DEFAULT_SCHEMA = {
    "benamning": None,
    "ansokningsnummer": None,
    "publiceringsnummer": None,
    "prioritetsnummer": None,
    "sokande": [],
    "uppfinnare": [],
    "ombud": None,
    "DPK": None,
    "IPC": None,
    "ansokningsdatum": None,
    "beviljatdatum": None,
    "utlaggningsdatum": None,
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


def save_debug_binary_file(subfolder, filename, suffix, content):
    folder = os.path.join(DEBUG_DIR, subfolder)
    os.makedirs(folder, exist_ok=True)
    safe_name = Path(filename).stem
    path = os.path.join(folder, f"{safe_name}.{suffix}")
    with open(path, "wb") as f:
        f.write(content)
    return path

# ===== 2. RENDER FIRST PAGE IMAGE =====
def render_first_page_png(pdf_path, dpi=220):
    with fitz.open(pdf_path) as doc:
        if len(doc) == 0:
            raise ValueError("PDF has no pages")

        page = doc[0]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        png_bytes = pix.tobytes("png")

        if not png_bytes:
            raise ValueError("No image rendered from first page")

        return png_bytes

# ===== 3. BUILD PROMPT =====
def build_prompt(filename):
    prompt = f"""
You extract bibliographic metadata from historical Swedish patent documents.
You are given only an image of the first page.
Use only this image as source.

Return EXACTLY one valid JSON object.
No explanation.
No markdown.
No comments.
No extra text.

Use exactly this structure:
{{
    "benamning": null,
  "ansokningsnummer": null,
    "publiceringsnummer": null,
  "prioritetsnummer": null,
  "sokande": [],
  "uppfinnare": [],
    "ombud": null,
    "DPK": null,
    "IPC": null,
    "ansokningsdatum": null,
    "beviljatdatum": null,
    "utlaggningsdatum": null
}}

General rules:
- Extract values only when explicitly present in the document.
- Do NOT guess or infer missing values.
- If a value is missing, return null.
- For sokande and uppfinnare, return [] if missing.
- Keep all extracted text exactly as it appears (no normalization).
- Do not split names and countries. Keep full strings.

Field extraction rules:

benamning:
- Usually appears as the invention title.
- Often centered and prominent.
- May appear under "Benamning" or without explicit label.
- In many historical Swedish patents, the title is unlabeled.
- Typical position: in the middle/upper-middle of the first page, visually prominent, often between bibliographic header blocks and the body text.
- If there is one clearly prominent candidate line/phrase matching title style, extract it as benamning.
- Do not require an explicit "Benamning" label.

ansokningsnummer:
- Appears near "Ans.", "Ans. nr", "P.ans.nr", or "inkom den".
- Extract the number only.
- Remove label text.

publiceringsnummer:
- Appears near "PATENT No".
- Extract only the number.

prioritetsnummer:
- Appears near "Prior.", "Prioritetsnummer".
- Extract only if explicitly stated.

sokande:
- Appears under "Sokande", "Innehavare", or "Patenthavare".
- Extract full text strings (including country if present).
- Return a list of strings.

uppfinnare:
- Appears under "Uppfinnare".
- Extract only if the label "Uppfinnare" appears explicitly.
- Extract names as full strings.
- Return a list of strings.

ombud:
- Appears under "Ombud".
- Extract full text if present.

DPK:
- Swedish national classification.
- Often labeled as "Klass" or "SVENSK".
- Typically starts with a number.

IPC:
- International classification.
- Often labeled as "INT. KLASS" or "INTERNATIONELL".
- Typically starts with a letter A-H.

ansokningsdatum:
- Filing date after "inkom den" or "Ans. den".
- Extract the date only.

beviljatdatum:
- Grant date.
- Indicated by "beviljat den".

utlaggningsdatum:
- Public availability date.
- Indicated by "utlagd den", "utlaggningsdatum".

Date rules:
- Dates are day-month-year. Normalize to YYYYMMDD.
- Convert Roman numerals (I-XII) to months (01-12).
- If unclear, return null.

Filename:
{filename}

Source:
- First-page patent image only

Final output rules:
- Return exactly one JSON object.
- Ensure valid JSON format.
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


def build_title_only_prompt(filename):
        return f"""
You are given only an image of the first page of a historical Swedish patent.

Task:
- Extract ONLY the invention title and return it in key "benamning".

Return EXACTLY one valid JSON object with this structure:
{{
    "benamning": null
}}

Rules:
- Extract only if explicitly visible in the image.
- In these documents, the title is often unlabeled and centered/prominent in the middle or upper-middle area.
- Do not require a "Benamning" label.
- Keep the text exactly as it appears.
- If uncertain, return null.
- No explanation, no markdown, no extra text.

Filename:
{filename}
""".strip()

# ===== 4. SEND TO MODEL =====
def ask_model(prompt, image_bytes):
    response = client.generate(
        model=MODEL_NAME,
        prompt=prompt,
        format="json",
        think=False,
        images=[image_bytes],
        stream=False,
        options={
            "temperature": 0,
            "top_p": 1,
            "num_predict": 700,
            "repeat_penalty": 1.0,
        }
    )

    if isinstance(response, dict):
        primary = response.get("response", "")
        if isinstance(primary, str) and primary.strip():
            return primary

        fallback = response.get("thinking", "")
        if isinstance(fallback, str) and fallback.strip():
            return fallback

        return ""

    primary = getattr(response, "response", "") or ""
    if isinstance(primary, str) and primary.strip():
        return primary

    fallback = getattr(response, "thinking", "") or ""
    if isinstance(fallback, str) and fallback.strip():
        return fallback

    return ""

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


def extract_title_with_fallback(filename, image_bytes):
    title_prompt = build_title_only_prompt(filename)
    raw_title = ask_model(title_prompt, image_bytes)
    title_json = extract_json_from_response(raw_title)
    title_json = deep_clean_empty_strings(title_json)

    if isinstance(title_json, dict):
        title = title_json.get("benamning")
        if isinstance(title, str):
            cleaned = title.strip()
            return cleaned or None

    return None

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


def canonicalize_common_aliases(data):
    if not isinstance(data, dict):
        return data

    # Map common alternative keys from vision outputs into this schema.
    aliases = {
        "benamning": ["title", "titel", "invention_title", "patent_title", "benamning"],
        "ansokningsnummer": ["application_number", "application_no", "ansokningsnr", "ansnr"],
        "publiceringsnummer": ["patent_number", "publication_number", "publicering_number", "nr"],
        "prioritetsnummer": ["priority_number", "priority_no", "prior_no"],
        "ombud": ["agent", "attorney", "representative"],
        "DPK": ["class", "swedish_class", "svensk", "klass", "dpk"],
        "IPC": ["ipc_class", "international_class", "internationell", "int_klass", "ipc"],
        "ansokningsdatum": ["filing_date", "application_date", "ingivningsdatum"],
        "beviljatdatum": ["grant_date", "date_issued", "patent_meddelat"],
        "utlaggningsdatum": ["publication_date", "date_published", "offentlighetsdatum", "utlagd_datum"],
    }

    def has_value(v):
        if v is None:
            return False
        if isinstance(v, str):
            return bool(v.strip())
        if isinstance(v, list):
            return len(v) > 0
        return True

    for target, keys in aliases.items():
        if has_value(data.get(target)):
            continue
        for src in keys:
            if has_value(data.get(src)):
                data[target] = data[src]
                break

    # Normalize applicant/inventor aliases into list fields.
    if not has_value(data.get("sokande")):
        for src in ["assignee", "applicant", "sokande", "patenthavare", "innehavare"]:
            if has_value(data.get(src)):
                data["sokande"] = data[src]
                break

    if not has_value(data.get("uppfinnare")):
        for src in ["inventor", "inventors", "uppfinnare"]:
            if has_value(data.get(src)):
                data["uppfinnare"] = data[src]
                break

    return data


def normalize_date_to_yyyymmdd(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    s = value.strip()
    if not s:
        return None

    # Already normalized.
    if re.fullmatch(r"\d{8}", s):
        return s

    # Common numeric patterns: dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy
    m = re.fullmatch(r"(\d{1,2})[\./-](\d{1,2})[\./-](\d{4})", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return f"{y:04d}{mo:02d}{d:02d}"

    # yyyy-mm-dd / yyyy/mm/dd / yyyy.mm.dd
    m = re.fullmatch(r"(\d{4})[\./-](\d{1,2})[\./-](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return f"{y:04d}{mo:02d}{d:02d}"

    roman_months = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8,
        "IX": 9,
        "X": 10,
        "XI": 11,
        "XII": 12,
    }

    swedish_months = {
        "JAN": 1, "JANUARI": 1,
        "FEB": 2, "FEBRUARI": 2,
        "MAR": 3, "MARS": 3,
        "APR": 4, "APRIL": 4,
        "MAJ": 5,
        "JUN": 6, "JUNI": 6,
        "JUL": 7, "JULI": 7,
        "AUG": 8, "AUGUSTI": 8,
        "SEP": 9, "SEPT": 9, "SEPTEMBER": 9,
        "OKT": 10, "OKTOBER": 10,
        "NOV": 11, "NOVEMBER": 11,
        "DEC": 12, "DECEMBER": 12,
    }

    # Text/Roman patterns like "2 JULI 1942" or "2 IX 1942"
    compact = re.sub(r"[,]+", " ", s)
    compact = re.sub(r"\s+", " ", compact).strip()
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-zÅÄÖåäö\.]+|[IVX]+)\s+(\d{4})", compact)
    if m:
        d = int(m.group(1))
        month_token = m.group(2).replace(".", "").upper()
        y = int(m.group(3))

        mo = None
        if month_token in swedish_months:
            mo = swedish_months[month_token]
        elif month_token in roman_months:
            mo = roman_months[month_token]

        if mo is not None and 1 <= d <= 31:
            return f"{y:04d}{mo:02d}{d:02d}"

    # Leave unchanged if unclear; validator/scoring will handle remaining quality.
    return s

def repair_common_format_issues(data):
    data = canonicalize_common_aliases(data)

    # sokande
    if isinstance(data.get("sokande"), str):
        name = data["sokande"].strip()
        data["sokande"] = [name] if name else []
    elif data.get("sokande") is None:
        data["sokande"] = []
    elif isinstance(data.get("sokande"), list):
        repaired = []
        for item in data["sokande"]:
            if isinstance(item, str):
                item = item.strip()
                if item:
                    repaired.append(item)
            elif isinstance(item, dict):
                # Backward-compat repair if old shape is returned.
                namn = item.get("namn")
                if isinstance(namn, str) and namn.strip():
                    repaired.append(namn.strip())
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

    # DPK
    if isinstance(data.get("DPK"), str):
        data["DPK"] = data["DPK"].strip() or None

    # IPC
    if isinstance(data.get("IPC"), list):
        data["IPC"] = data["IPC"][0] if data["IPC"] else None
    if isinstance(data.get("IPC"), str):
        data["IPC"] = data["IPC"].strip() or None

    # other scalar fields
    scalar_fields = [
        "benamning",
        "ansokningsnummer",
        "publiceringsnummer",
        "prioritetsnummer",
        "ombud",
        "ansokningsdatum",
        "beviljatdatum",
        "utlaggningsdatum",
    ]
    for field in scalar_fields:
        if isinstance(data.get(field), str):
            data[field] = data[field].strip() or None

    # Normalize date fields to YYYYMMDD when possible.
    for field in ["ansokningsdatum", "beviljatdatum", "utlaggningsdatum"]:
        data[field] = normalize_date_to_yyyymmdd(data.get(field))

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
        if not isinstance(item, str):
            raise ValueError("Each sokande item must be a string")

    for item in data["uppfinnare"]:
        if not isinstance(item, str):
            raise ValueError("Each uppfinnare item must be a string")

    for k, v in data.items():
        if isinstance(v, str) and v.strip() == "":
            raise ValueError(f"{k} is empty string; use null")

    if isinstance(data.get("DPK"), str) and not data["DPK"].strip():
        raise ValueError("DPK is empty string; use null")


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
    dpk = data.get("DPK")

    # If DPK looks like IPC and IPC is empty, move DPK -> IPC.
    if dpk is not None and looks_like_ipc(dpk):
        if ipc is None:
            data["IPC"] = dpk
            data["DPK"] = None
            return data

    # If DPK looks like IPC while IPC already exists, clear DPK.
    if dpk is not None and looks_like_ipc(dpk):
        data["DPK"] = None
        return data

    # If IPC does not look like IPC, clear it.
    if ipc is not None and not looks_like_ipc(ipc):
        data["IPC"] = None

    # If DPK does not look like a Swedish class code, clear it.
    if dpk is not None and not looks_like_klass(dpk):
        data["DPK"] = None

    return data

# ===== FULL CONFIDENCE SCORING =====

CONFIDENCE_FIELDS = [
    "benamning",
    "ansokningsnummer",
    "publiceringsnummer",
    "prioritetsnummer",
    "sokande",
    "uppfinnare",
    "ombud",
    "DPK",
    "IPC",
    "ansokningsdatum",
    "beviljatdatum",
    "utlaggningsdatum",
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


def score_ipc_field(value):
    if value is None:
        return None

    v = str(value).replace(" ", "").strip().upper()
    if re.match(r"^[A-H]\d", v):
        return 1.0
    if v:
        return 0.4
    return 0.0

def score_dpk_field(value):
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
        if isinstance(item, str) and item.strip():
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
    if field in ["ansokningsdatum", "beviljatdatum", "utlaggningsdatum"]:
        return score_date_field(value)

    if field == "IPC":
        return score_ipc_field(value)

    if field == "DPK":
        return score_dpk_field(value)

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

    if fields["DPK"] is not None and fields["DPK"] < 0.5:
        flags.append("low_confidence_dpk")

    if fields["prioritetsnummer"] is not None and fields["prioritetsnummer"] < 0.5:
        flags.append("low_confidence_prioritetsnummer")

    if fields["beviljatdatum"] is not None and fields["beviljatdatum"] < 0.5:
        flags.append("low_confidence_beviljatdatum")

    if fields["utlaggningsdatum"] is not None and fields["utlaggningsdatum"] < 0.5:
        flags.append("low_confidence_utlaggningsdatum")

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
    image_bytes = None
    prompt = None
    parsed_json = None
    start = time.time()

    try:
        log_step(filename, "render_image", "start")
        image_bytes = render_first_page_png(pdf_path, dpi=220)
        log_step(filename, "render_image", "ok", f"image_bytes={len(image_bytes)}")
        save_debug_binary_file("images", filename, "png", image_bytes)

        log_step(filename, "build_prompt", "start")
        prompt = build_prompt(filename)
        log_step(filename, "build_prompt", "ok", f"prompt_len={len(prompt)}")
        save_debug_file("prompts", filename, "prompt.txt", prompt)

        log_step(filename, "ask_model", "start")
        raw_response = ask_model(prompt, image_bytes)
        log_step(filename, "ask_model", "ok", f"output_len={len(raw_response)}")
        save_debug_file("raw", filename, "raw.txt", raw_response)

        log_step(filename, "parse_json", "start")
        try:
            parsed_json = extract_json_from_response(raw_response)
        except Exception as e:
            log_step(filename, "retry", "start", repr(e))
            retry_prompt = build_retry_prompt(prompt, raw_response, repr(e))
            save_debug_file("retry_prompts", filename, "retry.prompt.txt", retry_prompt)

            raw_retry = ask_model(retry_prompt, image_bytes)
            save_debug_file("retry_raw", filename, "retry.raw.txt", raw_retry)

            raw_response = raw_retry
            parsed_json = extract_json_from_response(raw_response)

        parsed_json = deep_clean_empty_strings(parsed_json)
        parsed_json = normalize_schema(parsed_json)
        parsed_json = repair_common_format_issues(parsed_json)

        if parsed_json.get("benamning") is None:
            try:
                log_step(filename, "title_fallback", "start")
                fallback_title = extract_title_with_fallback(filename, image_bytes)
                if fallback_title:
                    parsed_json["benamning"] = fallback_title
                    log_step(filename, "title_fallback", "ok", "benamning_filled")
                else:
                    log_step(filename, "title_fallback", "ok", "benamning_still_null")
            except Exception as e:
                log_step(filename, "title_fallback", "warn", repr(e))
       
        log_step(filename, "parse_json", "ok")

        log_step(filename, "validate", "start")
        validate_result(parsed_json)
        log_step(filename, "validate", "ok")

        parsed_json = apply_classification_postfix(parsed_json)

        parsed_json["_confidence"] = score_document_full(parsed_json, None, filename)

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

        if image_bytes is not None:
            save_debug_binary_file("images_failed", filename, "png", image_bytes)
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
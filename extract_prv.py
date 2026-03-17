import re
import json
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Any




SWEDISH_MONTHS = {
    "JAN": "01",
    "JAN.": "01",
    "FEB": "02",
    "FEBR": "02",
    "FEBR.": "02",
    "MARS": "03",
    "APR": "04",
    "APR.": "04",
    "MAJ": "05",
    "JUNI": "06",
    "JULI": "07",
    "AUG": "08",
    "AUG.": "08",
    "SEPT": "09",
    "SEPT.": "09",
    "SEP": "09",
    "SEP.": "09",
    "OKT": "10",
    "OKT.": "10",
    "NOV": "11",
    "NOV.": "11",
    "DEC": "12",
    "DEC.": "12",
}

COUNTRY_NORMALIZATION = {
    "SVERIGE": "SE",
    "TYSKA RIKET": "DE",
    "TYSKLAND": "DE",
    "DANMARK": "DK",
    "NORGE": "NO",
    "FINLAND": "FI",
    "FRANKRIKE": "FR",
    "STORBRITANNIEN": "GB",
    "ENGLAND": "GB",
    "USA": "US",
    "FÖRENTA STATERNA": "US",
    "SCHWEIZ": "CH",
    "ITALIEN": "IT",
    "NEDERLÄNDERNA": "NL",
    "BELGIEN": "BE",
    "ÖSTERRIKE": "AT",
}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def normalize_country(text: str) -> Optional[str]:
    text = text.strip().upper().rstrip(".")
    return COUNTRY_NORMALIZATION.get(text)


def month_to_number(month: str) -> Optional[str]:
    return SWEDISH_MONTHS.get(month.strip().upper())


def parse_swedish_date(day: str, month: str, year: str) -> Optional[str]:
    mm = month_to_number(month)
    if not mm:
        return None
    return f"{year}{mm}{int(day):02d}"


def parse_numeric_date(day: str, month: str, year: str) -> str:
    return f"{year}{int(month):02d}{int(day):02d}"


def extract_page_lines(pdf_path: str, page_number: int = 0) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    data = page.get_text("dict")

    lines = []
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue

            bbox = line["bbox"]
            avg_size = sum(span.get("size", 0) for span in spans) / max(len(spans), 1)

            lines.append({
                "text": normalize_whitespace(text),
                "x0": bbox[0],
                "y0": bbox[1],
                "x1": bbox[2],
                "y1": bbox[3],
                "font_size": avg_size,
                "page": page_number + 1,
            })

    lines.sort(key=lambda x: (round(x["y0"], 1), x["x0"]))
    return lines


def merge_nearby_lines(lines: List[Dict[str, Any]], y_tolerance: float = 3.0) -> List[Dict[str, Any]]:
    merged = []
    current = None

    for line in lines:
        if current is None:
            current = line.copy()
            continue

        same_row = abs(line["y0"] - current["y0"]) <= y_tolerance
        if same_row:
            current["text"] += " " + line["text"]
            current["x1"] = max(current["x1"], line["x1"])
            current["y1"] = max(current["y1"], line["y1"])
            current["font_size"] = max(current["font_size"], line["font_size"])
        else:
            current["text"] = normalize_whitespace(current["text"])
            merged.append(current)
            current = line.copy()

    if current:
        current["text"] = normalize_whitespace(current["text"])
        merged.append(current)

    return merged


def first_page_text(lines: List[Dict[str, Any]]) -> str:
    return "\n".join(line["text"] for line in lines)


def looks_like_title(text: str) -> bool:
    t = text.strip()
    if len(t) < 8 or len(t) > 150:
        return False
    if re.search(r"\b(PATENT|OFFENTLIGGJORD|BEVILJAT|PUBLICERAT|KLASS)\b", t, re.I):
        return False
    if re.search(r"\d{4}", t):
        return False
    if t.count(",") > 1:
        return False
    if len(t.split()) < 2:
        return False
    if not t.endswith("."):
        return False
    return True


def looks_like_name_location_line(text: str) -> bool:
    t = text.strip()
    if len(t) < 5 or len(t) > 150:
        return False
    if not t.endswith("."):
        return False
    if "," not in t:
        return False
    if looks_like_title(t):
        return False
    if re.search(r"\b(PATENT|PUBLICERAT|BEVILJAT|PATENTTID|ANS\.)\b", t, re.I):
        return False
    return True


def split_applicant_line(text: str) -> Dict[str, Optional[str]]:
    parts = [p.strip(" .") for p in text.split(",") if p.strip(" .")]
    result = {
        "raw": text.strip(),
        "name": None,
        "city": None,
        "country_text": None,
        "country_code": None,
    }

    if not parts:
        return result

    result["name"] = parts[0].title()

    if len(parts) >= 2:
        result["city"] = ", ".join(parts[1:-1]).title() if len(parts) > 2 else parts[1].title()

    if len(parts) >= 2:
        country_text = parts[-1].upper()
        result["country_text"] = country_text
        result["country_code"] = normalize_country(country_text)

    return result


def extract_publication_number(text: str) -> Optional[str]:
    patterns = [
        r"PATENT\s+N[O°º]?\s*([0-9 ]{3,})",
        r"PATENT\s+NR\.?\s*([0-9 ]{3,})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return re.sub(r"\s+", "", m.group(1))
    return None


def extract_application_number(text: str) -> Optional[str]:
    patterns = [
        r"nr\s+([0-9]+/[0-9]{4})",
        r"nr\.?\s+([0-9]+/[0-9]{4})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return m.group(1)
    return None


def extract_filing_date(text: str) -> Optional[str]:
    patterns = [
        r"Ans\.\s*den\s*(\d{1,2})[\/\- ](\d{1,2})[\/\- ](\d{4})",
        r"Ans\.\s*den\s*(\d{1,2})\s+([A-ZÅÄÖ]+\.?)\s+(\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            if re.match(r"\d{1,2}", m.group(2)):
                return parse_numeric_date(m.group(1), m.group(2), m.group(3))
            return parse_swedish_date(m.group(1), m.group(2), m.group(3))
    return None


def extract_publication_date(text: str) -> Optional[str]:
    patterns = [
        r"PUBLICERAT\s+DEN\s+(\d{1,2})\s+([A-ZÅÄÖ]+\.?)\s+(\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return parse_swedish_date(m.group(1), m.group(2), m.group(3))
    return None


def extract_effective_date(text: str) -> Optional[str]:
    patterns = [
        r"PATENTTID\s+FR[ÅA]N\s+DEN\s+(\d{1,2})[\/\- ](\d{1,2})[\/\- ](\d{4})",
        r"PATENTTID\s+FR[ÅA]N\s+DEN\s+(\d{1,2})\s+([A-ZÅÄÖ]+\.?)\s+(\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            if re.match(r"\d{1,2}", m.group(2)):
                return parse_numeric_date(m.group(1), m.group(2), m.group(3))
            return parse_swedish_date(m.group(1), m.group(2), m.group(3))
    return None


def extract_document_code_from_filename(pdf_path: str) -> Optional[str]:
    m = re.search(r"\.([A-Z]\d)\.pdf$", pdf_path, re.I)
    if m:
        return m.group(1).upper()
    return None


def find_title_and_applicant(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    top_lines = [ln for ln in lines if ln["y0"] < 260]

    title_idx = None
    title = None

    for i, line in enumerate(top_lines):
        if looks_like_title(line["text"]):
            title_idx = i
            title = {
                "value": line["text"].rstrip("."),
                "confidence": 0.85,
                "method": "standalone_title_line",
                "page": line["page"],
            }
            break

    applicant = None
    inventor_candidates = []

    if title_idx is not None:
        for j in range(title_idx - 1, -1, -1):
            candidate = top_lines[j]["text"]
            if looks_like_name_location_line(candidate):
                app = split_applicant_line(candidate)
                applicant = {
                    "value": app,
                    "confidence": 0.82,
                    "method": "line_immediately_above_title",
                    "page": top_lines[j]["page"],
                }
                break

        for j in range(title_idx + 1, min(title_idx + 6, len(top_lines))):
            candidate = top_lines[j]["text"]
            if re.search(r"\b(uppfinnare|original|inventor)\b", candidate, re.I):
                inventor_candidates.append({
                    "value": candidate,
                    "confidence": 0.70,
                    "method": "labeled_inventor_candidate",
                    "page": top_lines[j]["page"],
                })

    return {
        "title": title,
        "applicant": applicant,
        "inventor_candidates": inventor_candidates,
    }


def classify_layout(text: str) -> str:
    if re.search(r"\(\s*54\s*\)|\(\s*72\s*\)|\(\s*73\s*\)", text):
        return "B_INID"
    return "C_OLD"


def extract_patent_record(pdf_path: str) -> Dict[str, Any]:
    lines = extract_page_lines(pdf_path, page_number=0)
    lines = merge_nearby_lines(lines)
    page_text = first_page_text(lines)

    layout_type = classify_layout(page_text)
    title_app = find_title_and_applicant(lines)

    record = {
        "source_file": pdf_path,
        "layout_type": layout_type,
        "document_id": {
            "country_code": "SE" if "SVERIGE" in page_text.upper() else None,
            "publication_number": extract_publication_number(page_text),
            "document_code": extract_document_code_from_filename(pdf_path),
            "publication_date": extract_publication_date(page_text),
        },
        "application_id": {
            "country_code": "SE" if "SVERIGE" in page_text.upper() else None,
            "application_number": extract_application_number(page_text),
            "filing_date": extract_filing_date(page_text),
            "effective_date": extract_effective_date(page_text),
        },
        "priorities": [],
        "applicants": [],
        "inventors": [],
        "title": title_app["title"],
        "debug": {
            "top_lines": [ln["text"] for ln in lines[:20]]
        }
    }

    if title_app["applicant"]:
        record["applicants"].append(title_app["applicant"])

    for cand in title_app["inventor_candidates"]:
        record["inventors"].append(cand)

    return record


if __name__ == "__main__":
    pdf_path = "raw_pdfs/SE105337.C1.pdf"   # change this to your file path
    result = extract_patent_record(pdf_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
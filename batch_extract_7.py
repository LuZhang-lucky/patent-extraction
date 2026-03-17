import os
import json
import fitz
import requests
import time

OLLAMA_URL = "http://192.168.10.236:11434/api/chat"
MODEL_NAME = "qwen3:8b"

PDF_DIR = "raw_pdfs"
RESULT_DIR = "results_7"

MAX_FILES = 10


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    text = page.get_text("text")
    return f"--- PAGE 1 ---\n{text}"


def build_prompt(filename, patent_text):
    prompt = f"""
You are extracting bibliographic information from a historical Swedish patent document.

Return ONLY valid JSON.
Do not include explanations.
Do not include reasoning.
Do not include markdown.

Use exactly these fields:

{{
  "publiceringsnummer": null,
  "publiceringsdatum": null,
  "dokumentkod": null,
  "ansökningsnummer": null,
  "ansökningsdatum": null,
  "giltighetsdatum": null,
  "patenthavare": [],
  "uppfinnare": [],
  "benämning": null,
  "klass": null,
  "IPC": null
}}

Field definitions:
- publiceringsnummer: publication number of the patent document, often shown near "PATENT N".
- publiceringsdatum: publication date in YYYYMMDD format, often introduced by "PUBLICERAT DEN".
- dokumentkod: document code such as C1 or B. If the filename contains a suffix like ".C1.pdf", use "C1" only.
- ansökningsnummer: application number, often shown near "Ans. den" and "nr".
- ansökningsdatum: application filing date in YYYYMMDD format.
- giltighetsdatum: effective date in YYYYMMDD format, often introduced by "PATENTTID FRÅN DEN".
- patenthavare: list of patent holders/applicants. In historical PRV patents, the unlabeled line above the title is often the patent holder.
- uppfinnare: list of inventors. Only extract if clearly stated. Do not guess.
- benämning: patent title.
- klass: original patent classification if explicitly present.
- IPC: IPC classification if explicitly present, often marked as "Int. Cl.". May be absent in older patents.

Important rules:
- Dates in historical Swedish patents are usually in day-month-year order, not month-day-year.
- Example: "11/3 1941" means 11 March 1941 = 19410311.
- If a value is missing or unclear, use null.
- Do not confuse publiceringsnummer with ansökningsnummer.
- Do not invent values.
- patenthavare must always be a list of strings.
- uppfinnare must always be a list of strings.
- benämning must be a single string.
- If klass or IPC is not clearly present, use null.
- Return JSON only.

Filename:
{filename}

Patent text:
{patent_text}
"""
    return prompt.strip()


def ask_model(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "think": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=900)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"]


def save_result(filename, raw_response):
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_path = os.path.join(RESULT_DIR, filename.replace(".pdf", ".json"))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(raw_response)

    return output_path


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])[:MAX_FILES]

    print(f"Found {len(pdf_files)} PDF files.")

    for i, filename in enumerate(pdf_files, start=1):
        try:
            pdf_path = os.path.join(PDF_DIR, filename)
            print(f"\n[{i}/{len(pdf_files)}] Processing {filename}")

            patent_text = extract_text_from_pdf(pdf_path)
            prompt = build_prompt(filename, patent_text)
            raw_response = ask_model(prompt)
            output_path = save_result(filename, raw_response)

            print(f"Saved to {output_path}")

            # small pause so we don't overload the server
            time.sleep(2)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()
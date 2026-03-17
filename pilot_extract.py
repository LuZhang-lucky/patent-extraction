import os
import json
import fitz
import requests
import time

# ===== 1. SETTINGS =====
OLLAMA_URL = "http://192.168.10.236:11434/api/chat"
MODEL_NAME = "qwen3:8b"

PDF_DIR = "raw_pdfs2"
RESULT_DIR = "results2"

# Change this to the file you want to test first
#TEST_FILE = "SE184451.C1.pdf"


# ===== 2. READ PDF TEXT =====
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]

    text = page.get_text("text")

    # only keep the top of the page
    lines = text.split("\n")
    text = "\n".join(lines[:40])

    return text


# ===== 3. BUILD PROMPT =====
def build_prompt(filename, patent_text):

    prompt = f"""
You are extracting bibliographic information from a historical Swedish patent document (PRV).

Return ONLY valid JSON.
Do not include explanations, reasoning, or markdown.

Use exactly this schema:

{{
  "publiceringsnummer": null,
  "publiceringsdatum": null,
  "utläggningsdatum": null,
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

publiceringsnummer
Publication number, often shown near "PATENT N".

publiceringsdatum
Publication date of the patent document, usually introduced by
"PUBLICERAT DEN".

utläggningsdatum
Date when the application was laid open for public inspection,
usually introduced by "UTLAGD DEN".

dokumentkod
Document code such as C or C1. If the filename contains ".C1.pdf",
extract "C1".

ansökningsnummer
Application number. Often appears near "Ans." or before "inkom den".

ansökningsdatum
Application filing date, usually introduced by "inkom den" or
"Ans. den".

giltighetsdatum
Effective date, usually introduced by "PATENTTID FRÅN".

patenthavare
Patent holder or applicant. In many Type C patents the line
immediately above the title contains the patent holder.

uppfinnare
Inventor names if clearly stated.

benämning
Patent title.

klass
Swedish national patent classification.

IPC
International Patent Classification.

Important rules:

Dates in historical Swedish patents are usually written in
day-month-year order.

Example:
11/3 1941 → 19410311

Some late Type C patents use Roman numerals for months.

Examples:
17 VI 1959 → 19590617
31 X 1966 → 19661031

Convert Roman numerals (I–XII) to numeric months.

If a value is missing, unclear, or partly unreadable,
return null instead of guessing.

Do not confuse publiceringsnummer with ansökningsnummer.

Do not invent values.

patenthavare must always be a list of strings.

uppfinnare must always be a list of strings.

benämning must be a single string.

For publiceringsnummer, extract only the number.
Remove words such as "PATENT".
Remove spaces.

Example:
"PATENT 184 451" → "184451"

For ansökningsnummer, extract only the numeric application number.

Remove labels such as:
"Ans."
"Ans. nr"
"P.ans.nr"

Example:
"Ans. 9296/1956" → "9296/1956"

Inventor extraction rule:

Only extract uppfinnare if the word "UPPFINNARE" appears explicitly
in the document.

If the document does not contain the label "UPPFINNARE",
set "uppfinnare" to an empty list.

Do not infer inventors from the applicant or other names.

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

Return JSON only.

Filename:
{filename}

Patent text:
{patent_text}
"""

    return prompt.strip()




# ===== 4. SEND TO MODEL =====
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

    # Ollama chat response text is usually here
    content = data["message"]["content"]
    return content


# ===== 5. SAVE RESULT =====
def save_result(filename, raw_response):

    os.makedirs(RESULT_DIR, exist_ok=True)

    output_path = os.path.join(
        RESULT_DIR,
        filename.replace(".pdf", ".json")
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(raw_response)

    return output_path



# ===== 6. MAIN =====
def main():

    pdf_folder = PDF_DIR
    output_folder = RESULT_DIR

    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    print(f"Found {len(pdf_files)} PDF files.")

    for filename in pdf_files:

        pdf_path = os.path.join(pdf_folder, filename)

        print(f"\nProcessing {filename}")

        text = extract_text_from_pdf(pdf_path)

        prompt = build_prompt(filename, text)

        raw_response = ask_model(prompt)

        output_path = save_result(filename, raw_response)

        print(f"Saved result → {output_path}")

        time.sleep(2)
    
if __name__ == "__main__":
    main()    
    
    
    
    
    
    
    
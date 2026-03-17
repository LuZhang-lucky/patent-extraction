import os
import json
import fitz
import requests
import time
import re

# ===== 1. SETTINGS =====
OLLAMA_URL = "http://192.168.10.236:11434/api/chat"
MODEL_NAME = "qwen3:8b"

PDF_DIR = "raw_pdfs2"
RESULT_DIR = "results_20_rename_2"


# ===== 2. READ PDF TEXT =====
def extract_text_from_pdf(pdf_path, max_lines=40):
    doc = fitz.open(pdf_path)
    page = doc[0]

    text = page.get_text("text")

    # keep only the top part of page 1
    lines = text.split("\n")
    text = "\n".join(lines[:max_lines])

    return text


# ===== 3. BUILD PROMPT =====
def build_prompt(filename, patent_text):
    prompt = f"""
You are extracting bibliographic metadata from a historical Swedish patent document (PRV).

Return ONLY valid JSON.
Do not include explanations, reasoning, markdown, or comments.
Do not return any text before or after the JSON.

Use exactly this JSON structure:

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

Field definitions:

publiceringsnummer
Publication number, often shown near "PATENT N".
Extract only the number.
Remove words such as "PATENT N".
Remove spaces.

dokumentkod
Document code such as C or B.
If the filename contains ".C1.pdf", extract "C1".

publiceringsdatum
Publication date of the patent document, usually introduced by
"PUBLICERAT DEN".

ansokningsnummer
Application number.
Often appears near "Ans." or before "inkom den".
Remove labels such as:
"Ans."
"Ans. nr"
"P.ans.nr"

Example:
"Ans. 9296/1956" -> "9296/1956"

ansokningsdatum
Application filing date, usually introduced by
"inkom den" or "Ans. den".

patenttid_fr
This is the effective date introduced specifically by the phrase:
"PATENTTID FRÅN DEN"

prioritetsdatum
Priority date if explicitly present.

prioritetsnummer
Priority number if explicitly present.

sokande
Applicant / patent holder.
Return as a list of objects:
[
  {{
    "namn": "...",
    "landskod": "..."
  }}
]

If country code is missing, set landskod to null.

uppfinnare
Inventor names.

Return as a list of strings.

Example:
["C W Schroeder", "ME Doyle"]

Only extract uppfinnare if the word "UPPFINNARE" appears explicitly.
If not, return [].

Do not return inventor objects.
Do not include country code for inventors.

Inventor extraction rule:
Only extract uppfinnare if the word "UPPFINNARE" appears explicitly
in the document.
If the document does not contain the label "UPPFINNARE",
return a "null".

Do not infer inventors from applicant names or other names.

benamning
Patent title.
Return as a single string.

Important date rules:

Dates in historical Swedish patents are usually written in day-month-year order.

Example:
11/3 1941 -> 19410311

Some late Type C patents use Roman numerals for months.

Examples:
17 VI 1959 -> 19590617
31 X 1966 -> 19661031

Convert Roman numerals (I-XII) to numeric months.

If a date is unclear or partly unreadable, return null instead of guessing.

Classification rules:

Only extract classification values from the section labeled "KLASS" or "Klass".

Fields:
- IPC = international classification
- klass = Swedish national classification

Layout rules:
1. Earlier layout:
   - If only "KLASS" appears and a code is on the right side of the label,
     that right-side code is Swedish classification.

2. Later layout:
   - "INTERNATIONELL" -> IPC
   - "SVENSK" -> Swedish classification
   - IPC is usually below or immediately after "INTERNATIONELL"
   - Swedish classification is usually below or immediately after "SVENSK"
   - * code starting with A-H -> IPC
   - * code starting with a number -> klass

3. Later layout with only "KLASS":
   - if only one code appears below "KLASS" and it starts with a letter A-H,
     treat it as IPC

Format rules:
- IPC begins with a letter A-H
- Swedish classification begins with a number

Normalization rules:
- Normalize IPC by removing spaces and converting letters to uppercase
  Example:
  "C 08 F 279/02" -> "C08F279/02"
  "A41b9/02" -> "A41B9/02"
- Preserve Swedish classification as printed, except trim extra spaces

Examples:
"A41b9/02 3a:8/01" ->
{{
   {{
    "IPC": "A41B9/02",
    "klass": "3a:8/01"
  }}
}}

Restrictions:
- Do not guess missing values
- Do not infer metadata from technical text
- Do not swap IPC and klass
- If only IPC is present, set klass to null
- If only Swedish classification is present, set IPC to null

Output rules:
- Return JSON only
- Use null for missing scalar values
- Use [] for missing sokande or uppfinnare

Priority information rules:

Extract priority information only when it is explicitly stated.

Typical labels or phrases:
- "Prioritet begärd från den ..."

Location rule:
- Priority information is usually located below the inventor(uppfinnare) section.
- If no inventor section is present, it is usually located below the title (benämning).

Extract:
- prioritetsdatum
- prioritetsland
- prioritetsnummer

Date rules:
- Priority dates are written in day-month-year order.
- Normalize to YYYYMMDD.
- The month may be written as:
  - a Roman numeral, for example: 25 I 1960, 17 VI 1959, 31 X 1966
- Convert Roman numerals I-XII to numeric months.
- OCR may confuse "I" and "1"; interpret them by date context.

Priority number rules:
- If a country code appears in parentheses, such as "(US ####)", extract only the numeric part as the priority number.
- Example:
  "Prioritet begärd från den 25 I 1960 (US ####)"
  -> prioritetsdatum = "19600125"
  -> prioritetsland = "US"
  -> prioritetsnummer = "####"

Restrictions:
- Do not infer priority information from other identifiers.
- If no priority number is clearly present in a priority statement, return null.
- Never generate or hallucinate a number.
- Before returning the priority number, verify that the exact number appears in the document text.
- If the number cannot be directly located in the text, return null.

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
    content = data["message"]["content"]
    return content


def extract_json_from_response(raw_response):
    raw_response = raw_response.strip()

    # 1) try direct parse
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    # 2) extract JSON block
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found in model response.")

    json_text = match.group(0)

    # 3) remove bad control characters
    json_text = re.sub(r'[\x00-\x1F]', ' ', json_text)

    # 4) normalize whitespace a bit
    json_text = re.sub(r'\s+', ' ', json_text)

    # 5) try parse again
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        with open("debug_bad_json.txt", "w", encoding="utf-8") as f:
            f.write(json_text)
        raise




# ===== 6. SAVE RESULT =====
def save_result(filename, parsed_json):
    os.makedirs(RESULT_DIR, exist_ok=True)

    output_path = os.path.join(
        RESULT_DIR,
        filename.replace(".pdf", ".json")
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, ensure_ascii=False, indent=2)

    return output_path


# ===== 7. MAIN =====
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files.")

    for filename in pdf_files:
        pdf_path = os.path.join(PDF_DIR, filename)
        print(f"\nProcessing {filename}")
        
        raw_response = None


        try:
            text = extract_text_from_pdf(pdf_path, max_lines=60)
            prompt = build_prompt(filename, text)
            raw_response = ask_model(prompt)
            parsed_json = extract_json_from_response(raw_response)
            output_path = save_result(filename, parsed_json)
            print(f"Saved result -> {output_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
            if raw_response:
                error_path = os.path.join(
                    RESULT_DIR,
                    filename.replace(".pdf", "_raw_error.txt")
                )
                with open(error_path, "w", encoding="utf-8") as f:
                    f.write(raw_response)
                print(f"Saved raw response -> {error_path}")
        time.sleep(2)
          


if __name__ == "__main__":
    main()





import os
import pdfplumber
from PIL import Image
import pytesseract

# Input folder containing PDFs
input_folder = "raw_pdfs"
# Output folder for TXT files
output_folder = "txt_output"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all PDF files
for pdf_file in os.listdir(input_folder):
    if pdf_file.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_folder, pdf_file)
        output_txt_path = os.path.join(output_folder, pdf_file.replace(".pdf", ".txt"))

        print(f"Processing: {pdf_file}")
        full_text = ""

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Try extracting text normally
                text = page.extract_text()
                
                if text:
                    full_text += text + "\n\n"
                else:
                    # No text layer → run OCR
                    image = page.to_image(resolution=300).original
                    ocr_text = pytesseract.image_to_string(image, lang="swe")
                    full_text += ocr_text + "\n\n"

        # Save extracted text to TXT file
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"Saved text to {output_txt_path}\n")

print("All PDFs processed!")